# %%
# MAE-ConvNeXt (32x32-friendly): masked at deep stage, pre/post masked convs,
# configurable strides, and U-Net-ish decoder. Drop-in replacement for MAE-ResNet.

from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Optional helpers / fallbacks (so this file is standalone)
# -------------------------
from einops import rearrange
from utils.feats import unfold_feats, group_features
def _rand_mask_grid(B: int, H: int, W: int, ratio: float, device, dtype) -> torch.Tensor:
    """Randomly mask K=ratio*HW cells on an HxW grid: [B,1,H,W] with 1=masked."""
    K = int(round(ratio * H * W))
    out = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
    if K == 0:
        return out
    for b in range(B):
        idx = torch.randperm(H * W, device=device)[:K]
        out.view(B, 1, -1)[b, 0, idx] = 1.0
    return out


# -------------------------
# Core building blocks (ConvNeXt-V2 style)
# -------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = x.new_empty(shape).bernoulli_(keep)
        return x * rand / keep


class LayerNorm2d(nn.Module):
    """LN over channels for BCHW."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class GRN(nn.Module):
    """
    Global Response Normalization (ConvNeXt-V2).
    Y = gamma * (X * nx) + beta + X, with nx from L2 over spatial dims and channel-mean normalization.
    Paper pseudocode Alg.1; used after expansion MLP. :contentReference[oaicite:1]{index=1}
    """
    def __init__(self, num_channels: int, eps: float = 1e-6, scale_by_channels: bool = True):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps
        self.scale_by_channels = scale_by_channels
        self.num_channels = num_channels
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.linalg.vector_norm(x, ord=2, dim=(2, 3), keepdim=True)
        den = gx.mean(dim=1, keepdim=True) + self.eps
        nx = gx / den
        if self.scale_by_channels:
            nx = nx * (self.num_channels ** 0.5)
        return self.gamma * (x * nx) + self.beta + x


class MaskedConv2d(nn.Conv2d):
    """
    Dense conv with pre/post masking:
      - mask_in  zeros input at masked sites (no reading from masked)
      - mask_out zeros output at masked sites (no writing to masked)
    Equivalent in effect to sparse conv gating mentioned in FCMAE (but dense). :contentReference[oaicite:2]{index=2}
    """
    def forward(self, x: torch.Tensor, mask_in: Optional[torch.Tensor] = None, mask_out: Optional[torch.Tensor] = None):
        if mask_in is not None:
            x = x * (1.0 - mask_in)
        y = super().forward(x)
        m = mask_out if mask_out is not None else mask_in
        if m is not None:
            y = y * (1.0 - m)
        return y


class ConvNeXtBlockMasked(nn.Module):
    """
    ConvNeXt block with depthwise k=7, GELU MLP (1x1 convs), optional GRN.
    All convs are masked with the same per-stage mask (stride=1).
    """
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop_path: float = 0.0, use_grn: bool = True,
                 layer_scale_init_value: float = 0.0):
        super().__init__()
        self.dwconv = MaskedConv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.norm   = LayerNorm2d(dim, eps=1e-6)
        hidden_dim  = int(dim * mlp_ratio)
        self.pw1    = MaskedConv2d(dim, hidden_dim, kernel_size=1, bias=True)
        self.act    = nn.GELU()
        self.grn    = GRN(hidden_dim) if use_grn else nn.Identity()
        self.pw2    = MaskedConv2d(hidden_dim, dim, kernel_size=1, bias=True)
        self.gamma  = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None
        self.drop   = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x, mask_in=mask, mask_out=mask)
        x = self.norm(x)
        x = self.pw1(x, mask_in=mask, mask_out=mask)
        x = self.act(x)
        x = self.grn(x)
        x = self.pw2(x, mask_in=mask, mask_out=mask)
        if self.gamma is not None:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = shortcut + self.drop(x)
        if mask is not None:
            x = x * (1.0 - mask)
        return x


class StemPatchify(nn.Module):
    """Stem conv to stage-1 with configurable stride (default 4)."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 4):
        super().__init__()
        self.stride = stride
        self.conv = MaskedConv2d(in_ch, out_ch, kernel_size=stride, stride=stride, bias=True)
    def forward(self, x, mask_in, mask_out):
        return self.conv(x, mask_in=mask_in, mask_out=mask_out)


class DownsampleLayer(nn.Module):
    """Downsample between stages with configurable stride (default 2)."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.norm = LayerNorm2d(in_ch, eps=1e-6)
        k = 2 if stride == 2 else 1
        self.conv = MaskedConv2d(in_ch, out_ch, kernel_size=k, stride=stride, bias=True)
        self.stride = stride
    def forward(self, x, mask_in, mask_out):
        x = self.norm(x)
        return self.conv(x, mask_in=mask_in, mask_out=mask_out)


# -------------------------
# Encoder (ConvNeXt V2 style) with masked convs + adaptive mask base
# -------------------------
class ConvNeXtEncoderMasked(nn.Module):
    """
    Hierarchy: stage1 (H/s), stage2 (H/s/2), stage3 (H/s/4), stage4 (H/s/8) for s=stem_stride when ds_strides=(2,2,2).
    Every conv is pre/post masked.
    """
    def __init__(
        self,
        in_channels: int = 3,
        depths: Tuple[int, int, int, int] = (3, 3, 9, 3),
        dims: Tuple[int, int, int, int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        use_grn: bool = True,
        stem_stride: int = 4,
        ds_strides: Tuple[int, int, int] = (2, 2, 1),  # <- keep bottom not-too-small for 32x32
    ):
        super().__init__()
        self.stem_stride = stem_stride
        self.ds_strides = ds_strides

        self.stem = StemPatchify(in_channels, dims[0], stride=stem_stride)

        # stochastic-depth schedule
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        idx = 0
        def make_stage(dim: int, depth: int):
            nonlocal idx
            blocks = []
            for _ in range(depth):
                blocks.append(ConvNeXtBlockMasked(dim, drop_path=dpr[idx], use_grn=use_grn))
                idx += 1
            return nn.Sequential(*blocks)

        self.stage1 = make_stage(dims[0], depths[0])
        self.down12 = DownsampleLayer(dims[0], dims[1], stride=ds_strides[0])
        self.stage2 = make_stage(dims[1], depths[1])
        self.down23 = DownsampleLayer(dims[1], dims[2], stride=ds_strides[1])
        self.stage3 = make_stage(dims[2], depths[2])
        self.down34 = DownsampleLayer(dims[2], dims[3], stride=ds_strides[2])
        self.stage4 = make_stage(dims[3], depths[3])

        self.out_channels = {
            "stage1": dims[0],
            "stage2": dims[1],
            "stage3": dims[2],
            "stage4": dims[3],
        }

    # ---- shapes and mask pyramid ----
    def _stage_shapes(self, H: int, W: int):
        s1 = self.stem_stride
        d1, d2, d3 = self.ds_strides
        H1, W1 = H // s1, W // s1
        H2, W2 = H1 // d1, W1 // d1
        H3, W3 = H2 // d2, W2 // d2
        H4, W4 = H3 // d3, W3 // d3
        return (H1, W1), (H2, W2), (H3, W3), (H4, W4)

    @torch.no_grad()
    def _make_pyramid_masks(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        mask_lowest: Optional[torch.Tensor] = None,  # only used if base_from='stage4'
        base_from: str = "auto",                    # "auto" | "stage1".."stage4" | "img"
        min_base_grid: int = 4,                     # ensure the base grid isn't too tiny
        dilate_kernel: int = 1,                     # >1 to grow masks slightly (optional)
    ) -> Dict[str, torch.Tensor]:
        B, _, H, W = x.shape
        (H1, W1), (H2, W2), (H3, W3), (H4, W4) = self._stage_shapes(H, W)
        levels = [("stage4", (max(1, H4), max(1, W4))),
                  ("stage3", (max(1, H3), max(1, W3))),
                  ("stage2", (max(1, H2), max(1, W2))),
                  ("stage1", (max(1, H1), max(1, W1))),
                  ("img",    (H, W))]
        size_of = dict(levels)

        # choose base level
        if base_from == "auto":
            base_name, base_hw = None, None
            for name, (h, w) in levels:  # deepest-first
                if h >= min_base_grid and w >= min_base_grid:
                    base_name, base_hw = name, (h, w)
                    break
            if base_name is None:  # if *everything* is tiny
                base_name, base_hw = "img", (H, W)
        else:
            base_name, base_hw = base_from, size_of[base_from]

        # base mask
        if mask_lowest is not None and base_name == "stage4":
            base = mask_lowest
            assert base.shape == (B, 1, base_hw[0], base_hw[1]), "mask_lowest must match stage4"
        else:
            base = _rand_mask_grid(B, base_hw[0], base_hw[1], mask_ratio, x.device, x.dtype)

        # helpers (binary-preserving)
        def up(m, hw):   return F.interpolate(m, size=hw, mode="nearest")
        def down(m, hw): return F.adaptive_max_pool2d(m, output_size=hw)

        M: Dict[str, torch.Tensor] = {}
        for name, hw in levels:
            if hw == base_hw:
                m = base
            elif hw[0] * hw[1] > base_hw[0] * base_hw[1]:
                m = up(base, hw)
            else:
                m = down(base, hw)
            if dilate_kernel and dilate_kernel > 1:
                m = torch.clamp(F.max_pool2d(m, kernel_size=dilate_kernel,
                                             stride=1, padding=dilate_kernel // 2), 0, 1)
            M[name] = m
        return M

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        mask_lowest: Optional[torch.Tensor] = None,
        base_from: str = "auto",
        min_base_grid: int = 4,
        dilate_kernel: int = 1,
        return_masks: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        masks = self._make_pyramid_masks(
            x, mask_ratio=mask_ratio, mask_lowest=mask_lowest,
            base_from=base_from, min_base_grid=min_base_grid, dilate_kernel=dilate_kernel
        )

        feats: Dict[str, torch.Tensor] = {}

        # stem -> stage1
        x1 = self.stem(x, mask_in=masks["img"], mask_out=masks["stage1"])
        for blk in self.stage1:
            x1 = blk(x1, mask=masks["stage1"])
        feats["stage1"] = x1

        # stage2
        x2 = self.down12(x1, mask_in=masks["stage1"], mask_out=masks["stage2"])
        for blk in self.stage2:
            x2 = blk(x2, mask=masks["stage2"])
        feats["stage2"] = x2

        # stage3
        x3 = self.down23(x2, mask_in=masks["stage2"], mask_out=masks["stage3"])
        for blk in self.stage3:
            x3 = blk(x3, mask=masks["stage3"])
        feats["stage3"] = x3

        # stage4
        x4 = self.down34(x3, mask_in=masks["stage3"], mask_out=masks["stage4"])
        for blk in self.stage4:
            x4 = blk(x4, mask=masks["stage4"])
        feats["stage4"] = x4

        return (feats, masks) if return_masks else (feats, None)


# -------------------------
# Decoder (U-Net-ish; unmasked to inpaint)
# -------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.bn(self.conv(x)))

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.proj   = ConvBNReLU(in_ch + skip_ch, out_ch, k=3, s=1, p=1)
        self.refine = ConvBNReLU(out_ch, out_ch, k=3, s=1, p=1)
    def forward(self, x, skip: torch.Tensor):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        x = self.refine(x)
        return x

class UNetDecoderConvNeXt(nn.Module):
    def __init__(self, enc_channels: Dict[str, int], out_channels: int, stem_stride: int = 4):
        super().__init__()
        self.stem_stride = stem_stride
        c1 = enc_channels["stage1"]
        c2 = enc_channels["stage2"]
        c3 = enc_channels["stage3"]
        c4 = enc_channels["stage4"]

        self.bridge = ConvBNReLU(c4, c4, 3, 1, 1)
        self.up43   = UpBlock(c4, c3, c3)
        self.up32   = UpBlock(c3, c2, c2)
        self.up21   = UpBlock(c2, c1, c1)
        self.head   = nn.Conv2d(c1, out_channels, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feats: Dict[str, torch.Tensor], target_hw: Optional[Tuple[int,int]] = None) -> torch.Tensor:
        x = self.bridge(feats["stage4"])
        x = self.up43(x, feats["stage3"])
        x = self.up32(x, feats["stage2"])
        x = self.up21(x, feats["stage1"])
        out = self.head(x)
        # Nominal up to input size based on stem stride
        H0 = feats["stage1"].shape[-2] * self.stem_stride
        W0 = feats["stage1"].shape[-1] * self.stem_stride
        out = F.interpolate(out, size=(H0, W0), mode="bilinear", align_corners=False)
        if target_hw is not None and out.shape[-2:] != target_hw:
            out = F.interpolate(out, size=target_hw, mode="bilinear", align_corners=False)
        return out


# -------------------------
#       MAE-ConvNeXt
# -------------------------
class MAEConvNeXt(nn.Module):
    """
    ConvNeXt-V2 encoder with GRN + masked conv gating, unmasked decoder, and CE + masked MSE loss.
    Mask is generated at the deepest adequate stage and propagated across scales. :contentReference[oaicite:3]{index=3}
    """
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        depths: Tuple[int, int, int, int] = (3, 3, 9, 3),
        dims: Tuple[int, int, int, int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        use_grn: bool = True,
        stem_stride: int = 4,
        ds_strides: Tuple[int, int, int] = (2, 2, 1),  # 32x32-friendly default
    ):
        super().__init__()
        self.in_channels = in_channels
        self.encoder = ConvNeXtEncoderMasked(
            in_channels=in_channels,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            use_grn=use_grn,
            stem_stride=stem_stride,
            ds_strides=ds_strides,
        )
        top_ch = self.encoder.out_channels["stage4"]

        # Classification head: masked global average pooling over stage-4 visibles
        self.fc = nn.Linear(top_ch, num_classes)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

        self.decoder = UNetDecoderConvNeXt(
            enc_channels=self.encoder.out_channels,
            out_channels=in_channels,
            stem_stride=stem_stride,
        )

    # ---------- Core interface ----------
    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        lambda_cls: float,
        mask_ratio: float = 0.75,
        mask_lowest: Optional[torch.Tensor] = None,  # [B,1,H4,W4] if base_from='stage4'
        mask_base_from: str = "auto",
        min_base_grid: int = 4,
        dilate_kernel: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert 0.0 <= lambda_cls <= 1.0, "lambda_cls must be in [0,1]"
        feats, masks = self.encoder(
            x, mask_ratio=mask_ratio, mask_lowest=mask_lowest,
            base_from=mask_base_from, min_base_grid=min_base_grid,
            dilate_kernel=dilate_kernel, return_masks=True
        )

        # masked pooling @ stage-4
        top = feats["stage4"]      # [B,C,H4,W4]
        m4  = masks["stage4"]      # [B,1,H4,W4]  (1 = masked)
        visible = (1.0 - m4)
        vis_area = visible.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        pooled = (top * visible).sum(dim=(2, 3), keepdim=True) / vis_area
        logits = self.fc(pooled.flatten(1))

        # decoder to input resolution
        recon = self.decoder(feats, target_hw=x.shape[-2:])

        # losses (masked MSE in image space)
        cls_loss = F.cross_entropy(logits, labels)
        mse = (recon - x) ** 2
        mask_img = masks["img"]
        masked_mse = (mse * mask_img).sum() / (mask_img.sum() + 1e-8)
        zero_recon_loss = (x ** 2 * mask_img).sum() / (mask_img.sum() + 1e-8)

        loss = lambda_cls * cls_loss + (1.0 - lambda_cls) * masked_mse

        with torch.no_grad():
            acc = (logits.argmax(dim=1) == labels).float().mean()
            info = {
                "loss": loss.detach(),
                "cls_loss": cls_loss.detach(),
                "recon_loss": masked_mse.detach(),
                "zero_recon_loss": zero_recon_loss.detach(),
                "acc": acc,
                "mask_ratio": torch.as_tensor(mask_ratio, device=x.device, dtype=x.dtype),
                "stage1_std": feats["stage1"].std().mean(),
                "stage2_std": feats["stage2"].std().mean(),
                "stage3_std": feats["stage3"].std().mean(),
                "stage4_std": feats["stage4"].std().mean(),
            }
        return loss, info

    @torch.no_grad()
    def reconstruct(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.75,
        mask_lowest: Optional[torch.Tensor] = None,
        mask_base_from: str = "auto",
        min_base_grid: int = 4,
        dilate_kernel: int = 1,
        only_masked: bool = False,
    ) -> torch.Tensor:
        feats, masks = self.encoder(
            x, mask_ratio=mask_ratio, mask_lowest=mask_lowest,
            base_from=mask_base_from, min_base_grid=min_base_grid,
            dilate_kernel=dilate_kernel, return_masks=True
        )
        rec = self.decoder(feats, target_hw=x.shape[-2:])
        if only_masked:
            rec = x * (1.0 - masks["img"]) + rec * masks["img"]
        return rec

    # ---------- Activations API (parity with your LatentResNet.activations) ----------
    def get_activations(
        self,
        x: torch.Tensor,
        num_acti: int = 3,
        min_res: int = 16,
        use_std: bool = False,
        use_mean: bool = False,
        residual_std_mean: bool = False,   # unused; kept for parity
        unfold_kernel_list: List[int] = [],
        patch_group_size: int = 1,
        mask_ratio: float = 0.0,
        mask_base_from: str = "auto",
        min_base_grid: int = 4,
        dilate_kernel: int = 1,
    ) -> Dict[str, torch.Tensor]:
        assert len(x.shape) == 4 and x.shape[-1] == x.shape[-2], "x must be [B,C,H,W] square"
        if x.shape[-1] < min_res:
            x = F.interpolate(x, size=min_res, mode="bicubic")

        feats, _ = self.encoder(
            x, mask_ratio=mask_ratio, mask_lowest=None,
            base_from=mask_base_from, min_base_grid=min_base_grid,
            dilate_kernel=dilate_kernel, return_masks=False
        )
        for x in feats.values():
            print(x.shape)
        res: Dict[str, torch.Tensor] = {}

        def push(name: str, feat: torch.Tensor) -> bool:
            nonlocal res
            res[name] = group_features(feat, patch_group_size)
            if use_mean:
                res[f"{name}_mean"] = rearrange(feat, "b c h w -> b (h w) c").mean(dim=1, keepdim=True)
            if use_std:
                reshaped = rearrange(feat, "b c h w -> b (h w) c")
                res[f"{name}_std"] = reshaped if reshaped.shape[1] == 1 else reshaped.std(dim=1, keepdim=True)
            
            if len(res) >= num_acti:
                keys = list(res.keys())[:num_acti]
                res = {k: res[k] for k in keys}
            return len(res) >= num_acti

        if push("stage1", feats["stage1"]): return res
        if push("stage2", feats["stage2"]): return res
        if push("stage3", feats["stage3"]): return res
        if push("stage4", feats["stage4"]): return res
        return res

    activations = get_activations


# -------------------------
#          Builder & Specs
# -------------------------
def get_convnextv2_spec(arch: str):
    """
    Depths/dims for common ConvNeXt-V2 variants (paper & community configs).
    """
    arch = arch.lower()
    if arch in ("atto", "convnextv2-atto"):
        return (2, 2, 6, 2), (40,  80, 160, 320)
    if arch in ("femto", "convnextv2-femto"):
        return (2, 2, 6, 2), (48,  96, 192, 384)
    if arch in ("pico", "convnextv2-pico"):
        return (2, 2, 6, 2), (64, 128, 256, 512)
    if arch in ("nano", "tiny", "convnextv2-nano"):
        return (3, 3, 9, 3), (96, 192, 384, 768)
    if arch in ("base", "b", "convnextv2-base"):
        return (3, 3, 27, 3), (128, 256, 512, 1024)
    if arch in ("large", "l", "convnextv2-large"):
        return (3, 3, 27, 3), (192, 384, 768, 1536)
    if arch in ("huge", "h", "convnextv2-huge"):
        return (3, 3, 27, 3), (352, 704, 1408, 2816)
    raise ValueError(f"Unknown ConvNeXtV2 arch: {arch}")

def build_mae_convnext(**kwargs) -> MAEConvNeXt:
    """
    Builder with optional checkpoint loading (parity with your LatentResNet builder).
    Args (subset):
        load_dict (dict, optional): {'run_id': str, 'epoch': 'latest'|int, 'load_entry': 'model'|...}
        num_classes (int), in_channels (int), depths/dims, drop_path_rate (float),
        stem_stride (int), ds_strides (tuple of 3 ints), use_grn (bool)
    """
    load_dict = kwargs.pop("load_dict", None)
    model = MAEConvNeXt(**kwargs)
    if load_dict is not None and load_dict.get("run_id", ""):
        from utils.ckpt_utils import load_ckpt_epoch
        from utils.misc import EasyDict
        load_dict = EasyDict(load_dict)
        if not hasattr(load_dict, "load_entry"):
            load_dict.load_entry = "model"
        loaded = load_ckpt_epoch(run_id=load_dict.run_id, epoch=load_dict.epoch)[load_dict.load_entry]
        state = loaded if isinstance(loaded, dict) else loaded.state_dict()
        model.load_state_dict(state, strict=True)
    return model


# -------------------------
#            Demo
# -------------------------
if __name__ == "__main__":
    # Example: 4-channel latent, 32x32. Choose a ConvNeXt-V2 "nano"-like config.
    depths, dims = get_convnextv2_spec("nano")

    # Keep bottom features non-trivial on 32x32:
    # - stem_stride=4, ds_strides=(2,2,1) => stage sizes: 8x8, 4x4, 2x2, 2x2
    model = build_mae_convnext(
        num_classes=1000,
        in_channels=4,
        depths=depths,
        dims=dims,
        drop_path_rate=0.0,
        use_grn=True,
        stem_stride=2,
        ds_strides=(2, 2, 1),   # prevents 1x1 at the bottom for 32x32 inputs
    )

    x = torch.randn(2, 4, 32, 32)
    y = torch.randint(0, 10, (2,))

    # Auto-select deepest base with >=4x4 grid for masking
    loss, info = model(
        x, y, lambda_cls=0.5, mask_ratio=0.6,
        mask_base_from="auto", min_base_grid=4, dilate_kernel=1
    )
    print("loss:", float(info["loss"]), "acc:", float(info["acc"]))

    # Inpaint only masked regions
    recon = model.reconstruct(
        x, mask_ratio=0.6, mask_base_from="auto", min_base_grid=4, only_masked=True
    )
    print("recon:", recon.shape)
    print((recon[0] == x[0]).any(dim=0))

    # Grab a small set of activations (parity with your LatentResNet API)
    acts = model.get_activations(
        x, num_acti=20, use_mean=True, use_std=True, unfold_kernel_list=[3],
        patch_group_size=2, mask_ratio=0.0, mask_base_from="auto", min_base_grid=4
    )
    print({k: v.shape for k, v in acts.items()})

# python -m model.mae_convnext