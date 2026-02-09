# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Union
from einops import rearrange
from utils.persistence import persistent_class
from utils.feats import unfold_feats, group_features
import math

from utils.misc import custom_compile

# ---- GroupNorm utils ----
def _choose_gn_groups(num_channels: int, max_groups: int = 32) -> int:
    """Choose the largest number of groups (<= max_groups) that divides num_channels.
    Fallback to 1 if none divide (behaves similar to LayerNorm across channels).
    """
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return max(g, 1)

# Reuse your BasicBlock if already defined; otherwise keep this in-file.
class BasicBlock(nn.Module):
    """Basic ResNet block for ResNet-18/34."""
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_prob=0.0, gn_max_groups: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.GroupNorm(_choose_gn_groups(out_channels, gn_max_groups), out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.GroupNorm(_choose_gn_groups(out_channels, gn_max_groups), out_channels)
        self.downsample = downsample

    def forward(self, x, return_residual: bool = False):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        residual_branch = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(residual_branch + identity)
        if return_residual:
            return out, residual_branch
        return out


# -------------------------
#      Encoder (ResNet)
# -------------------------
class ResNetEncoder(nn.Module):
    """
    ResNet encoder that produces multi-resolution features:
    conv1 -> layer1 -> layer2 -> layer3 -> layer4
    Each subsequent layer downsamples by stride=2 (except layer1).
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        layers: List[int] = (2, 2, 2, 2),  # ResNet-18 by default
        dropout_prob: float = 0.0,
        block: nn.Module = BasicBlock,
        gn_max_groups: int = 32,
    ):
        super().__init__()
        self.gn_max_groups = gn_max_groups
        self.layers_spec = list(layers)  # keep for external use
        self.inplanes = base_channels
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.GroupNorm(_choose_gn_groups(base_channels, self.gn_max_groups), base_channels)
        self.relu  = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, base_channels,     layers[0], stride=1, dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2, dropout_prob=dropout_prob)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2, dropout_prob=dropout_prob)

        self.out_channels = {
            "conv1": base_channels,
            "layer1": base_channels * block.expansion,
            "layer2": base_channels * 2 * block.expansion,
            "layer3": base_channels * 4 * block.expansion,
            "layer4": base_channels * 8 * block.expansion,
        }

        # Stage-level GroupNorm to keep skip features well-scaled
        self.layer1_norm = nn.GroupNorm(_choose_gn_groups(self.out_channels["layer1"], self.gn_max_groups), self.out_channels["layer1"])
        self.layer2_norm = nn.GroupNorm(_choose_gn_groups(self.out_channels["layer2"], self.gn_max_groups), self.out_channels["layer2"])
        self.layer3_norm = nn.GroupNorm(_choose_gn_groups(self.out_channels["layer3"], self.gn_max_groups), self.out_channels["layer3"])
        self.layer4_norm = nn.GroupNorm(_choose_gn_groups(self.out_channels["layer4"], self.gn_max_groups), self.out_channels["layer4"])

        self._init_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1, dropout_prob=0.0):
        downsample = None
        if stride != 1 or self.inplanes != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(_choose_gn_groups(out_channels * block.expansion, self.gn_max_groups), out_channels * block.expansion),
            )
        layers = [block(self.inplanes, out_channels, stride=stride, downsample=downsample, dropout_prob=dropout_prob, gn_max_groups=self.gn_max_groups)]
        self.inplanes = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, out_channels, dropout_prob=dropout_prob, gn_max_groups=self.gn_max_groups))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_residuals: bool = False, return_block_outputs: bool = False):
        """
        Returns a dict of multi-scale activations for skip connections.
        Keys: 'conv1', 'layer1', 'layer2', 'layer3', 'layer4'

        If return_residuals is True, also returns a dict mapping layer names
        to lists of residual branch tensors (pre-addition) for each block in that layer.
        If return_block_outputs is True, returns a dict mapping layer names
        to lists of block outputs (post-addition+ReLU) for each block in that layer.
        """
        feats: Dict[str, torch.Tensor] = {}
        if return_residuals:
            residuals: Dict[str, List[torch.Tensor]] = {
                "layer1": [],
                "layer2": [],
                "layer3": [],
                "layer4": [],
            }
        if return_block_outputs:
            block_outputs: Dict[str, List[torch.Tensor]] = {
                "layer1": [],
                "layer2": [],
                "layer3": [],
                "layer4": [],
            }

        x = self.relu(self.bn1(self.conv1(x)))   # B, C, H, W
        feats["conv1"] = x

        def run_layer(layer: nn.Sequential, layer_name: str, x_in: torch.Tensor) -> torch.Tensor:
            x_local = x_in
            if return_residuals or return_block_outputs:
                for block in layer:
                    if isinstance(block, BasicBlock):
                        if return_residuals:
                            x_local, res = block(x_local, return_residual=True)
                            residuals[layer_name].append(res)
                        else:
                            x_local = block(x_local)
                    else:
                        x_local = block(x_local)
                    if return_block_outputs:
                        block_outputs[layer_name].append(x_local)
                return x_local
            else:
                return layer(x_local)

        x = run_layer(self.layer1, "layer1", x)
        x = self.layer1_norm(x)
        feats["layer1"] = x
        x = run_layer(self.layer2, "layer2", x)
        x = self.layer2_norm(x)
        feats["layer2"] = x
        x = run_layer(self.layer3, "layer3", x)
        x = self.layer3_norm(x)
        feats["layer3"] = x
        x = run_layer(self.layer4, "layer4", x)
        x = self.layer4_norm(x)
        feats["layer4"] = x

        if return_residuals and return_block_outputs:
            return feats, residuals, block_outputs
        if return_residuals:
            return feats, residuals
        if return_block_outputs:
            return feats, block_outputs
        return feats


# -------------------------
#      Decoder (U-Net)
# -------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, gn_max_groups: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.GroupNorm(_choose_gn_groups(out_ch, gn_max_groups), out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.bn(self.conv(x)))

class UpBlock(nn.Module):
    """
    Upsample by x2, concatenate skip connection, then refine.
    """
    def __init__(self, in_ch, skip_ch, out_ch, gn_max_groups: int = 32):
        super().__init__()
        self.x_norm = nn.GroupNorm(_choose_gn_groups(in_ch + skip_ch, gn_max_groups), in_ch + skip_ch)
        self.proj  = ConvBNReLU(in_ch + skip_ch, out_ch, k=3, s=1, p=1, gn_max_groups=gn_max_groups)
        self.refine = ConvBNReLU(out_ch, out_ch, k=3, s=1, p=1, gn_max_groups=gn_max_groups)

    def forward(self, x, skip: torch.Tensor):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.x_norm(x)
        x = self.proj(x)
        x = self.refine(x)
        return x

class SkipFuseBlock(nn.Module):
    """
    单步“跳连融合”模块：
    - 将当前特征 x 上采样到 skip 的空间分辨率（若需要）
    - 拼接后进行 GN + 2*ConvBNReLU 细化
    - 输出通道数为 out_ch（通常等于该分辨率的 stage 通道数）
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, gn_max_groups: int = 32):
        super().__init__()
        self.in_ch   = in_ch
        self.skip_ch = skip_ch
        self.out_ch  = out_ch
        self.concat_norm = nn.GroupNorm(_choose_gn_groups(in_ch + skip_ch, gn_max_groups), in_ch + skip_ch)
        self.proj   = ConvBNReLU(in_ch + skip_ch, out_ch, k=3, s=1, p=1, gn_max_groups=gn_max_groups)
        self.refine = ConvBNReLU(out_ch, out_ch, k=3, s=1, p=1, gn_max_groups=gn_max_groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.concat_norm(x)
        x = self.proj(x)
        x = self.refine(x)
        return x


class UNetDecoder(nn.Module):
    """
    解码器：
    - skip_ev_k == 0：保持传统 4 个 UpBlock 的对称 U-Net 结构。
    - skip_ev_k > 0：根据各分辨率被抽中的 skip 数动态构建一串 SkipFuseBlock，
      以深->浅顺序依次融合；可跨分辨率直接上采样到目标 skip 尺寸。
    """
    def __init__(
        self,
        enc_channels: Dict[str, int],
        out_channels: int,
        gn_max_groups: int = 32,
        layers_spec: Tuple[int, int, int, int] = (2, 2, 2, 2),
        skip_ev_k: int = 0,
    ):
        super().__init__()
        self.skip_ev_k = int(skip_ev_k)
        self.gn_max_groups = gn_max_groups

        c1 = enc_channels["conv1"]
        c2 = enc_channels["layer1"]
        c3 = enc_channels["layer2"]
        c4 = enc_channels["layer3"]
        c5 = enc_channels["layer4"]

        self._chs = dict(conv1=c1, layer1=c2, layer2=c3, layer3=c4, layer4=c5)

        # Bridge 始终保留
        self.bridge = ConvBNReLU(c5, c5, k=3, s=1, p=1, gn_max_groups=gn_max_groups)

        if self.skip_ev_k == 0:
            # 传统路径
            self.mode = "classic"
            self.up43 = UpBlock(c5, c4, c4, gn_max_groups=gn_max_groups)   # -> layer3 scale
            self.up32 = UpBlock(c4, c3, c3, gn_max_groups=gn_max_groups)   # -> layer2 scale
            self.up21 = UpBlock(c3, c2, c2, gn_max_groups=gn_max_groups)   # -> layer1 scale
            self.up10 = UpBlock(c2, c1, c1, gn_max_groups=gn_max_groups)   # -> conv1 scale
        else:
            # 动态路径：生成一个按“深->浅”顺序的 step 模块列表
            self.mode = "dense"
            l1, l2, l3, l4 = list(layers_spec)  # blocks per layer
            # 计算每个 stage 需要的步数（每 ev_k 个 block 一个 skip）
            m4 = l4 // self.skip_ev_k
            m3 = l3 // self.skip_ev_k
            m2 = l2 // self.skip_ev_k
            m1 = l1 // self.skip_ev_k

            plan: List[Tuple[str, int]] = []
            # layer4 的同分辨率融合 m4 次
            plan += [("layer4", c5)] * m4
            # 接着 layer3 的 m3 次、layer2 的 m2 次、layer1 的 m1 次
            plan += [("layer3", c4)] * m3
            plan += [("layer2", c3)] * m2
            plan += [("layer1", c2)] * m1
            # 最后总是与 conv1 融合 1 次，确保输出分辨率
            plan += [("conv1", c1)]

            self._expected_steps = len(plan)
            steps = []
            current_ch = c5  # 来自 bridge 的通道
            for (_, st_ch) in plan:
                steps.append(SkipFuseBlock(current_ch, st_ch, st_ch, gn_max_groups=gn_max_groups))
                current_ch = st_ch
            self.steps = nn.ModuleList(steps)

        # 最终预测头
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, feats: Dict[str, torch.Tensor], skip_list: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        x  = self.bridge(feats["layer4"])

        if self.skip_ev_k == 0:
            # 经典 4 步解码
            x  = self.up43(x, feats["layer3"])
            x  = self.up32(x, feats["layer2"])
            x  = self.up21(x, feats["layer1"])
            x  = self.up10(x, feats["conv1"])
            out = self.head(x)
            # 与 conv1 对齐（以防奇偶尺寸）
            H0, W0 = feats["conv1"].shape[-2:]
            if out.shape[-2:] != (H0, W0):
                out = F.interpolate(out, size=(H0, W0), mode="bilinear", align_corners=False)
            return out

        # 动态密集解码：需要传入 skip_list（深->浅）
        assert skip_list is not None, "skip_list must be provided when skip_ev_k > 0"
        assert len(skip_list) == self._expected_steps, \
            f"skip_list length {len(skip_list)} != expected {self._expected_steps}"

        for step, skip in zip(self.steps, skip_list):
            x = step(x, skip)

        out = self.head(x)
        # 以最后一个 skip 的尺寸或 conv1 尺寸对齐（conv1 始终是最后一个）
        target_h, target_w = skip_list[-1].shape[-2], skip_list[-1].shape[-1]
        if out.shape[-2:] != (target_h, target_w):
            out = F.interpolate(out, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return out


# -------------------------
#     Masking utilities
# -------------------------
def _make_patch_mask(
    x: torch.Tensor,
    mask_ratio: Union[float, torch.Tensor],
    patch_size: int = 4,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Create a binary mask of shape [B,1,H,W] by masking a ratio of non-overlapping patches.
    Masked pixels are 1; unmasked are 0.

    mask_ratio can be a float (applied to all samples) or a tensor of shape [B]
    with per-sample ratios.
    """
    B, _, H, W = x.shape
    # Validate mask_ratio
    if isinstance(mask_ratio, torch.Tensor):
        assert mask_ratio.dim() == 1 and mask_ratio.shape[0] == B, "mask_ratio tensor must be [B]"
    else:
        assert 0.0 <= mask_ratio <= 1.0, "mask_ratio must be in [0,1]"
    ph, pw = patch_size, patch_size
    # Make sure H, W are divisible by patch size (pad if needed)
    pad_h = (ph - (H % ph)) % ph
    pad_w = (pw - (W % pw)) % pw
    if pad_h or pad_w:
        x_ = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        H_, W_ = x_.shape[-2:]
    else:
        x_ = x
        H_, W_ = H, W

    # Number of patches
    nh, nw = H_ // ph, W_ // pw
    # Vectorized Bernoulli sampling per patch with per-sample probability
    device = x.device
    if isinstance(mask_ratio, torch.Tensor):
        probs = mask_ratio.to(device=device, dtype=x.dtype).clamp(0.0, 1.0).view(B, 1, 1).expand(B, nh, nw)
    else:
        probs = torch.full((B, nh, nw), float(mask_ratio), device=device, dtype=x.dtype).clamp(0.0, 1.0)
    rand = torch.rand((B, nh, nw), generator=generator, device=device, dtype=x.dtype)
    mask = (rand < probs).to(dtype=x.dtype).unsqueeze(1)  # [B,1,nh,nw]

    # Up-sample mask back to pixels by repeating within each patch
    mask = mask.repeat_interleave(ph, dim=2).repeat_interleave(pw, dim=3)  # [B,1,H_,W_]
    # Crop to original if padded
    mask = mask[:, :, :H, :W]
    return mask


# -------------------------
#       MAE-ResNet
# -------------------------
@persistent_class
class MAEResNetGN(nn.Module):
    """
    MAE-ResNet: ResNet encoder + U-Net decoder for masked autoencoding,
    with a classification head. The classification and reconstruction
    losses are combined via lambda:  loss = lambda*CE + (1-lambda)*MSE_masked.
    """
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        base_channels: int = 64,
        layers: Tuple[int, int, int, int] = (2, 2, 2, 2),
        dropout_prob: float = 0.0,
        patch_size: int = 4,
        block: nn.Module = BasicBlock,
        gn_max_groups: int = 32,
        skip_ev_k: int = 0,  # ★ 新增：跳连抽样步长；0 表示沿用旧的 4 层 U-Net
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size  = patch_size
        self.gn_max_groups = gn_max_groups
        self.skip_ev_k = int(skip_ev_k)

        # Encoder
        self.encoder = ResNetEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            layers=list(layers),
            dropout_prob=dropout_prob,
            block=block,
            gn_max_groups=self.gn_max_groups,
        )

        # Classification head (on top encoder feature)
        top_ch = self.encoder.out_channels["layer4"]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(top_ch, num_classes)

        # Decoder
        self.decoder = UNetDecoder(
            enc_channels=self.encoder.out_channels,
            out_channels=in_channels,
            gn_max_groups=self.gn_max_groups,
            layers_spec=tuple(layers),
            skip_ev_k=self.skip_ev_k,
        )

        # Init the classifier head
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    # ---------- Core interface ----------

    def _collect_skips_dense(self, feats: Dict[str, torch.Tensor], block_outputs: Dict[str, List[torch.Tensor]]) -> List[torch.Tensor]:
        """
        构建“深->浅”的 skip 列表：
        - layer4 的被选 block（倒序）
        - layer3 的被选 block（倒序）
        - layer2 的被选 block（倒序）
        - layer1 的被选 block（倒序）
        - 最后追加 conv1
        每个被选 block 先过对应的 stage-level GroupNorm。
        """
        k = self.skip_ev_k
        assert k > 0

        # 取每层被抽中的索引（1-based）
        def take_indices(num_blocks: int, step: int) -> List[int]:
            if step <= 0:
                return []
            return list(range(step, num_blocks + 1, step))

        # 编码器的 stage-level Norm
        stage_norm = {
            "layer1": self.encoder.layer1_norm,
            "layer2": self.encoder.layer2_norm,
            "layer3": self.encoder.layer3_norm,
            "layer4": self.encoder.layer4_norm,
        }

        skip_list: List[torch.Tensor] = []

        # 注意：按深->浅顺序 & 每个分辨率内部倒序（反向连接）
        for lname in ["layer4", "layer3", "layer2", "layer1"]:
            blocks = block_outputs.get(lname, [])
            n = len(blocks)
            sel = take_indices(n, k)
            # 倒序以便先连接最新的该分辨率输出
            for idx in reversed(sel):
                feat = blocks[idx - 1]  # 1-based -> 0-based
                feat = stage_norm[lname](feat)  # 对齐尺度
                skip_list.append(feat)

        # 最后追加 conv1（最高分辨率）
        skip_list.append(feats["conv1"])
        return skip_list

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        lambda_cls: float,
        mask_ratio_min: Union[float, torch.Tensor] = 0.75,
        mask_ratio_max: Union[float, torch.Tensor] = 0.95,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass computing the weighted sum of classification and masked reconstruction losses.

        Args:
            x:        Tensor [B,C,H,W], input image/latent.
            labels:   Tensor [B], class indices.
            lambda_cls: float in [0,1], weight for classification loss; (1 - lambda_cls) is for reconstruction.
            mask_ratio_min: float or Tensor [B], minimum fraction to mask if `mask` not provided.
            mask_ratio_max: float or Tensor [B], maximum fraction to mask if `mask` not provided.
            mask:     Optional binary mask [B,1,H,W] (1 = masked, 0 = visible). If None, a random patch mask is used.

        Returns:
            loss: scalar tensor, combined objective.
            info: dict with auxiliary stats (acc, cls_loss, recon_loss, mask_ratio, per-scale stds).
        """
        assert 0.0 <= lambda_cls <= 1.0, "lambda_cls must be in [0,1]"

        # Build mask if not provided
        mask = self._prepare_mask(x, mask, mask_ratio_min, mask_ratio_max)

        # Mask input by zeroing masked pixels
        x_masked = x * (1.0 - mask)

        # Encode masked input
        if self.skip_ev_k == 0:
            feats = self.encoder(x_masked)
        else:
            feats, block_outputs = self.encoder(x_masked, return_block_outputs=True)

        # Classification logits from top-level feature
        top = feats["layer4"]
        logits = self.fc(self.avgpool(top).flatten(1))

        # Decode full reconstruction
        if self.skip_ev_k == 0:
            recon = self.decoder(feats)
        else:
            skip_list = self._collect_skips_dense(feats, block_outputs)
            recon = self.decoder(feats, skip_list=skip_list)

        # --- Losses ---
        cls_loss   = F.cross_entropy(logits, labels)
        # MSE on masked pixels only; normalize by number of masked pixels to keep scale stable
        mse = (recon - x) ** 2
        masked_mse = (mse * mask).sum() / (mask.sum() + 1e-8)
        zero_recon_loss = (x ** 2 * mask).sum() / (mask.sum() + 1e-8)

        # Combined
        loss = lambda_cls * cls_loss + (1.0 - lambda_cls) * masked_mse

        # Stats
        with torch.no_grad():
            acc = (logits.argmax(dim=1) == labels).float().mean()
            info = {
                "loss": loss.detach(),
                "cls_loss": cls_loss.detach(),
                "recon_loss": masked_mse.detach(),
                "zero_recon_loss": zero_recon_loss.detach(),
                "acc": acc,
                # report actual masked pixel ratio per batch (mean over spatial dims)
                "mask_ratio": mask.to(dtype=torch.float32).mean(dim=(1,2,3)).mean(),
                "scale_conv1_std": feats["conv1"].std().mean(),
                "scale_1_std": feats["layer1"].std().mean(),
                "scale_2_std": feats["layer2"].std().mean(),
                "scale_3_std": feats["layer3"].std().mean(),
                "scale_4_std": feats["layer4"].std().mean(),
            }
        return loss, info

    @torch.no_grad()
    def reconstruct(
        self,
        x: torch.Tensor,
        mask_ratio_min: Union[float, torch.Tensor] = 0.75,
        mask_ratio_max: Union[float, torch.Tensor] = 0.95,
        mask: Optional[torch.Tensor] = None,
        only_masked: bool = False,
    ) -> torch.Tensor:
        """
        Reconstruct the input (or only its masked part) given a mask ratio or explicit mask.

        Args:
            x:           [B,C,H,W]
            mask_ratio_min:  float or Tensor [B], minimum fraction to mask if `mask` not provided.
            mask_ratio_max:  float or Tensor [B], maximum fraction to mask if `mask` not provided.
            mask:        optional [B,1,H,W] (1 = masked), overrides mask_ratio if provided.
            only_masked: if True, returns the inpainted result (input with masked areas replaced by predictions).
                         if False, returns the full reconstruction image.

        Returns:
            recon: [B,C,H,W] tensor. If only_masked=True, it's the inpainted image (x on visibles, pred on masked).
        """
        mask = self._prepare_mask(x, mask, mask_ratio_min, mask_ratio_max)
        x_masked = x * (1.0 - mask)

        if self.skip_ev_k == 0:
            feats = self.encoder(x_masked)
            recon = self.decoder(feats)
        else:
            feats, block_outputs = self.encoder(x_masked, return_block_outputs=True)
            skip_list = self._collect_skips_dense(feats, block_outputs)
            recon = self.decoder(feats, skip_list=skip_list)

        if only_masked:
            recon = x * (1.0 - mask) + recon * mask
        return recon

    # ---------- Activations API ----------

    def activations(
        self,
        x: torch.Tensor,
        min_res: int = 32,
        use_std: bool = False,
        use_mean: bool = False,
        residual_std_mean: bool = False,
        use_layers=[0,1,2,3,4], # conv1, layer1, layer2, layer3, layer4
        unfold_kernel_list: List[int] = [],
        patch_group_size: int = 1,
        transpose_on_layers=[0,1,2,3,4],
        patch_mean_size: List[int] = [], # will compute mean over these sizes
        patch_std_size: List[int] = [], # will compute std over these sizes
        transpose_goal_dims: List[int] = [], # will transpose & divide the channels, aimed for goal_dims
        every_k_block: Union[int, float] = float("inf"),
    ) -> Dict[str, torch.Tensor]:
        """
        Collect a small set of activations in the same spirit as your LatentResNet.activations.
        Returns a dict mapping names -> [B, F, D] sequences or statistics.
        The number of returned entries is capped by `num_acti` (first-come).
        """
        assert len(x.shape) == 4 and x.shape[-1] == x.shape[-2], "x must be [B,C,H,W] square"
        if x.shape[-1] < min_res:
            x = F.interpolate(x, size=min_res, mode="bicubic")

        feats, residuals, block_outputs = self.encoder(x, return_residuals=True, return_block_outputs=True)
        res: Dict[str, torch.Tensor] = {}

        def push_tensor(name: str, feat: torch.Tensor, residuals: List[torch.Tensor], layer_idx: int) -> bool:
            nonlocal res
            if layer_idx not in use_layers:
                return
            res[name] = group_features(feat, patch_group_size)
            for size in patch_mean_size:
                grp_features = group_features(feat, size) # [B, H * W // (size ** 2), size ** 2, C]
                res[f"{name}_mean_{size}"] = grp_features.mean(dim=2)
            for size in patch_std_size:
                grp_features = group_features(feat, size) # [B, H * W // (size ** 2), size ** 2, C]
                res[f"{name}_std_{size}"] = grp_features.std(dim=2)
            if layer_idx in transpose_on_layers:
                for i in transpose_goal_dims:
                    b, c, h, w = feat.shape
                    chunk_size = max(i // (h * w), 1)
                    res[f"{name}_transpose_{i}"] = rearrange(feat, "b (c1 c2) h w -> b c1 (c2 h w)", c2=chunk_size)
                    # (B H W C)
            
            if use_mean:
                res[f"{name}_mean"] = rearrange(feat, "b c h w -> b (h w) c").mean(dim=1, keepdim=True)
            if use_std:
                reshaped = rearrange(feat, "b c h w -> b (h w) c")
                res[f"{name}_std"] = reshaped if reshaped.shape[1] == 1 else reshaped.std(dim=1, keepdim=True)
            for k in unfold_kernel_list:
                if feat.shape[-1] > 1 and feat.shape[-2] > 1:
                    unfolded, corr = unfold_feats(feat, k, return_corr=True)
                    res[f"{name}_unfold_{k}"] = rearrange(unfolded, "b k l h w c -> b (h w) (k l c)")
                    res[f"{name}_corr_{k}"] = corr
            if residual_std_mean:
                for i in range(len(residuals)):
                    res[f"{name}_residual_{i}_mean"] = rearrange(residuals[i], "b c h w -> b (h w) c").mean(dim=1, keepdim=True)
                    res[f"{name}_residual_{i}_std"] = rearrange(residuals[i], "b c h w -> b (h w) c").std(dim=1, keepdim=True)

        # Order: encoder pyramid then a decoder intermediate
        push_tensor("conv1", feats["conv1"], [], 0)
        push_tensor("layer1", feats["layer1"], residuals["layer1"], 1)
        push_tensor("layer2", feats["layer2"], residuals["layer2"], 2)
        push_tensor("layer3", feats["layer3"], residuals["layer3"], 3)
        push_tensor("layer4", feats["layer4"], residuals["layer4"], 4)

        # Additionally, return every k-th block output (1-based index), if requested
        if isinstance(every_k_block, (int, float)) and not math.isinf(float(every_k_block)) and every_k_block >= 1:
            k = int(every_k_block)
            for lname, lidx in [("layer1", 1), ("layer2", 2), ("layer3", 3), ("layer4", 4)]:
                blocks = block_outputs.get(lname, [])
                for i, feat_i in enumerate(blocks, start=1):
                    if i % k == 0:
                        push_tensor(f"{lname}_blk{i}", feat_i, [], lidx)
        return res

    # ---------- Helpers ----------

    def _prepare_mask(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        mask_ratio_min: Union[float, torch.Tensor],
        mask_ratio_max: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """Ensure we have a [B,1,H,W] binary mask (1=masked).

        If mask is None, sample per-sample mask ratios uniformly between
        mask_ratio_min and mask_ratio_max (both can be float or [B] tensor),
        then build a Bernoulli mask with those probabilities.
        """
        if mask is None:
            B = x.shape[0]
            device = x.device
            dtype = x.dtype
            if isinstance(mask_ratio_min, torch.Tensor):
                min_t = mask_ratio_min.to(device=device, dtype=dtype).view(B)
            else:
                min_t = torch.full((B,), float(mask_ratio_min), device=device, dtype=dtype)
            if isinstance(mask_ratio_max, torch.Tensor):
                max_t = mask_ratio_max.to(device=device, dtype=dtype).view(B)
            else:
                max_t = torch.full((B,), float(mask_ratio_max), device=device, dtype=dtype)
            lo = torch.minimum(min_t, max_t).clamp(0.0, 1.0)
            hi = torch.maximum(min_t, max_t).clamp(0.0, 1.0)
            # sample per-sample probabilities uniformly in [lo, hi]
            probs = lo + (hi - lo) * torch.rand(B, device=device, dtype=dtype)
            mask = _make_patch_mask(x, mask_ratio=probs, patch_size=self.patch_size)
        else:
            if mask.dim() == 3:  # [B,H,W]
                mask = mask.unsqueeze(1)
            assert mask.shape[:2] == (x.shape[0], 1) and mask.shape[-2:] == x.shape[-2:], "mask must be [B,1,H,W]"
            mask = mask.to(dtype=x.dtype, device=x.device)
        return mask


# -------------------------
#          Builder
# -------------------------
def build_mae_resnet_gn(**kwargs) -> MAEResNetGN:
    """
    Builder for MAEResNet with optional checkpoint loading.
    Usage matches build_latent_resnet(...).

    Args:
        load_dict (dict, optional): {'run_id': str, 'epoch': 'latest'|int, 'load_entry': 'model'|...}
        num_classes (int): number of classes for classification head
        in_channels (int): input channels
        base_channels (int): base channel width of ResNet
        layers (tuple/list of 4 ints): ResNet block counts (e.g., [2,2,2,2] for ResNet-18)
        dropout_prob (float): dropout probability in BasicBlock
        patch_size (int): masking patch size (default: 4)
        block (nn.Module): residual block class (default: BasicBlock)
        skip_ev_k (int): ★ 新增，按每 k 个 block 抽一条跳连；0 表示关闭，退化为经典 U-Net。

    Returns:
        MAEResNet instance (optionally loaded from checkpoint).
    """
    load_dict = kwargs.pop("load_dict", None)
    model = MAEResNetGN(**kwargs)
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

def get_resnet_spec(arch_name: str):
    """Utility to mirror your existing helper."""
    if arch_name == "resnet18":
        return BasicBlock, [2, 2, 2, 2]
    elif arch_name == "resnet34":
        return BasicBlock, [3, 4, 6, 3]
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")


# -------------------------
#            Demo
# -------------------------
if __name__ == "__main__":
    # Example: 4-channel latent, 32x32, ResNet-18 backbone
    block, layers = get_resnet_spec("resnet18")
    # 1) 经典：skip_ev_k = 0（保持原行为）
    model0 = build_mae_resnet_gn(
        num_classes=1000,
        in_channels=4,
        base_channels=128,
        layers=layers,
        dropout_prob=0.0,
        patch_size=4,
        block=block,
        gn_max_groups=32,
        skip_ev_k=0,
    )
    x = torch.randn(2, 4, 32, 32)
    y = torch.randint(0, 10, (2,))
    loss0, info0 = model0(x, y, lambda_cls=0.5, mask_ratio_min=0.5, mask_ratio_max=0.5)
    print("[k=0] loss:", float(info0["loss"]), "acc:", float(info0["acc"]))

    # 2) 密集跳连：例如每 1 个 block 取一次（ResNet-18 每层 2 个 block）
    model1 = build_mae_resnet_gn(
        num_classes=1000,
        in_channels=4,
        base_channels=128,
        layers=layers,
        dropout_prob=0.0,
        patch_size=4,
        block=block,
        gn_max_groups=32,
        skip_ev_k=2,  # <- 每个分辨率用所有 block 的输出做跳连（深->浅依次融合）
    )
    loss1, info1 = model1(x, y, lambda_cls=0.5, mask_ratio_min=0.5, mask_ratio_max=0.5)
    print("[k=1] loss:", float(info1["loss"]), "acc:", float(info1["acc"]))
    recon = model1.reconstruct(x, mask_ratio_min=0.5, mask_ratio_max=0.5, only_masked=True)
    print("recon:", recon.shape)
    acts = model1.activations(x, use_mean=True, use_std=True, unfold_kernel_list=[], patch_group_size=2)
    print({k: v.shape for k, v in acts.items()})

# python -m model.unet_gn
# 32x32x128, 32x32x128,16x16x256, 8x8x512, 4x4x1024
