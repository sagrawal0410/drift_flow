# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List, Union
import torch.nn.functional as F
from einops import repeat, rearrange
from utils.profile import print_module_summary

from timm.models.vision_transformer import PatchEmbed, Block
from utils.misc import custom_compile



class MAEViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=32, patch_size=2, in_channels=4,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, n_cls_tokens=1,
                 num_classes: int = 0):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.in_channels = in_channels
        self.n_cls_tokens = int(n_cls_tokens)
        num_patches = self.patch_embed.num_patches

        # multiple learnable class tokens
        self.cls_token = nn.Parameter(torch.randn(1, self.n_cls_tokens, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.num_classes = num_classes
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # decoder pos embed mirrors encoder length (n_cls_tokens + patches)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_channels, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.encoder_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm_pix_loss = norm_pix_loss
        
        self.fc = nn.Linear(embed_dim, self.num_classes)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding; cls positions as zeros
        grid_size = int(self.patch_embed.num_patches**.5)
        patch_pos = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size, cls_token=False)
        pe = torch.zeros(1, patch_pos.shape[0], self.pos_embed.shape[-1])       
        pe[:, :] = torch.from_numpy(patch_pos).float()
        self.pos_embed.data.copy_(pe)

        dec_patch_pos = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], grid_size, cls_token=False)
        dpe = torch.zeros(1, dec_patch_pos.shape[0], self.decoder_pos_embed.shape[-1])
        dpe[:, :] = torch.from_numpy(dec_patch_pos).float()
        self.decoder_pos_embed.data.copy_(dpe)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_mask_token, std=.02)

        # init classifier if present
        if hasattr(self, "fc"):
            torch.nn.init.normal_(self.fc.weight, std=0.01)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias, 0)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 * C)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        c = imgs.shape[1]
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        c = x.shape[-1] // (p * p)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, return_block_outputs=False):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        # masking: if mask_ratio is None, skip masking
        if mask_ratio is None:
            B, L, _ = x.shape
            mask = torch.zeros((B, L), device=x.device, dtype=x.dtype)
            ids_restore = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        else:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        x = torch.cat((repeat(self.cls_token, '1 n d -> b n d', b=x.shape[0]), x), dim=1)

        # apply Transformer blocks
        block_outputs = []
        for blk in self.blocks:
            x = blk(x)
            if return_block_outputs:
                block_outputs.append(x)
        x = self.norm(x)
        
        if return_block_outputs:
            return x, mask, ids_restore, block_outputs
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + self.n_cls_tokens - x.shape[1], 1)
        x_ = torch.cat([x[:, self.n_cls_tokens:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :self.n_cls_tokens, :], x_ + self.decoder_pos_embed], dim=1)  # append cls token

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, self.n_cls_tokens:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x, labels: Optional[torch.Tensor] = None, lambda_cls: float = 0.0, mask_ratio_min=0.5, mask_ratio_max=0.95, random_mask_prefix: float = 0.0):
        mask_ratio = torch.rand(1, device=x.device) * (mask_ratio_max - mask_ratio_min) + mask_ratio_min
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        # latent: [B, N+self.n_cls_tokens, D]

        N, L, D = latent.shape
        to_mask_prefix = (torch.rand(N) < random_mask_prefix).to(latent) # [N]
        masked = torch.arange(L).to(latent).unsqueeze(0).repeat(N, 1) / L # [N, L]
        threshold = torch.rand(N, device=latent.device) # [N]
        threshold = to_mask_prefix * threshold + (1 - to_mask_prefix) # [N]
        remain = (masked < threshold.unsqueeze(1)).to(latent)[:, :, None] # [N, L, 1]
        latent = latent * remain + self.encoder_mask_token * (1 - remain)

        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        recon_loss = self.forward_loss(x, pred, mask)

        cls_tokens = latent[:, :self.n_cls_tokens, :]
        cls_pooled = cls_tokens.mean(dim=1)
        logits = self.fc(cls_pooled)
        cls_loss = F.cross_entropy(logits, labels)
        loss = (1.0 - float(lambda_cls)) * recon_loss + float(lambda_cls) * cls_loss

        with torch.no_grad():
            acc = (logits.argmax(dim=1) == labels).float().mean()
            info = {
                "loss": loss.detach(),
                "cls_loss": cls_loss.detach(),
                "recon_loss": recon_loss.detach(),
                "acc": acc,
            }
        return loss, info

    @custom_compile()
    def get_activations(self, x: torch.Tensor, target_resolutions: List[int], every_k_block: Union[int, float] = float("inf"), ) -> Dict[str, torch.Tensor]:
        """
        Reuse the encoder to obtain token embeddings, then downsample patch tokens
        to each target resolution using adaptive average pooling.

        Args:
            x: [B, C, H, W]
            target_resolutions: list of ints r, each producing [B, r*r, D]
            every_k_block: optional, if specified, returns the output of every k-th transformer block.

        Returns:
            dict with keys:
              - 'cls_tokens': [B, n_cls_tokens, D]
              - f'res_{r}': [B, r*r, D] for each r in target_resolutions
        """
        # forward encoder without masking
        import math
        forward_kwargs = {"mask_ratio": None}
        return_block_outputs = isinstance(every_k_block, (int, float)) and not math.isinf(float(every_k_block)) and every_k_block >= 1
        if return_block_outputs:
            forward_kwargs["return_block_outputs"] = True
            latent, _, _, block_outputs = self.forward_encoder(x, **forward_kwargs)
        else:
            latent, _, _ = self.forward_encoder(x, **forward_kwargs)
            block_outputs = []

        # --- Unified processing for selected layers ---
        outputs = dict()
        
        layers_to_process = []
        if return_block_outputs:
            k = int(every_k_block)
            for i, blk_out in enumerate(block_outputs, start=1):
                if i % k == 0:
                    layers_to_process.append((f"block_{i}", blk_out))
        layers_to_process.append(("final", latent))

        for name, layer_latent in layers_to_process:
            # split cls vs patch tokens
            cls_tokens = layer_latent[:, :self.n_cls_tokens, :]
            patch_tokens = layer_latent[:, self.n_cls_tokens:, :]  # [B, L, D]
    
            B, L, D = patch_tokens.shape
            side = int(L ** 0.5)
            assert side * side == L, "Number of patches must be a perfect square"
            patch_tokens_2d = patch_tokens.view(B, side, side, D) # [B, S, S, D]
            
            # process patch tokens for each target resolution
            for r in target_resolutions:
                pooled = rearrange(patch_tokens_2d, 'b (h a) (w x) d -> b (h w) (a x) d', h=r, w=r).mean(dim=2)
                outputs[f"{name}_res_{r}"] = pooled
                outputs[f'{name}_res_{r}_mean'] = pooled.mean(dim=1, keepdim=True)
                outputs[f'{name}_res_{r}_std'] = pooled.std(dim=1, keepdim=True)

            # process cls tokens
            outputs[f"{name}_cls_tokens"] = cls_tokens # [B, n_cls_tokens, D]
            outputs[f'{name}_cls_tokens_mean'] = cls_tokens.mean(dim=1, keepdim=True)
            outputs[f'{name}_cls_tokens_std'] = cls_tokens.std(dim=1, keepdim=True)

        return outputs
    
    activations = get_activations



def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def build_mae_vit(**kwargs) -> MAEViT:
    """
    Builder for MAEViT with optional checkpoint loading.
    Usage matches build_latent_resnet(...).

    Args:
        load_dict (dict, optional): {'run_id': str, 'epoch': 'latest'|int, 'load_entry': 'model'|...}
        **kwargs: Standard MAEViT constructor arguments (in_channels, embed_dim, depth, num_heads, decoder_embed_dim, decoder_depth, decoder_num_heads, mlp_ratio, norm_layer, norm_pix_loss, n_cls_tokens, num_classes)

    Returns:
        MAEViT instance (optionally loaded from checkpoint).
    """
    load_dict = kwargs.pop("load_dict", None)
    model = MAEViT(**kwargs)
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

if __name__ == "__main__":
    model = MAEViT(img_size=32, patch_size=2, in_chans=4, embed_dim=768, depth=12, num_heads=12, decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, n_cls_tokens=32, num_classes=1000)
    x = torch.randn(1, 4, 32, 32)
    print_module_summary(model, inputs=[x,], kwargs=dict(mask_ratio=0.0, labels=torch.randint(0, 1000, (1,))))
    loss, info = model(x, mask_ratio=0.75, labels=torch.randint(0, 1000, (1,)))
    # print(loss, info)
    print(loss)
    for k, v in info.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        else:
            print(k, v)
    acts = model.get_activations(x, target_resolutions=[16, 8, 4, 2])
    for k, v in acts.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        else:
            print(k, v)
    # print(acts)

# python -m model.mae_vit