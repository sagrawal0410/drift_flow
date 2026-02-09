"""
Lightning DiT's codes are built from original DiT & SiT.
(https://github.com/facebookresearch/DiT; https://github.com/willisma/SiT)
It demonstrates that a advanced DiT together with advanced diffusion skills
could also achieve a very promising result with 1.35 FID on ImageNet 256 generation.

Enjoy everyone, DiT strikes back!

by Maple (Jingfeng Yao) from HUST-VL
"""

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import PatchEmbed, Mlp
from model.LightningDiT.swiglu_ffn import SwiGLUFFN 
from model.LightningDiT.pos_embed import VisionRotaryEmbeddingFast
from utils.misc import custom_compile
from einops import repeat

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# @custom_compile()
def modulate(x, shift, scale):
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Attention(nn.Module):
    """
    Attention module of LightningDiT.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_attn: bool = True,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        
        if use_rmsnorm:
            norm_layer = RMSNorm
            
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if rope is not None:
            q = rope(q)
            k = rope(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LightningDiTBlock(nn.Module):
    """
    Lightning DiT Block. We add features including: 
    - ROPE
    - QKNorm 
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False, 
        use_rmsnorm=False,
        wo_shift=False,
        cond_dim=None, # without specifying: should be hidden_size
        **block_kwargs
    ):
        super().__init__()
        
        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
            
        # Initialize attention layer
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )
        
        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            hid_size = int(2/3 * mlp_hidden_dim)
            hid_size = (hid_size + 31) // 32 * 32 # round up to the nearest multiple of 32
            self.mlp = SwiGLUFFN(hidden_size, hid_size)
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )
            
        # Initialize AdaLN modulation
        # print("!!", cond_dim)
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size if cond_dim is None else cond_dim, 4 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size if cond_dim is None else cond_dim, 6 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift
# 
    # @custom_compile()
    def forward(self, x, c, feat_rope=None):
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of LightningDiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False, cond_dim=None):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size if cond_dim is None else cond_dim, 2 * hidden_size, bias=True)
        )
    # @custom_compile()
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class LightningDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        out_channels=32,
        use_qknorm=False,
        use_swiglu=False,
        use_rope=False,
        use_rmsnorm=False,
        wo_shift=False,
        use_checkpoint=False,
        cond_dim=None, # if None, use hidden_size
        attn_drop=0.0,
        proj_drop=0.0,
        n_cls_tokens=0,
        compile_mode="none",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.depth = depth
        self.hidden_size = hidden_size
        self.use_checkpoint = use_checkpoint
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.n_cls_tokens = n_cls_tokens
        if n_cls_tokens > 0:
            self.cls_embed = nn.Parameter(torch.randn(1, n_cls_tokens, hidden_size) * 0.02, requires_grad=True)
            self.cond_to_token = nn.Linear(cond_dim if cond_dim is not None else hidden_size, hidden_size, bias=True)

        # use rotary position encoding, borrow from EVA
        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
                num_cls_token=n_cls_tokens,
            )
        else:
            self.feat_rope = None

        self.blocks = nn.ModuleList([
            LightningDiTBlock(hidden_size, 
                     num_heads, 
                     mlp_ratio=mlp_ratio, 
                     use_qknorm=use_qknorm, 
                     use_swiglu=use_swiglu, 
                     use_rmsnorm=use_rmsnorm,
                     wo_shift=wo_shift,
                     cond_dim=cond_dim,
                     attn_drop=attn_drop,
                     proj_drop=proj_drop,
                     ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm, cond_dim=cond_dim)
        self.initialize_weights()
        if compile_mode != "none":
            self.forward = torch.compile(self.forward, 
                                        mode=compile_mode)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, cond=None):
        """
        Forward pass of LightningDiT.
        x: (B, C, H, W) tensor of spatial inputs (images or latent representations of images)
        cond: (B, D) tensor of condition embeddings
        use_checkpoint: boolean to toggle checkpointing
        """

        use_checkpoint = self.use_checkpoint

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        c = cond
        if self.n_cls_tokens > 0:
            c_tokens = self.cond_to_token(cond) # [B, D2]
            c_tokens = repeat(c_tokens, 'b d2 -> b n d2', n=self.n_cls_tokens) + self.cls_embed # [B, N, D2]
            x = torch.cat([c_tokens, x], dim=1)

        for block in self.blocks:
            if use_checkpoint:
                x = checkpoint(block, x, c, self.feat_rope, use_reentrant=True)
            else:
                x = block(x, c, self.feat_rope)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        if self.n_cls_tokens > 0:
            x = x[:, self.n_cls_tokens:, :]
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class DitGen(nn.Module):
    def __init__(self, cond_dim, num_classes=1001,noise_classes=0,noise_coords=1,input_size=32,in_channels=3,noise_layers=0,noise_pad=0,n_cls_tokens=0, **kwargs):
        super().__init__()
        # print("!!", n_cls_tokens)
        self.dit = LightningDiT(**kwargs, input_size=input_size, in_channels=in_channels, num_classes=num_classes, cond_dim=cond_dim, n_cls_tokens=n_cls_tokens)
        self.cond_dim = cond_dim
        self.class_embed = nn.Embedding(num_classes, cond_dim)
        self.class_embed.weight.data.normal_(mean=0.0, std=0.02)
        self.cfg_scale_embed = nn.Sequential(
            TimestepEmbedder(cond_dim),
            nn.RMSNorm(cond_dim),
        )
        self.noise_classes = noise_classes
        self.input_size = input_size
        self.in_channels = in_channels
        self.noise_layers = noise_layers
        self.noise_pad = noise_pad
        if self.noise_layers > 0:
            feature_dims = [self.noise_pad + self.cond_dim] + [self.cond_dim] * self.noise_layers
            noise_maps = []
            for (xi, xo) in zip(feature_dims[:-1], feature_dims[1:]):
                noise_maps.append(nn.Linear(xi, xo))
                noise_maps.append(nn.SiLU())
            noise_maps.pop()
            self.f_noise = nn.Sequential(*noise_maps)
        if noise_classes > 0:
            self.noise_coords = noise_coords
            self.noise_embeds = nn.ModuleList([
                nn.Embedding(noise_classes, cond_dim) for _ in range(noise_coords)
            ])
            for embed in self.noise_embeds:
                embed.weight.data.normal_(mean=0.0, std=0.02)
    
    def generate_noise_dict(self, bsz, device):
        """
        Generates a dictionary of noise tensors for the forward pass.
        Args:
            bsz (int): The batch size.
            device (torch.device): The device to create tensors on.
        Returns:
            dict: A dictionary of noise tensors.
        """
        noise_dict = {}
        if self.noise_classes > 0:
            noise_dict['noise_labels'] = torch.randint(0, self.noise_classes, (bsz, self.noise_coords), device=device)
        if self.noise_layers > 0 and self.noise_pad > 0:
            noise_dict['concat_noise'] = torch.randn((bsz, self.noise_pad), device=device) * 0.02
        noise_dict['x'] = torch.randn((bsz, self.in_channels, self.input_size, self.input_size), device=device)
        return noise_dict

    def forward(self, c, cfg_scale=1.0, temp=1.0, noise_dict=None):
        """
        Forward pass of the generator.
        Args:
            c (torch.Tensor): The class conditioning tensor.
            cfg_scale (float | [B]): The cfg scale for sampling.
            temp (float): The temperature for sampling.
            noise_dict (dict, optional): A dictionary of pre-generated noise. If None, new noise will be generated.
        Returns:
            dict with entries:
                samples: torch.Tensor: The generated samples.
                noise: dict: The noise dictionary used for generation.
        """
        if noise_dict is None:
            noise_dict = self.generate_noise_dict(c.shape[0], c.device)

        cond = self.class_embed(c)
        if self.noise_classes > 0:
            for coord_idx in range(self.noise_coords):
                noise_labels = noise_dict['noise_labels'][:, coord_idx]
                noise_embed = self.noise_embeds[coord_idx](noise_labels)
                cond = cond + noise_embed
        # Ensure cfg_scale is a 1D tensor of shape [B]
        if isinstance(cfg_scale, (float, int)):
            cfg_scale_t = torch.full((c.shape[0],), float(cfg_scale), device=c.device)
        else:
            cfg_scale_t = torch.as_tensor(cfg_scale, device=c.device)
            if cfg_scale_t.ndim == 0:
                cfg_scale_t = cfg_scale_t.expand(c.shape[0])
            else:
                cfg_scale_t = cfg_scale_t.view(-1)
                if cfg_scale_t.numel() == 1:
                    cfg_scale_t = cfg_scale_t.expand(c.shape[0])
                elif cfg_scale_t.shape[0] != c.shape[0]:
                    raise ValueError(f"cfg_scale has incompatible shape: {cfg_scale_t.shape} for batch {c.shape[0]}")
        cfg_embed = self.cfg_scale_embed(cfg_scale_t) * 0.02
        cond = cond + cfg_embed
        if self.noise_layers > 0:
            cond = cond * (cond.square().mean(dim=1, keepdim=True) + 1e-6).rsqrt()
            if self.noise_pad > 0:
                concat_noise = noise_dict['concat_noise']
                concat_noise = concat_noise * (concat_noise.square().mean(dim=1, keepdim=True) + 1e-6).rsqrt()
                cond = torch.cat([concat_noise, cond], dim=1)
            cond = self.f_noise(cond)
        x = noise_dict['x'] * temp
        return {
            "samples": self.dit(x, cond=cond),
            "noise": noise_dict
        }
    
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
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
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
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
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                             LightningDiT Configs                              #
#################################################################################

def LightningDiT_XL_1(**kwargs):
    return LightningDiT(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)

def LightningDiT_XL_2(**kwargs):
    return LightningDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def LightningDiT_L_2(**kwargs):
    return LightningDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def LightningDiT_B_1(**kwargs):
    return LightningDiT(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def LightningDiT_B_2(**kwargs):
    return LightningDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def LightningDiT_1p0B_1(**kwargs):
    return LightningDiT(depth=24, hidden_size=1536, patch_size=1, num_heads=24, **kwargs)

def LightningDiT_1p0B_2(**kwargs):
    return LightningDiT(depth=24, hidden_size=1536, patch_size=2, num_heads=24, **kwargs)

def LightningDiT_1p6B_1(**kwargs):
    return LightningDiT(depth=28, hidden_size=1792, patch_size=1, num_heads=28, **kwargs)

def LightningDiT_1p6B_2(**kwargs):
    return LightningDiT(depth=28, hidden_size=1792, patch_size=2, num_heads=28, **kwargs)

LightningDiT_models = {
    'LightningDiT-B/1': LightningDiT_B_1, 'LightningDiT-B/2': LightningDiT_B_2,
    'LightningDiT-L/2': LightningDiT_L_2,
    'LightningDiT-XL/1': LightningDiT_XL_1, 'LightningDiT-XL/2': LightningDiT_XL_2,
    'LightningDiT-1p0B/1': LightningDiT_1p0B_1, 'LightningDiT-1p0B/2': LightningDiT_1p0B_2,
    'LightningDiT-1p6B/1': LightningDiT_1p6B_1, 'LightningDiT-1p6B/2': LightningDiT_1p6B_2,
}

from utils.profile import print_module_summary
import time
if __name__ == '__main__':
    from utils.profile import print_module_summary
    model = LightningDiT_B_2(cond_dim=1024, in_channels=32, use_rope=True, use_swiglu=True, use_qknorm=True, use_rmsnorm=True, wo_shift=True)
    model = model.cuda()
    x = torch.randn(1, 32, 32, 32).cuda()
    cond = torch.randn(1, 1024).cuda()
    # out = model(x, cond)

    # f = model(x, cond)
    # print_module_summary(model, [x.cuda(), cond.cuda()])
    # print_module_summary(model, [x.cuda(), cond.cuda()])
    f = model(x, cond)
    print(f.shape)
    gen = DitGen(cond_dim=1024, 
                    num_classes=1001,
                    input_size=32,
                    in_channels=4,
                    use_rope=False, 
                    use_swiglu=True, 
                    use_qknorm=False, 
                    use_rmsnorm=False, 
                    wo_shift=False, 
                    out_channels=3,
                    hidden_size=1024,
                    depth=24,
                    num_heads=16,
                    mlp_ratio=4.0,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    patch_size=2,
                    noise_layers=0,
                    n_cls_tokens=64,
                    noise_pad=128,
                    noise_coords=32,
                    noise_classes=64, 
                    compile_mode="default"
                    ).cuda()
    for x in range(10):
        before_time = time.time()
            
        use_bf16 = True
        context = torch.cuda.amp.autocast(dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()
        with torch.no_grad(), context:
            c = torch.randint(0, 1001, (256,)).cuda()
            f = gen(c)
            # print(f.shape)
            after_time = time.time()
            print(f"Time taken: {after_time - before_time} seconds")

# python -m model.LightningDiT.lightningdit
# base: 132.2M; 21.8G / sample
# time: for 16 samples, 0.1s; for 8 samples: 0.063s. 
# compilation: 32s for 64 samples.