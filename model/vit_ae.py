# %%
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
import sys
sys.path.append('..')
from utils.misc import EasyDict
import torch.nn.functional as F
from utils.persistence import persistent_class
from timm.layers import use_fused_attn
def modulate(x, shift, scale):
    return x * (1 + scale) + shift

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

def style(x):
    mn = x.mean(dim=1, keepdim=True)
    msqr = ((x - mn) ** 2).mean(dim=1, keepdim=True).clamp(min=1e-6).sqrt()
    return mn, msqr

@persistent_class
class InputProcess(nn.Module):
    def __init__(self, input_shape, hidden_size, patch_size=2, mlp_layers=0):
        '''
        Args:
            input_shape: (C, H, W) or (L, D)
            hidden_size: int, hidden size of the transformer
            patch_size: int, patch size of the input tensor

        Help process the input tensor to the transformer, including mapping the dimensions and adding positional embedding.
        '''

        super().__init__()
        self.input_shape = input_shape
        self.hidden_size = hidden_size

        L, D = shape_to_ld(input_shape, patch_size)
        if mlp_layers > 0:
            self.mlp_list = []
            for i in range(mlp_layers):
                self.mlp_list.append(nn.SiLU())
                self.mlp_list.append(nn.Linear(hidden_size, hidden_size))
            self.mlp = nn.Sequential(*self.mlp_list)
        else:
            self.mlp = nn.Identity()

        if len(input_shape) == 3:
            assert input_shape[1] == input_shape[2], "H and W must be the same, found {} and {}".format(input_shape[1], input_shape[2])

            # patchify input, to obtain (B, C, H, W) -> (B, L, hidden_size)
            self.emb = PatchEmbed(img_size=input_shape[-1], patch_size=patch_size, in_chans=input_shape[0], embed_dim=hidden_size)
            w = self.emb.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.emb.proj.bias, 0)

            # construct positional embedding
            values = get_2d_sincos_pos_embed(hidden_size, (input_shape[-1] // patch_size))
            self.input_pos_embed = nn.Parameter(torch.zeros(1, L, hidden_size), requires_grad=True)
            self.input_pos_embed.data.copy_(torch.from_numpy(values * 0.02).float().unsqueeze(0))
        else:
            self.emb = nn.Linear(D, hidden_size)
            nn.init.xavier_uniform_(self.emb.weight)
            nn.init.constant_(self.emb.bias, 0)
            self.input_pos_embed = nn.Parameter(torch.randn(1, L, hidden_size) * 0.02, requires_grad=True)
        
        self.all_pos_embed = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02, requires_grad=True)
        
        self.input_ln = nn.LayerNorm(hidden_size, eps=1e-6)
    
    def forward(self, x):
        '''
        Args:
            x: torch.Tensor, input tensor of shape (batch_size, *input_shape)
        Returns:
            torch.Tensor, output tensor of shape (batch_size, L, hidden_size)
        '''
        x = self.input_ln(self.mlp(self.emb(x))) * 0.02
        x = x + self.input_pos_embed
        x = x + self.all_pos_embed
        return x

    def activations(self, x):
        '''
        Returns a list of activations and their names, for probing uses. 
        Args:
            x: torch.Tensor, input tensor of shape (batch_size, *input_shape)
        Returns:
            ([activations ...], [names ...])
        '''
        u = self.emb(x)
        v = self.input_ln(u) + self.input_pos_embed + self.all_pos_embed
        v = nn.functional.layer_norm(v, (self.hidden_size,), eps=1e-6)
        return [u, *style(u), v, *style(v)]

    def activation_names(self):
        return ["emb", "emb_mean", "emb_std", "input_ln", "input_ln_mean", "input_ln_std"]
    
def shape_to_ld(shape, patch_size=2):
    '''
    Args:
        shape: (C, H, W) or (L, D)
    Returns:
        (L, D)
    '''
    if len(shape) == 3:
        return shape[1] * shape[2] // patch_size ** 2, shape[0] * patch_size ** 2
    elif len(shape) == 2:
        return shape
    assert False, "Invalid shape: {}".format(shape)

def unpatchify(x, patch_size=2, input_shape=(3, 16, 16)):
    '''
    Args:
        x: torch.Tensor, input tensor of shape (B, L, D)
        patch_size: int, patch size of the input tensor
        input_shape: (C, H, W) or (L, D)
    Returns:
        torch.Tensor, output tensor of shape (B, *input_shape)
    '''
    L, D = shape_to_ld(input_shape, patch_size)
    assert x.shape[1] == L
    assert x.shape[2] == D
    if len(input_shape) == 3:
        return rearrange(x, 'b (h w) (p1 p2 c)-> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=input_shape[1] // patch_size, w=input_shape[2] // patch_size)
    else:
        return x

from timm.models.vision_transformer import PatchEmbed, Mlp

@persistent_class
class Attention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.fused_attn = use_fused_attn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

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

@persistent_class
class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, per_pos_condition=False, cond_dim=None, repeat_bsz=False, double_norm=False, **block_kwargs):
        '''
        Args:
            hidden_size: int, hidden size of the transformer

            num_heads: int, number of attention heads

            mlp_ratio: float, ratio of mlp hidden dim to transformer hidden dim

            per_pos_condition: bool, 
                If False (defualt): cond have shape (B, cond_dim)
                If True: cond have shape (B, L, cond_dim)

            cond_dim: int, dimension of the condition tensor; if None (default): cond_dim = hidden_size

            repeat_bsz: 
                bool, whether to repeat the batch size; False by default. 
                If True: cond have shape (B / r, (L) (optional), cond_dim). Will repeat the batch size r times. 
        '''
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.per_pos_condition = per_pos_condition
        self.repeat_bsz = repeat_bsz
        self.double_norm = double_norm

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size if cond_dim is None else cond_dim, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        '''
        Args:
            x: torch.Tensor, input tensor of shape (B, L, D)
            c: torch.Tensor, condition tensor of shape 
                (B', cond_dim) if per_pos_condition is False
                (B', L, cond_dim) if per_pos_condition is True

                By default, B' = B; 
                if repeat_bsz is True, equivalent to ((B // B' B') ...), but only passes B' to save compute of adaln_modulation. 
        Returns:
            torch.Tensor, output tensor of shape (B, L, D)
        '''
        if not self.per_pos_condition:
            assert len(c.shape) == 2
            c = c.unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        
        if self.double_norm:
            shift_msa = self.norm1(shift_msa)
            scale_msa = self.norm1(scale_msa)
            shift_mlp = self.norm2(shift_mlp)
            scale_mlp = self.norm2(scale_mlp)

        if self.repeat_bsz:
            assert x.shape[0] % c.shape[0] == 0
            r = x.shape[0] // c.shape[0]
            shift_msa = repeat(shift_msa, 'b l d -> (repeat b) l d', repeat=r)
            scale_msa = repeat(scale_msa, 'b l d -> (repeat b) l d', repeat=r)
            gate_msa = repeat(gate_msa, 'b l d -> (repeat b) l d', repeat=r)
            shift_mlp = repeat(shift_mlp, 'b l d -> (repeat b) l d', repeat=r)
            scale_mlp = repeat(scale_mlp, 'b l d -> (repeat b) l d', repeat=r)
            gate_mlp = repeat(gate_mlp, 'b l d -> (repeat b) l d', repeat=r)
        # print("!!! F ", x.shape, shift_msa.shape, scale_msa.shape)
        x = x + gate_msa * self.attn(modulate(self.norm1(x.contiguous()), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x.contiguous()), shift_mlp, scale_mlp))
        return x

@persistent_class
class Transformer(nn.Module):
    def __init__(self, depth, hidden_size, num_heads, mlp_ratio=4.0, cond_dim=None, per_pos_condition=False, repeat_bsz=False, final_ln=True, double_norm=False, **block_kwargs):
        '''
        Args:
            depth: int, number of transformer blocks
            hidden_size: int, hidden size of the transformer
            num_heads: int, number of attention heads
            mlp_ratio: float, ratio of mlp hidden dim to transformer hidden dim

            cond_dim: int, dimension of the condition tensor; if None (default): cond_dim = hidden_size


            per_pos_condition: bool, 
                If False (default): cond have shape (B, cond_dim)
                If True: cond have shape (B, L, cond_dim)

            repeat_bsz: 
                bool, whether to repeat the batch size of condition;
                If True: cond have shape (B / r, (L) (optional), cond_dim). Will repeat the batch size r times.
                If False (default): cond have shape (B, cond_dim)

            final_ln: bool, whether to apply a final layer norm before output; 
                If True (default): apply a final layer norm
                If False: no final layer norm

            **block_kwargs: additional kwargs for the transformer block
        '''
        super().__init__()

        cond_dim = cond_dim if cond_dim is not None else hidden_size
        self.double_norm = double_norm
        all_blocks = [
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, per_pos_condition=per_pos_condition, repeat_bsz=repeat_bsz, cond_dim=cond_dim, double_norm=double_norm, **block_kwargs) for _ in range(depth)
        ]
        self.blocks = nn.ModuleList(all_blocks)

        self.input_ln = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cond_ln = nn.LayerNorm(cond_dim, elementwise_affine=False, eps=1e-6) 
        self.output_ln = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        )
        self.per_pos_condition = per_pos_condition
        self.repeat_bsz = repeat_bsz

        self.final_ln = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6) if final_ln else nn.Identity()
        self.initialize_weights()
    
    def compile_model(self):
        for i in range(len(self.blocks)):
            self.blocks[i] = torch.compile(self.blocks[i])

    def forward(self, x, c):
        '''
        Args:
            x: torch.Tensor, input tensor of shape (batch_size, hidden_size)
            c: torch.Tensor, (batch_size, hidden_size)
        Returns:
            torch.Tensor, output tensor of shape (batch_size, hidden_size)
        '''
        x = self.input_ln(x)
        c = self.cond_ln(c)
        for block in self.blocks:
            x = block(x, c)
        x = self.output_ln(x)
        shift, scale = self.final_adaLN_modulation(c.unsqueeze(1) if not self.per_pos_condition else c).chunk(2, dim=-1)
        
        if self.double_norm:
            shift = self.final_ln(shift)
            scale = self.final_ln(scale)

        if not self.repeat_bsz:
            assert c.shape[0] == x.shape[0]
        else:
            assert x.shape[0] % c.shape[0] == 0
            r = x.shape[0] // c.shape[0]
            shift = repeat(shift, 'b l d -> (repeat b) l d', repeat=r)
            scale = repeat(scale, 'b l d -> (repeat b) l d', repeat=r)
        
        x = modulate(x, shift, scale)
        x = self.final_ln(x)
        return x

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_adaLN_modulation[-1].bias, 0)



@persistent_class
class Encoder(nn.Module):
    def __init__(self, transformer_config, input_shape, output_shape, cond_dim, patch_size=2, pad_tokens=8, ret_std=False):
        '''
        Args:
            transformer_config: dict, configuration for the transformer
            input_shape: tuple, shape of the input tensor
            output_shape: tuple, shape of the output tensor
            cond_dim: int, dimension of the condition tensor
            patch_size: int, patch size of the input tensor
            pad_tokens: int, number of padding tokens
        
        Will concat inputs and outputs, to produce the outputs. 
        '''
        super().__init__()
        hidden_size = transformer_config['hidden_size']
        self.input_pre = InputProcess(input_shape, hidden_size, patch_size)
        self.transformer = Transformer(**transformer_config)
        self.output_pre = InputProcess(output_shape, hidden_size, patch_size)
        self.pad_pre = InputProcess((pad_tokens, 1), hidden_size, patch_size)
        self.c_cond = nn.Linear(cond_dim, hidden_size)
        L, D = shape_to_ld(output_shape, patch_size)

        self.ret_std = ret_std
        if not self.ret_std:
            self.ln_output = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
            self.output_proj = nn.Linear(hidden_size, D)
        else:
            self.output_proj = nn.Linear(hidden_size, 2 * D)
        self.pad_shape = (pad_tokens, 1)
        self.output_shape = output_shape
        self.patch_size = patch_size
    
    def get_transformer(self):
        return self.transformer

    def set_transformer(self, transformer):
        self.transformer = transformer

    def forward(self, x, c):
        '''
        Args:
            x: torch.Tensor, input tensor of shape (batch_size, *input_shape)
            c: torch.Tensor, condition tensor of shape (batch_size, cond_dim)
        Returns:
            if not self.ret_std:
                torch.Tensor, output tensor of shape (batch_size, *output_shape)
            else:
                (mean, std), output tensor of shape (batch_size, *output_shape)
        '''
        x = self.input_pre(x)
        pad = self.pad_pre(torch.zeros(x.shape[0], *self.pad_shape, device=x.device, dtype=x.dtype))
        pad_output = self.output_pre(torch.zeros(x.shape[0], *self.output_shape, device=x.device, dtype=x.dtype))
        x = self.transformer(torch.cat([x, pad, pad_output], dim=1), self.c_cond(c))

        last_elts = x[:, -pad_output.shape[1]:]
        if not self.ret_std:
            return unpatchify(self.ln_output(self.output_proj(last_elts)), self.patch_size, self.output_shape)
        else:
            otp = self.output_proj(last_elts)
            mean = unpatchify(otp[:, :, :otp.shape[2] // 2], self.patch_size, self.output_shape)
            std = unpatchify(otp[:, :, otp.shape[2] // 2:], self.patch_size, self.output_shape)
            std = ((std ** 2 + 4).sqrt() - std) / 2
            return mean, std
    
    def get_emb_layer(self):
        return self.input_pre

@persistent_class
class Decoder_head(nn.Module):
    '''
    A decoder head that takes in per-position condition, and generates many samples for each condition. 
    
    Optimized for generating many samples for each condition: the condition adaln parameters will only be computed once.  
    '''
    def __init__(self, condition_shape, output_shape, patch_size, head_mult, num_heads, pad_tokens=1, depth=3, noise_blocks=0, double_norm=False):
        '''
        Args:
            condition_shape: tuple, (L, D_cond)
            output_shape: tuple, (L, D) or (C, H, W); L should match; 
            patch_size: the patch size of the output tensor; 
            head_mult: int, multiplier for the head
            num_heads: int, number of heads
            noise_blocks: int, number of noise blocks to add to the noise input;
        '''
        super().__init__()
        print("!!! noise_blocks", noise_blocks)
        self.output_shape = output_shape
        self.patch_size = patch_size
        L, D = shape_to_ld(output_shape, patch_size)
        hidden_dim = D * head_mult
        
        self.noise_pre = InputProcess(output_shape, hidden_dim, patch_size, mlp_layers=noise_blocks)
        self.input_pre = InputProcess(output_shape, hidden_dim, patch_size, mlp_layers=noise_blocks)

        self.pad_pre = InputProcess((pad_tokens, 1), hidden_dim, patch_size)
        self.pad_shape = (pad_tokens, 1)
        self.transformer = Transformer(depth=depth, 
                                        hidden_size=hidden_dim, 
                                        num_heads=num_heads, 
                                        mlp_ratio=4.0, 
                                        cond_dim=condition_shape[1], 
                                        per_pos_condition=True, 
                                        repeat_bsz=True, 
                                        final_ln=False,
                                        double_norm=double_norm)
        self.proj_out = nn.Linear(hidden_dim, D)

    def compile_model(self):
        self.transformer.compile_model()

    def forward(self, c, input_skip, repeat_bsz=1, cfg_scale=1.0, drop_emb=None):
        '''
        Args:

            c: torch.Tensor, condition tensor of shape (B, L, D_cond)
            input_skip: torch.Tensor, input tensor of shape (B, *self.output_shape)

            repeat_bsz: int, repeat the batch size by this amount

            cfg_scale: float, the scale for the classifier-free guidance; 
                if != 1.0: will use drop_emb to guide the decoding. 

            drop_emb: torch.Tensor, (B, L, D_cond); the weaker emb if cfg enabled. None if cfg_scale == 1.0

        Returns:
            torch.Tensor, output tensor of shape ((repeat_bsz B), L, D)
        '''
        noises = torch.rand((c.shape[0] * repeat_bsz, *self.output_shape), device=c.device, dtype=c.dtype) - 0.5
        noises = self.noise_pre(noises) # (r B, L, hidden_dim)

        inputs = self.input_pre(input_skip) # (B, L, hidden_dim)
        inputs = repeat(inputs, 'b ... -> (repeat b) ...', repeat=repeat_bsz)
        inputs = inputs + noises

        if cfg_scale != 1.0:
            assert drop_emb is not None, "drop_emb must be provided if cfg_scale != 1.0"
            assert drop_emb.shape == c.shape, "drop_emb must have the same shape as c, found {} and {}".format(drop_emb.shape, c.shape)
            # c = torch.cat([c, drop_emb], dim=0)
            # inputs = rearrange(inputs, '(r b) ... -> r b ...', r = repeat_bsz)
            # inputs = torch.cat([inputs, inputs], dim=1)
            # inputs = rearrange(inputs, 'r b ... -> (r b) ...')
            print("old norm", (c ** 2).mean())
            print("emb norm", (drop_emb ** 2).mean())
            print("diff norm", ((c - drop_emb) ** 2).mean())
            c = c + (cfg_scale - 1.0) * (c - drop_emb)

        pad_tokens = self.pad_pre(torch.zeros(inputs.shape[0], *self.pad_shape, device=inputs.device, dtype=inputs.dtype)) # (r B, pad_tokens, hidden_dim)
        pad_cond = repeat(c.mean(dim=1, keepdim=True), 'b x d -> b (r x) d', r = pad_tokens.shape[1]) # [B, pad_tokens, D_cond]
        outputs = self.transformer(torch.cat([inputs, pad_tokens], dim=1), torch.cat([c, pad_cond], dim=1))
        outputs = self.proj_out(outputs[:, :inputs.shape[1], :])
        outputs = unpatchify(outputs, self.patch_size, self.output_shape)

        if cfg_scale != 1.0:
            # outputs = rearrange(outputs, '(r b) ... -> r b ...', r = repeat_bsz)
            # out_c, out_g = outputs[:, :outputs.shape[1] // 2], outputs[:, outputs.shape[1] // 2:]
            # out_c = rearrange(out_c, 'r b ... -> (r b) ...')
            # out_g = rearrange(out_g, 'r b ... -> (r b) ...')
            # delta = (out_c - out_g)
            # delta_dot = torch.einsum("b...,b...->b", delta, out_c)
            # orig = torch.einsum("b...,b...->b", out_c, out_c)
            # coeff = delta_dot / (orig + 1e-8)
            # coeff = coeff.reshape(-1, *([1] * (len(out_c.shape) - 1)))
            # delta = delta - coeff * out_c
            # outputs = out_c + (cfg_scale - 1.0) * delta
            pass

        return outputs

@persistent_class
class Decoder(nn.Module):
    def __init__(self, transformer_config, latent_shape, output_shape, cond_dim, patch_size=2, pad_tokens=8, head_mult=4, head_blocks=3, num_heads=6, has_l2_head=True, share_pos=False, noise_blocks=0, double_norm=False, num_dec_heads=0):
        '''
        Args:
            transformer_config: dict, configuration for the transformer
            latent_shape: tuple, shape of the latent tensor
            output_shape: tuple, shape of the output tensor
            cond_dim: int, dimension of the condition tensor
            patch_size: int, patch size of the input tensor
            pad_tokens: int, number of padding tokens
            head_mult: int, multiplier for the head
            head_blocks: int, number of blocks for the head
            noise_blocks: int, number of noise blocks to add to the noise input;
            num_dec_heads: int, number of decoder heads to add to the decoder; if 0: use the same head as the transformer head. 
        '''
        super().__init__()

        hidden_size = transformer_config['hidden_size']
        self.latent_pre = InputProcess(latent_shape, hidden_size, patch_size)
        self.transformer = Transformer(**transformer_config, double_norm=double_norm)
        self.share_pos = share_pos

        if not share_pos:
            self.output_pre = InputProcess(output_shape, hidden_size, patch_size)

        self.pad_pre = InputProcess((pad_tokens, 1), hidden_size, patch_size)
        self.c_cond = nn.Linear(cond_dim, hidden_size)
        self.pad_shape = (pad_tokens, 1)
        self.output_shape = output_shape
        self.patch_size = patch_size

        L, D = shape_to_ld(output_shape, patch_size)
        
        if has_l2_head:
            self.l2_head = nn.Linear(hidden_size, D)
        
        self.gen_head = Decoder_head(
            condition_shape=(L, hidden_size),
            output_shape=output_shape,
            patch_size=patch_size,
            head_mult=head_mult,
            num_heads=num_heads if num_dec_heads == 0 else num_dec_heads,
            pad_tokens=pad_tokens,
            depth=head_blocks,
            noise_blocks=noise_blocks,
            double_norm=double_norm,
        )
    def compile_model(self):
        self.transformer.compile_model()
        self.gen_head.compile_model()

    def get_transformer(self):
        return self.transformer
    
    def set_transformer(self, transformer):
        self.transformer = transformer
    
    def get_emb_layer(self):
        return self.latent_pre
    
    def get_condition(self, x, c):
        '''
        Args:
            x: torch.Tensor, input tensor of shape (batch_size, *latent_shape)
            c: torch.Tensor, condition tensor of shape (batch_size, cond_dim)
        Returns:
            torch.Tensor, condition tensor of shape (B, L, hidden_size); the conditions for the output tokens. 
        '''
        x = self.latent_pre(x)
        c = self.c_cond(c)
        pad = self.pad_pre(torch.zeros(x.shape[0], *self.pad_shape, device=x.device, dtype=x.dtype))
        if not self.share_pos:
            pad_output = self.output_pre(torch.zeros(x.shape[0], *self.output_shape, device=x.device, dtype=x.dtype))
            conditions = self.transformer(torch.cat([x, pad, pad_output], dim=1), c)
            return conditions[:, -pad_output.shape[1]:]
        else:
            conditions = self.transformer(torch.cat([x, pad], dim=1), c)
            return conditions[:, :x.shape[1]]
    
    def l2_recon(self, conditions):
        '''
        Args:
            conditions: torch.Tensor, condition tensor of shape (B, L, hidden_size)
        Returns:
            torch.Tensor, output tensor of shape (B, *output_shape)
        '''
        return unpatchify(self.l2_head(conditions), self.patch_size, self.output_shape)
    
    
    def forward(self, latent, temp=1.0, label_emb=None, unique_bsz=None, cfg_scale=1.0, drop_emb=None):
        B = latent.shape[0]
        i = unique_bsz if unique_bsz is not None else B

        condition = self.get_condition(x=latent[:i], c=label_emb[:i])
        if cfg_scale != 1.0:
            drop_cond = self.get_condition(x=latent[:i], c=drop_emb[:i])
        else:
            drop_cond = None

        if temp == 0.0:
            dec = self.l2_recon(repeat(condition, "b ... -> (repeat b) ...", repeat=B // i))
        else:
            skip = latent[:i]
            if latent.shape[1:] != self.output_shape:
                skip = torch.zeros((i, *self.output_shape), device=latent.device, dtype=latent.dtype)
            dec = self.gen_head(condition, input_skip=skip, repeat_bsz=B // i, cfg_scale=cfg_scale, drop_emb=drop_cond)

        return dec
    
def vit_kwargs():
    '''
    Args:
        **kwargs: additional kwargs for the transformer block
    Returns:
        dict, dictionary of the transformer block; can be used for Transformer(**config)
    '''
    return EasyDict(
        type="vit_ae",
        transformer_config=dict(depth=12, 
                                hidden_size=384, 
                                num_heads=6, 
                                mlp_ratio=4.0, 
                                qk_norm=True, 
                                attn_drop=0.0, 
                                drop=0.0),
        patch_size=2, 
        pad_tokens=8,
        decoder=dict(
            head_mult=4,
            head_blocks=3,
            has_l2_head=False,
        )
    )

class DownsampleEncoder(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert len(input_shape) == 3
        assert len(output_shape) == 3
    
    def forward(self, x, c):
        return F.interpolate(x, size=self.output_shape[1:], mode='bicubic')
        
@persistent_class
class UpsampleDecoder(nn.Module):
    def __init__(self, transformer_config, latent_shape, output_shape, cond_dim, patch_size=2, pad_tokens=8, head_mult=4, head_blocks=3, num_heads=6, noise_blocks=0, double_norm=True):
        super().__init__()
        self.latent_shape = latent_shape
        self.output_shape = output_shape
        self.decoder = Decoder(
            transformer_config=transformer_config,
            latent_shape=output_shape,
            output_shape=output_shape,
            cond_dim=cond_dim,
            patch_size=patch_size,
            pad_tokens=pad_tokens,
            head_mult=head_mult,
            head_blocks=head_blocks,
            num_heads=num_heads,
            has_l2_head=False, 
            share_pos=True,
            double_norm=double_norm,
            noise_blocks=noise_blocks,
        )
    
    def compile_model(self):
        self.decoder.compile_model()
    
    def upsample(self, x):
        return F.interpolate(x, size=self.output_shape[1:], mode='bicubic')

    def forward(self, latent, temp=1.0, label_emb=None, unique_bsz=None, cfg_scale=1.0, drop_emb=None):
        base = self.upsample(latent)
        decoded = self.decoder(base, temp, label_emb, unique_bsz, cfg_scale=cfg_scale, drop_emb=drop_emb)
        return decoded + base

def SiT_dict(**kwargs):
    '''
    Args:
        **kwargs: additional kwargs for the transformer block
    Returns:
        dict, dictionary of the transformer block; can be used for Transformer(**config)
    '''
    dict_base = dict(depth=12, hidden_size=384, patch_size=4, num_heads=6, qk_norm=True, attn_drop=0.0, proj_drop=0.0)
    dict_base.update(kwargs)
    return dict_base 


if __name__ == '__main__':
    input_shape = (3, 16, 16)
    output_shape = (64, 16, 16)
    cond_dim = 32
    batch_size = 2

    transformer_config = dict(depth=2, hidden_size=64, num_heads=4, mlp_ratio=2.0)

    # Instantiate encoder
    encoder = Encoder(
        transformer_config=transformer_config,
        input_shape=input_shape,
        output_shape=output_shape,
        cond_dim=cond_dim,
        patch_size=2,
        pad_tokens=2
    )

    decoder = Decoder(
        transformer_config=transformer_config,
        latent_shape=output_shape,
        output_shape=input_shape,
        cond_dim=cond_dim,
        patch_size=2,
        pad_tokens=2,
        head_mult=4,
        head_blocks=3,
    )

    # Dummy input and condition
    x = torch.randn(batch_size, input_shape[0], input_shape[1], input_shape[2])  # (B, C, H, W)
    c = torch.randn(batch_size, cond_dim)

    # Forward pass
    y = encoder(x, c)

    # Check output shape
    print(f"Output shape: {y.shape}", y.mean(), y.std())

    dec = decoder(y, c)
    print(f"Output shape: {dec.shape}", dec.mean(), dec.std())
    cond = decoder.get_condition(y, c)
    recon_elts = decoder.gen_recon(cond, repeat_bsz=8)
    print(f"Output shape: {recon_elts.shape}", recon_elts.mean(), recon_elts.std())

    dec_head = Decoder_head(
        condition_shape=(8, 32),
        output_shape=(8, 4),
        patch_size=2,
        head_mult=4,
        pad_tokens=2,
        depth=3,
    )
    conditions = torch.randn(batch_size, 8, 32)
    y = dec_head(conditions, repeat_bsz=8)
    print(f"Output shape: {y.shape}", y.mean(), y.std())
# %%
# need to fix the spike (probably by warmup)