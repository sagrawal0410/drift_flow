# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

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


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        noise_classes=0,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.noise_classes = noise_classes
        self.initialize_weights()
        if noise_classes > 0:
            self.noise_embed = nn.Embedding(noise_classes, hidden_size)
            self.noise_embed.weight.data.normal_(mean=0.0, std=0.02)

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

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
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

    def forward(self, x, t, y):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        if self.noise_classes > 0:
            noise_labels = torch.randint(0, self.noise_classes, (y.shape[0],), device=y.device)
            y = y + self.noise_embed(noise_labels)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of SiT, but also batches the unconSiTional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

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
#                                   SiT Configs                                  #
#################################################################################

SiT_models = {
    'SiT-XL/2': {'depth': 28, 'hidden_size': 1152, 'patch_size': 2, 'num_heads': 16},
    'SiT-XL/4': {'depth': 28, 'hidden_size': 1152, 'patch_size': 4, 'num_heads': 16},
    'SiT-XL/8': {'depth': 28, 'hidden_size': 1152, 'patch_size': 8, 'num_heads': 16},
    'SiT-L/2':  {'depth': 24, 'hidden_size': 1024, 'patch_size': 2, 'num_heads': 16},
    'SiT-L/4':  {'depth': 24, 'hidden_size': 1024, 'patch_size': 4, 'num_heads': 16},
    'SiT-L/8':  {'depth': 24, 'hidden_size': 1024, 'patch_size': 8, 'num_heads': 16},
    'SiT-B/2':  {'depth': 12, 'hidden_size': 768, 'patch_size': 2, 'num_heads': 12},
    'SiT-B/4':  {'depth': 12, 'hidden_size': 768, 'patch_size': 4, 'num_heads': 12},
    'SiT-B/8':  {'depth': 12, 'hidden_size': 768, 'patch_size': 8, 'num_heads': 12},
    'SiT-S/2':  {'depth': 12, 'hidden_size': 384, 'patch_size': 2, 'num_heads': 6},
    'SiT-S/4':  {'depth': 12, 'hidden_size': 384, 'patch_size': 4, 'num_heads': 6},
    'SiT-S/8':  {'depth': 12, 'hidden_size': 384, 'patch_size': 8, 'num_heads': 6},
}


class FMLoss(torch.nn.Module):
    def __init__(self, data_shape=(3, 32, 32), mu=2.0, sigma=2.0, sit_type="SiT-B/2",
                 depth=None, patch_size=None, num_heads=None, hidden_size=None, **kwargs):
        '''
        Args:
            net: nn.Module, the model to be trained; 
                takes in x, t, r, **labels; return v prediction. 
            data_shape: tuple, the shape of the data.
            mu: float, the mean of the normal distribution to sample t;
            sigma: float, the standard deviation of the normal distribution to sample t.
            sit_type: str, the type of SiT model to use.
            depth: int, the depth of the transformer.
            patch_size: int, the size of the patches.
            num_heads: int, the number of attention heads.
            hidden_size: int, the hidden size of the transformer.
        '''
        super().__init__()
        
        sit_kwargs = {}
        if sit_type is not None:
            sit_kwargs.update(SiT_models[sit_type])

        # Overwrite with any explicitly provided values
        if depth is not None: sit_kwargs['depth'] = depth
        if patch_size is not None: sit_kwargs['patch_size'] = patch_size
        if num_heads is not None: sit_kwargs['num_heads'] = num_heads
        if hidden_size is not None: sit_kwargs['hidden_size'] = hidden_size
        
        # Check if all required args are present
        required_args = ['depth', 'patch_size', 'num_heads', 'hidden_size']
        if not all(arg in sit_kwargs for arg in required_args):
            raise ValueError(f"Missing one of the required arguments: {required_args} either via sit_type or direct parameters.")

        sit_kwargs.update(kwargs)

        self.net = SiT(
            input_size=data_shape[1], 
            in_channels=data_shape[0], 
            num_classes=1001, 
            **sit_kwargs
        )
        self.data_shape = data_shape
        self.mu = mu
        self.sigma = sigma

    def forward(self, x, class_labels): 
        '''
        Args:
            x: torch.Tensor, (B, *data_shape)
            class_labels: torch.Tensor, (B,)
        Returns:
            loss: torch.Tensor, (B,)
            info: dict, info to be logged for debugging.
        '''
        B = x.shape[0]

        t = torch.randn(B, device=x.device) * self.sigma + self.mu
        t = torch.sigmoid(t)

        eps = torch.randn_like(x)
        t_shape_goal = (-1, *([1] * (len(x.shape) - 1)))
        xt = x * (1 - t).reshape(t_shape_goal) + eps * (t.reshape(t_shape_goal)) 

        v_pred = self.net(xt, t, class_labels)
        v_tgt = eps - x

        loss = (v_pred - v_tgt) ** 2
        loss = loss.mean(dim=tuple(range(1, len(x.shape))))

        return loss.mean(), {}
    
    def sample(self, bsz, n_steps, class_labels, sampler="heun"):
        '''
        Args:
            B: int, the number of samples to generate.
            extra_kwargs: dict, extra kwargs to be passed to the net;
                the net takes in x, t, r, **extra_kwargs; return v prediction. 
            sampler: str, the sampler to use;
                "euler": Euler solver;
                "edm": EDM solver. 
        Returns:
            x: torch.Tensor, (B, *data_shape)
        '''
        device = next(self.net.parameters()).device
        with torch.no_grad():
            x = torch.randn(bsz, *self.data_shape, device=device)
            if sampler == "euler":
                t = torch.linspace(1, 0, n_steps + 1, device=device)
            else:
                t_steps = torch.linspace(0, 1, n_steps + 1, device=device)
                sigma_list = (
                    torch.linspace(
                        80 ** (1 / 7), 0.002 ** (1 / 7), n_steps - 1, device=device
                    )
                    ** 7
                )
                t_steps[1:-1] = 1 / (sigma_list + 1)
                t = 1 - t_steps # we are flipping the convention here. 
            
            for i in range(n_steps):

                t_cur = torch.ones(bsz, device=device) * t[i]
                t_nx = torch.ones(bsz, device=device) * t[i + 1]

                v = self.net(
                    x, t_cur, class_labels
                )
                x_euler = x + v * (t[i + 1] - t[i])

                if sampler == "heun":
                    if i < n_steps - 1:
                        v_next = self.net(
                            x_euler, t_nx, class_labels
                        )
                        x = x + 0.5 * (t[i + 1] - t[i]) * (v + v_next)
                    else:
                        x = x_euler
                else:
                    assert sampler == "euler"
                    x = x_euler
        return x

if __name__ == "__main__":
    fm_model = FMLoss(mu=1.2, sigma=1.2, sit_type="SiT-B/2", data_shape=(3, 32, 32)).to("cuda")
    x = torch.randn(1, 3, 32, 32).to("cuda")
    class_labels = torch.randint(0, 1000, (1,)).to("cuda")
    loss = fm_model(x, class_labels)
    print(loss)

    x = fm_model.sample(1, 100, class_labels, sampler="heun")
    print(x.shape)