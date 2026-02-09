import torch
from einops import rearrange

from utils.misc import custom_compile
import torch.nn.functional as F

@custom_compile(dynamic=True)
def group_features(x, patch_group_size):
    '''
    Args:
        x: [B, C, H, W]
        patch_group_size: int

    Returns:
        [B, H * W // (patch_group_size ** 2), patch_group_size ** 2, C]
        (pad when not multiple of patch_group_size)
        if max(h, w) <= patch_group_size, return [B, 1, H * W, C]
    '''
    h, w = x.shape[-2:]
    if max(h, w) <= patch_group_size:
        return rearrange(x, 'b c h w -> b 1 (h w) c')
    h_pad = (patch_group_size - h % patch_group_size) % patch_group_size
    w_pad = (patch_group_size - w % patch_group_size) % patch_group_size
    x = F.pad(x, (0, w_pad, 0, h_pad))
    return rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_group_size, p2=patch_group_size)

def unfold_feats_merge(x, kernel):
    y = unfold_feats(x, kernel)
    y = rearrange(y, 'b k l h w c -> b (k l c) h w')
    return y
    
@custom_compile(dynamic=True)
def unfold_feats(x, kernel, return_corr=False):
    '''
    Args:
        x: (B, C, H, W)
        kernel: int
    Returns:
        feats: (B, kernel, kernel, H, W, C)
        (optional) corr: (B, kernel * kernel - 1, C); correlation of the feats. 
    '''
    B, C, H, W = x.shape
    assert kernel > 1, "kernel must be greater than 1"
    assert H > 1 and W > 1, "H and W must be greater than 1"
    device = x.device
    b_idx, kh_idx, kw_idx, h_idx, w_idx = torch.meshgrid(
        torch.arange(B, device=device),
        torch.arange(kernel, device=device),
        torch.arange(kernel, device=device),
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    h_idx = (h_idx + kh_idx) % (2 * H - 2)
    h_idx = torch.minimum(h_idx, 2 * H - 2 - h_idx)
    w_idx = (w_idx + kw_idx) % (2 * W - 2)
    w_idx = torch.minimum(w_idx, 2 * W - 2 - w_idx)
    indices = b_idx * H * W + h_idx * W + w_idx
    x_flatten = rearrange(x, 'b c h w -> (b h w) c')
    result = x_flatten[indices.flatten()]
    unfolded_feats = result.reshape(B, kernel, kernel, H, W, C)
    if return_corr:
        feats_mean = unfolded_feats.mean(dim=(3,4), keepdim=True)
        feats_std = unfolded_feats.std(dim=(3,4), keepdim=True)
        feats_normalized = (unfolded_feats - feats_mean) / (feats_std + 1e-3)
        feats_normalized = rearrange(feats_normalized, 'b k l h w c -> b (k l) c (h w)')
        first_elts = feats_normalized[:, 0:1, :, :]
        other_elts = feats_normalized[:, 1:, :, :]
        dot_prod = (first_elts * other_elts).mean(dim=-1)
        return unfolded_feats, dot_prod
    else:
        return unfolded_feats


if __name__ == '__main__':
    x = torch.randn(1, 128, 256, 256, device='cuda')
    feats, corr = unfold_feats(x, 3, return_corr=True)
    print(feats.shape, corr.shape)
    print(corr[:, 0].mean(), corr[:, 1].mean())