import torch
import torch.nn as nn
from einops import rearrange, repeat
from utils.misc import EasyDict, sg
from utils.ckpt_utils import load_ckpt_epoch
from model.resnet import ResNet, build_latent_resnet
from utils.misc import EvalWrapper
from math import prod
import math

from utils.persistence import persistent_class
from model.clip_model import RN50_clip
from model.vgg import vgg16_perceptual
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from utils.misc import custom_compile


def cdist(x, y, eps=1e-8):
    """
    Args:
        x: [B, C1, D]
        y: [B, C2, D]
    Returns: [B, C1, C2]

    Same effect as torch.cdist, but faster.
    """
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    return (xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot).clamp(min=eps).sqrt()



def kernel(x, kernel_type="log"):
    """
    Kernel function for the contrastive loss
    x: the square of euclidean distance
    """
    if kernel_type == "log":
        return (x + 1e-3**2).sqrt().log()
    elif kernel_type == "sqrt":
        return (x + 1e-6).sqrt().sqrt()
    elif kernel_type == "dist":
        return (x + 1e-6).sqrt()
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

# @custom_compile(dynamic=True)
@torch.compile(dynamic=True)
def attn_loss_new(
    gen,
    fixed_pos,
    fixed_neg=None,
    weight_gen=None,
    weight_pos=None,
    weight_neg=None,
    R_list=None,
    transpose_aff=False,
    old_gen=None,
    step_size=1.0,
    temp_isqrt_d=False,
    same_group_mask=None,
    target_ratio=None, 
    n_sinkhorn_steps=0,
    has_repulsion=False,
    exp_affinity=False,
    no_ratio=False,
    proj_dim=0,
    **kwargs,
):
    """
    Args:
        gen: [B, C1, S]
        fixed_pos: [B, C2, S]
        fixed_neg: [B, C3, S] (optional, can be None)
        weight_gen: [B, C1] (optional; if None: weight is 1)
        weight_pos: [B, C2] (optional; if None: weight is 1)
        weight_neg: [B, C3] (optional; if None: weight is 1)
        R_list: a list of R values to use for the kernel function
        transpose_aff: whether to transpose the affinity matrix; if false: softmax on the targets; if true: average softmax on gen & targets.
        old_gen: [B, C1, S] (optional); if provided, use this for target computation.
        temp_isqrt_d: whether to use isqrt(d) to scale R
        same_group_mask: [B, C1, C1] (optional); if provided, mask out the corresponding pairs in gen.
        target_ratio: float; if specified, want max / mean_ratio to be target_ratio.
        n_sinkhorn_steps: int; number of additional sinkhorn steps to take. Odd steps: each gen normed to 1; even steps: each target normed to 1.
        proj_dim: int; if specified, project the features to the given dimension.
    Returns:
        loss: [batch_size]
        (optional) info: a dict with entries:

        # TODO: per sample normalization of output; per sample normalization of force.
    """
    # print("Owo")
    if R_list is None:
        R_list = [0.2]
    if len(kwargs) > 0:
        print("Additional keys:", kwargs.keys())
    old_dtype = fixed_pos.dtype
    # convert everything to float32

    if fixed_neg is None:
        fixed_neg = torch.zeros_like(fixed_pos[:, :0, :])
    if weight_pos is None:
        weight_pos = torch.ones_like(fixed_pos[:, :, 0])
    if weight_gen is None:
        weight_gen = torch.ones_like(gen[:, :, 0])
    if weight_neg is None:
        weight_neg = torch.ones_like(fixed_neg[:, :, 0])
    if old_gen is None:
        old_gen = gen
    
    # Convert everything to float32
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    gen = gen.float()
    old_gen = old_gen.float()
    weight_pos = weight_pos.float()
    weight_gen = weight_gen.float()
    weight_neg = weight_neg.float()
    
    old_gen = old_gen.detach()

    B, C_g, S = old_gen.shape
    B, C_p, S = fixed_pos.shape
    B, C_n, S = fixed_neg.shape
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
    targets_w = torch.cat(
        [weight_gen, weight_neg, weight_pos], dim=1
    )  # [B, C_g + C_n + C_p]

    if proj_dim > 0:
        proj_matrix = torch.randn(B, S, proj_dim).to(old_gen)
        old_gen = torch.einsum("bnd,bdm->bnm", old_gen, proj_matrix)
        gen = torch.einsum("bnd,bdm->bnm", gen, proj_matrix)
        targets = torch.einsum("bnd,bdm->bnm", targets, proj_matrix)
    info = dict()
    with torch.no_grad():
        dist = cdist(old_gen, targets)  # [B, C_g, C_g + C_n + C_p]
        scale = (dist * targets_w[:, None, :]).mean() / targets_w.mean()
        info["scale"] = scale.mean()
        scale_inputs = (scale / (math.sqrt(S))).clamp_min(
            1e-3
        )  # scale to make sure coords are N[0,1]
        dist = dist / scale.clamp_min(1e-3)  # dist_map: elts of order 1
        dist[:, :C_g, :C_g] = (
            dist[:, :C_g, :C_g] + torch.eye(C_g, device=dist.device) * 100
        )
        if same_group_mask is not None:
            dist[:, :C_g, :C_g][same_group_mask] = 100
    old_gen, targets, gen = (
        old_gen / scale_inputs,
        targets / scale_inputs,
        gen / scale_inputs,
    )
    with torch.no_grad():
        force_across_R = torch.zeros_like(old_gen)
        for R in R_list:
            temp = R if not temp_isqrt_d else R / math.sqrt(S)
            R_coeff = torch.zeros_like(dist)  # [B, C_g, C_g + C_n + C_p];
            affinity = (-dist / temp).softmax(dim=-1)  # [B, C_g, C_g + C_n + C_p]
            entropy = (-affinity * (affinity + 1e-8).log()).sum(dim=-1)
            info[f'temp_{R}'] = temp
            info[f'effective_samples_{R}'] = (entropy.mean()).exp()
            if transpose_aff:
                affinity = (
                    (affinity * (-dist / temp).softmax(dim=-2)).clamp_min(0.0).sqrt()
                )
            for i in range(n_sinkhorn_steps):
                if i % 2 == 0:
                    affinity = affinity / (affinity.sum(dim=-1, keepdim=True) + 1e-3)
                else:
                    affinity = affinity / (affinity.sum(dim=-2, keepdim=True) + 1e-3)
            if exp_affinity:
                affinity = (-dist / temp).exp()
            
            info[f'mean_affinity_{R}'] = affinity.mean()
            info[f'max_affinity_{R}'] = affinity.max(dim=-1).values.mean()
            ratio = affinity.max(dim=-1).values.mean() / affinity.mean()
            if target_ratio is not None:
                k = math.log(target_ratio) / math.log(ratio)
                affinity = affinity.pow(k)
            info[f'max_to_mean_affinity_{R}'] = affinity.max(dim=-1).values.mean() / affinity.mean()

            # weight by the weights
            affinity = affinity * targets_w[:, None, :]  # [B, C_g, C_g + C_n + C_p]

            # Make sure pos_ker ~= neg_ker, for normalization of forces
            info[f"pos_ker_{R}"] = affinity[:, :, C_g + C_n :].sum(dim=-1).mean()
            info[f"neg_ker_{R}"] = affinity[:, :, : C_g + C_n].sum(dim=-1).mean()
            ratio = info[f"neg_ker_{R}"] / (info[f"pos_ker_{R}"] + 1e-3)
            if no_ratio:
                ratio = torch.ones_like(ratio)
            info[f"ratio_{R}"] = ratio.mean()
            affinity[:, :, C_g + C_n :] = (
                affinity[:, :, C_g + C_n :] * ratio
            )  # make sure: pos sum ~= neg sum
            R_coeff[:, :, C_g + C_n :] = affinity[:, :, C_g + C_n :] * (
                affinity[:, :, : C_g + C_n].sum(dim=-1, keepdim=True)
            )
            R_coeff[:, :, : C_g + C_n] = -affinity[:, :, : C_g + C_n] * (
                affinity[:, :, C_g + C_n :].sum(dim=-1, keepdim=True)
            )
            if has_repulsion:
                R_coeff[:, :, C_g + C_n :] = affinity[:, :, C_g + C_n :]
                R_coeff[:, :, : C_g + C_n] = -affinity[:, :, : C_g + C_n]

            # estimation of force, when **no cancellation**.
            norm_est = (
                ((affinity * dist) ** 2)
                .mean(dim=(-1, -2), keepdim=True)
                .clamp_min(1e-8)
                .sqrt()
            )
            info[f"norm_est_{R}"] = norm_est.mean()
            R_coeff = R_coeff / norm_est
            total_force_R = torch.einsum("biy,byx->bix", R_coeff, targets)
            if has_repulsion:
                total_force_R = total_force_R - R_coeff.sum(dim=-1,keepdim=True) * old_gen
            info[f"f_norm_{R}"] = (total_force_R**2).mean()
            force_across_R = (
                force_across_R
                + total_force_R / (total_force_R**2).mean().clamp_min(1e-8).sqrt()
            )

        goal = (old_gen + force_across_R * step_size).detach()

    loss = ((gen - goal) ** 2).mean(dim=(-1, -2)).to(old_dtype)
    info["diff_base"] = ((gen - old_gen) ** 2).mean()
    return loss, info


@custom_compile(dynamic=True)
def attn_contra_loss(
    target,
    recon,
    return_info=False,
    sample_norm=True,
    weight_r=None,
    scale_dist=False,
    no_R_norm=False,
    no_global_norm=False,
    new_R_norm=False,
    scale_dist_normed=True,
    R_list=None,
    coord_norm=False,
    softmax=True,
    gen_attend_data=True,
    data_attend_gen=True,
    softmax_p=0.5,
    norm_R_force=False,
    scale_by_pos=False,
    old_recon=None,
    same_group_mask=None,
):
    """
    Best recommendation:
        sample_norm: True
        scale_disr_normed: True
        Other: False
        This will align different features. Other settings are mostly for exploration.
    Args:
        target: [batch_size, C1, S]
        recon: [batch_size, C2, S]
        return_info: whether to return the info dict
        sample_norm: whether to normalize the loss by the sample norm;
            if enabled: loss will have shape (B, )
        weight_r:
            The weight for negative samples. None or shape (B, C2).
            When enabled: all repulsions will be weighted by weight_r.
        softmax: whether to use softmax to normalize the affinity matrix.
        scale_by_pos: whether to scale the dist only by distance of positive samples.
        old_recon: [B, C2, S] (optional); if provided, use this for target computation.
        same_group_mask: [B, C2, C2] (optional); if provided, mask out the corresponding pairs in recon.
    Returns:
        loss: [batch_size]
        (optional) info: a dict with entries:
            force_norm: the norm of the force.
            prec: the precision of the target.

    """
    if R_list is None:
        R_list = [0.2]
    if old_recon is None:
        old_recon = recon
    old_recon = old_recon.detach()
    B, C1, S = target.shape
    B, C2, S = old_recon.shape
    if coord_norm:
        # normalize, to make sure every coordinate has mean 0 & var 1
        coord_mean = torch.cat([target, old_recon], dim=1).mean(dim=(0, 1)).detach()
        coord_std = torch.cat([target, old_recon], dim=1).std(dim=(0, 1)).detach()
        target = (target - coord_mean) / (coord_std + 1e-3)
        old_recon = (old_recon - coord_mean) / (coord_std + 1e-3)

    with torch.no_grad():
        pos_neg = torch.cat([target, old_recon], dim=1)
        dist = cdist(pos_neg, pos_neg)  # [B, C1 + C2, C1 + C2]
        # assert target.shape[1] == 1
        scale = dist.mean()
        if sample_norm:
            scale = dist.mean(dim=(-1, -2), keepdim=True)
        if scale_by_pos:
            scale = dist[:, :C1, :C1].mean()
            if sample_norm:
                scale = dist[:, :C1, :C1].mean(dim=(-1, -2), keepdim=True)
    if scale_dist:
        target, old_recon, pos_neg, recon = (
            target / scale,
            old_recon / scale,
            pos_neg / scale,
            recon / scale,
        )
    if scale_dist_normed:
        assert not scale_dist
        scale_2 = scale / (math.sqrt(S))
        target, old_recon, pos_neg, recon = (
            target / scale_2,
            old_recon / scale_2,
            pos_neg / scale_2,
            recon / scale_2,
        )
    with torch.no_grad():
        dist = dist + torch.eye(C1 + C2, device=dist.device) * 100 * scale.mean()
        if same_group_mask is not None:
            dist[:, C1:, C1:][same_group_mask] = 100 * scale.mean()

        info = {
            "nn_data": dist[:, :C1].min(dim=-1).values.mean(),
            "nn_samples": dist[:, C1:].min(dim=-1).values.mean(),
            "scale": scale.mean(),
            "scale_normed": scale.mean() / (math.sqrt(S)),
        }

        dist = dist / scale
        info["dist_std"] = (
            (dist - torch.eye(C1 + C2, device=dist.device) * 100)
            .std(dim=(-1, -2))
            .mean()
        )

        total_pos = torch.zeros_like(dist)

        for R in R_list:
            if softmax:
                affinity = torch.ones_like(dist)
                if gen_attend_data:
                    affinity = affinity * torch.pow(
                        (-dist / R).softmax(dim=-1), softmax_p
                    )
                if data_attend_gen:
                    affinity = affinity * torch.pow(
                        (-dist / R).softmax(dim=-2), softmax_p
                    )
            else:
                affinity = (-dist / R).exp()

            if weight_r is not None:
                affinity[:, :, C1:] = affinity[:, :, C1:] * weight_r[:, None, :]

            if sample_norm:
                norm_est = (
                    ((affinity * dist) ** 2)
                    .mean(dim=(-1, -2), keepdim=True)
                    .clamp_min(1e-8)
                    .sqrt()
                )
                if norm_R_force:
                    norm_est = norm_est * affinity.mean(dim=(-1, -2), keepdim=True)
            cur_force = torch.zeros_like(dist)
            info[f"pos_ker_{R}"] = affinity[:, C1:, :C1].sum(dim=-1).mean()
            info[f"neg_ker_{R}"] = affinity[:, C1:, C1:].sum(dim=-1).mean()
            cur_force[:, C1:, C1:] = -affinity[:, C1:, C1:] * (
                affinity[:, C1:, :C1].sum(dim=-1, keepdim=True)
            )
            cur_force[:, C1:, :C1] = affinity[:, C1:, :C1] * (
                affinity[:, C1:, C1:].sum(dim=-1, keepdim=True)
            )
            if not sample_norm:
                norm_est = ((cur_force * dist) ** 2).mean().clamp_min(1e-8).sqrt()

            if new_R_norm:
                norm_est = (affinity * dist).mean().clamp_min(1e-8)
                if norm_R_force:
                    norm_est = norm_est * affinity.mean(dim=(-1, -2), keepdim=True)

            if not no_R_norm:
                cur_force = cur_force / norm_est
            total_pos = total_pos + cur_force
            info[f"norm_{R}"] = norm_est.mean()
            info[f"norm_{R}_std"] = (
                norm_est.std()
                if (len(norm_est.shape) > 0 and norm_est.shape[0] > 1)
                else 0
            )

        sum_forces = torch.einsum("biy,byx->bix", total_pos[:, C1:], pos_neg)
        info["f_norm"] = (sum_forces**2).mean()
        if not no_global_norm:
            sum_forces = sum_forces / ((sum_forces**2).mean().clamp_min(1e-8).sqrt())
        goal = sg(old_recon + sum_forces)

    loss = ((recon - goal) ** 2).mean(dim=(-1, -2))
    if return_info:
        return loss, info
    return loss


def group_contra_loss(
    pos,
    gen,
    neg=None,
    pos_w=None,
    gen_w=None,
    neg_w=None,
    kernel_type="log",
    return_info=True,
    old_gen=None,
    same_group_mask=None,
    **contra_dict,
):
    """
    Args:
        pos: [batch_size, C1, S]
        gen: [batch_size, C2, S]
        neg: [batch_size, C3, S]
        pos_w: [batch_size, C1] | None; default to 1
        gen_w: [batch_size, C2] | None; default to 1
        neg_w: [batch_size, C3] | None; default to 1
        same_group_mask: [batch_size, C2, C2] | None; default to None; When None, should use diagonal mask
        use_base: whether to use the base loss
        kernel_type: the type of kernel function
        use_scale: whether to use the scale loss
        return_info: whether to return the info dict
        old_gen: [batch_size, C2, S] (optional); if provided, use this for target computation.
    Returns:
        loss: [batch_size] or scalar
        (optional) info: a dict
    """
    if kernel_type == "attn":
        # gen and neg are repelled from each other and attracted to pos.
        # loss is only on gen.
        recon = gen
        if neg is not None:
            recon = torch.cat([gen, neg], dim=1)

        if gen_w is None:
            gen_w = torch.ones_like(gen[:, :, 0])
        if neg is not None and neg_w is None:
            neg_w = torch.ones_like(neg[:, :, 0])

        weight_r = gen_w
        if neg is not None:
            weight_r = torch.cat([gen_w, neg_w], dim=1)
        
        old_recon = None
        if old_gen is not None:
            old_recon = old_gen
            if neg is not None:
                old_recon = torch.cat([old_gen, neg], dim=1)

        cur_same_group_mask = same_group_mask
        if neg is not None and same_group_mask is not None:
            B, C_gen, _ = gen.shape
            C_neg = neg.shape[1]
            new_mask = torch.zeros(
                B, C_gen + C_neg, C_gen + C_neg, dtype=torch.bool, device=gen.device
            )
            new_mask[:, :C_gen, :C_gen] = same_group_mask
            cur_same_group_mask = new_mask

        return attn_contra_loss(
            pos,
            recon,
            weight_r=weight_r,
            **contra_dict,
            return_info=return_info,
            old_recon=old_recon,
            same_group_mask=cur_same_group_mask,
        )

    elif kernel_type == "attn_new":
        # note: won't have weight_r here...
        # print(gen_w.shape, pos_w.shape)
        # print(gen.shape, pos.shape)
        return attn_loss_new(
            gen=gen,
            fixed_pos=pos,
            fixed_neg=neg,
            weight_gen=gen_w,
            weight_pos=pos_w,
            weight_neg=neg_w,
            old_gen=old_gen,
            same_group_mask=same_group_mask,
            **contra_dict,
        )
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


if __name__ == "__main__":
    pos = torch.randn(10, 10, 10)
    gen = torch.randn(10, 10, 10)
    neg = torch.randn(10, 10, 10)
    # loss, info = group_contra_loss(pos, gen, neg, kernel_type="attn_new", target_ratio=4.0)
    loss, info = attn_loss_new(gen, pos, neg, target_ratio=4.0, n_sinkhorn_steps=2)
    print(loss)
    print(info)