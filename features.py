import torch
import torch.nn as nn
from einops import rearrange, repeat
from utils.misc import EasyDict, sg
from utils.ckpt_utils import load_ckpt_epoch
from model.resnet import ResNet, build_latent_resnet
from model.mae_convnext import build_mae_convnext
from math import prod
import math

from model.mae_resnet import build_mae_resnet
from model.mae_resnet_gn import build_mae_resnet_gn
from model.mae_vit import build_mae_vit
from utils.persistence import persistent_class
from model.clip_model import RN50_clip
from model.vgg import vgg16_perceptual
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from utils.misc import custom_compile
from energy_loss import group_contra_loss
from functools import partial
from utils.feats import unfold_feats_merge
import copy

class EvalWrapper(nn.Module):
    '''
    Wrapper for a model, to keep it frozen & always in eval mode.
    '''
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def train(self, mode=True):
        self.model.eval()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def state_dict(self, *args, **kwargs): # no state dict!
        return {}
    
    def load_state_dict(self, *args, **kwargs): # no load state dict!
        return
    

# @persistent_class
class FeatureExtractor(nn.Module):
    def __init__(self, input_shape, max_features=None):
        """
        input_shape: the shape of the input
        max_features: the maximum number of features to extract.
            If None, all features are extracted; otherwise: randomly select max_features features.
        self should support functions
            f_map:
                takes in (B, *input_shape);
                returns a dict {name: (B, F, D) or (B, F, G, D)}.
            feature_names:
                returns a list of names for the features.
        """
        super().__init__()
        self.input_shape = torch.Size(input_shape)
        self.max_features = max_features

    def extract_features(self, x):
        """
        x: (..., *input_shape)
        Returns: Dict[str, Tensor(..., F, D)]
        """
        assert (
            x.shape[-len(self.input_shape) :] == self.input_shape
        ), f"x.shape: {x.shape}, self.input_shape: {self.input_shape}"
        x_reshaped = x.reshape(-1, *self.input_shape)  # (B, *input_shape)
        f_result = self.f_map(x_reshaped)  # Dict[str, (B, F, D)]
        return {
            k: v.reshape(x.shape[: -len(self.input_shape)] + v.shape[1:])
            for k, v in f_result.items()
        }

    def _subsample_features(self, tensor, indices):
        '''
        Subsample the features.
        Args:
            tensor: (B, F, D)
            indices: (B, F_new)
        Returns:
            (B, F_new, D)
        '''
        if tensor is None:
            return None

        batch_size = tensor.size(0)
        index = (
            indices.unsqueeze(1)
            .unsqueeze(3)
            .expand(
                batch_size,
                tensor.size(1),
                self.max_features,
                tensor.size(3),
            )
        )
        return torch.gather(tensor, 2, index)

    # @torch.compile
    # @custom_compile()
    def forward(
        self,
        target,
        recon,
        fixed_neg=None,
        target_w=None,
        recon_w=None,
        fixed_neg_w=None,
        contra_dict=dict(
            use_base=True,
            kernel_type="log",
            use_scale=False,
            no_e=0.0,
            contra_factor=0.0,
        ),
        old_recon=None,
        debug=False
    ):
        """
        Args:
            target: (B, F1 (optional), *input_shape)
            recon: (B, F2, *input_shape)
            fixed_neg: (B, F3, *input_shape)
            ..._w: weights for samples
            contra_dict: the dictionary of contrastive loss parameters;
            old_recon: (B, F2, *input_shape) (optional); if provided, use this for target computation.
        Returns:
            loss: (B,); the total loss;
            info: a dict with entries:
                {id}l2_recon: avg d_recon; id is the id of the feature (skipped if length is 1)
                {id}l2_contra: avg d_contra; id is the id of the feature (skipped if length is 1)
        """
        def extract_features(target, fixed_neg, recon, old_recon):
            with torch.no_grad():  # disable gradient here
                if len(target.shape) == len(self.input_shape) + 1:
                    target = target.unsqueeze(1) 
                target_f_dict = self.extract_features(target)  # Dict[str, [B, F1, F, D]]
                if fixed_neg is not None:
                    fixed_neg_f_dict = self.extract_features(fixed_neg)
                else:
                    fixed_neg_f_dict = {k: None for k in target_f_dict}

            recon_f_dict = self.extract_features(recon)  # Dict[str, [B, F2, F, D]]
            if old_recon is not None:
                old_recon_f_dict = self.extract_features(old_recon)  # Dict[str, [B, F2, F, D]]
            else:
                old_recon_f_dict = recon_f_dict
            return target_f_dict, recon_f_dict, fixed_neg_f_dict, old_recon_f_dict
        
        target_f_dict, recon_f_dict, fixed_neg_f_dict, old_recon_f_dict = extract_features(target, fixed_neg, recon, old_recon)

        total_loss = 0
        all_info = dict()
        feature_names = list(target_f_dict.keys())
        for f_name in feature_names:
            if debug:
                before_time = time.time()
            target_f, recon_f, fixed_neg_f, old_recon_f = (
                target_f_dict[f_name],
                recon_f_dict[f_name],
                fixed_neg_f_dict[f_name],
                old_recon_f_dict[f_name],
            )
            # each: [B, X, F, G (optional), D]
            g = 1 # For each sample, the group size per feature. 
            if len(recon_f.shape) == 5:
                # when multiple groups are used, the feature shape is [B, X, F, G, D]
                g = recon_f.shape[3]
                def flatten(x): 
                    if x is None:
                        return None
                    return rearrange(x, "b x f g d -> b x f (g d)")
                target_f, recon_f, old_recon_f, fixed_neg_f = (
                    flatten(target_f),
                    flatten(recon_f),
                    flatten(old_recon_f),
                    flatten(fixed_neg_f),
                )   
            
            assert len(recon_f.shape) == 4

            if self.max_features is not None and target_f.shape[2] > self.max_features:
                batch_size = target_f.size(0)
                original_dim_size = target_f.size(2)
                indices = torch.rand(
                    batch_size, original_dim_size, device=target_f.device
                ).argsort(dim=1)[:, : self.max_features]
                target_f = self._subsample_features(target_f, indices)
                recon_f = self._subsample_features(recon_f, indices)
                old_recon_f = self._subsample_features(old_recon_f, indices)
                fixed_neg_f = self._subsample_features(fixed_neg_f, indices)
            
            B, X_recon, F_recon, _ = recon_f.shape
            same_group_mask = None
            if g > 1:
                indices = torch.arange(X_recon * g, device=recon_f.device)
                group_ids = indices // g
                mask = group_ids[:, None] == group_ids[None, :]
                same_group_mask = mask.unsqueeze(0).expand(B * F_recon, -1, -1)
                # print("same_group_mask", same_group_mask.shape, same_group_mask.float().mean(dim=0))

            def reshape_elts(features, w):
                '''
                features: (B, X, F, D)
                w: (B, X) | None
                Returns: ((B F), X, D) & ((B F), X) | None
                '''
                if features is None:
                    assert w is None
                    return None, None
                assert len(features.shape) == 4
                f = features.shape[2]
                features = rearrange(features, "b id f (g d) -> (b f) (id g) d", g=g) # [B * F, C_id * G, D]
                if w is not None:
                    w = repeat(w, "b i -> (b f) (i g)", f=f, g=g)
                return features, w

            B = target_f.shape[0]
            target_f, cur_target_w = reshape_elts(target_f, target_w)
            recon_f, cur_recon_w = reshape_elts(recon_f, recon_w)
            old_recon_f, _ = reshape_elts(old_recon_f, None)
            fixed_neg_f, cur_fixed_neg_w = reshape_elts(fixed_neg_f, fixed_neg_w)
            if debug:
                middle_time = time.time()
        
            loss, info = group_contra_loss(
                pos=target_f,
                gen=recon_f,
                neg=fixed_neg_f,
                pos_w=cur_target_w,
                gen_w=cur_recon_w,
                neg_w=cur_fixed_neg_w,
                old_gen=old_recon_f,
                same_group_mask=same_group_mask,
                **contra_dict,
                return_info=True,
            )
            if debug:
                after_loss_time = time.time()
            loss = rearrange(loss, "(b f) -> b f", b=B)
            loss = loss.mean(dim=1)

            total_loss = loss + total_loss
            cur_id = f_name
            all_info[f"{cur_id}loss"] = loss.mean()
            for k, v in info.items():
                all_info[f"{cur_id}{k}"] = v
            if debug:
                after_time = time.time()
                print(f"Time taken for {f_name}: {after_time - before_time} seconds, {middle_time - before_time} seconds for middle, {after_loss_time - middle_time} seconds for loss, for {f_name} with shape {target_f.shape}, {recon_f.shape}, {fixed_neg_f.shape}, {old_recon_f.shape}")
        return total_loss, all_info

# @persistent_class
class Flatten(FeatureExtractor):
    """
    Extra a single feature, corresponding to the flattened input.
    """

    def __init__(self, input_shape):
        self.input_shape = input_shape
        super().__init__(input_shape)

    def f_map(self, x):
        return {"global": nn.Flatten(start_dim=1)(x)[:, None, :]}
    
    def name(self):
        return "global"



# @persistent_class
class Coordinate(FeatureExtractor):
    """
    Extra a single feature, corresponding to the coordinates of the input.
    """

    def __init__(self, input_shape):
        super().__init__(input_shape)

    def f_map(self, x):
        return {"local": nn.Flatten(start_dim=1)(x)[:, :, None]}
    
    def name(self):
        return "local"

class UnfoldFeatures(FeatureExtractor):
    def __init__(self, input_shape, patch_size, ds_steps=1, use_mean_std=True, ds_first=False, ds_with_std=True, concat_res=False):
        '''
        input_shape: (C, H, W)
        patch_size: int
        ds_steps: int; will have (ds_steps + 1) feature maps; each one is based on the previous one
        '''
        # n_layers: shrink
        super().__init__(input_shape)
        self.patch_size = patch_size
        self.ds_steps = ds_steps
        self.use_mean_std = use_mean_std
        self.ds_first = ds_first
        self.ds_with_std = ds_with_std
        self.concat_res = concat_res

    def f_map(self, x):
        result_dict = dict()

        def push_tensor(name, tensor):
            nonlocal result_dict
            result_dict[name] = rearrange(tensor, "b c h w -> b (h w) c")
            if self.use_mean_std:
                result_dict[f"{name}_mean"] = result_dict[name].mean(dim=1, keepdim=True)
                result_dict[f"{name}_std"] = result_dict[name].std(dim=1, keepdim=True)
        
        def downsample(x): 
            def normalize(x):
                channel_mean = x.mean(dim=(2,3),keepdim=True)
                channel_std = x.std(dim=(2,3),keepdim=True)
                return ((x - channel_mean) / (channel_std + 1e-3)).clamp(-4.0,4.0)
            if x.shape[-1] % 2:
                x = torch.cat([x, x[:, :, :, -1:]], dim=3)
            if x.shape[-2] % 2:
                x = torch.cat([x, x[:, :, -1:, :]], dim=2)
            reshaped = rearrange(x, "b c (h t) (w r) -> b (t r) c h w", t = 2, r = 2)
            reshaped_std = reshaped.std(dim=1)
            reshaped_mean = reshaped.mean(dim=1)
            if self.ds_with_std:
                unfolded = torch.cat([reshaped_mean, reshaped_std], dim=1)
            else:
                unfolded = reshaped_mean
            return normalize(unfolded)
        
        if self.concat_res:
            downsamples = []
            for i in range(self.ds_steps + 1):
                downsamples.append(unfold_feats_merge(x, self.patch_size))
                if i != self.ds_steps:
                    x = downsample(x)
            # reverse
            downsamples = downsamples[::-1]
            cur = None # (B, C, H, W)
            for (i, x) in enumerate(downsamples):
                if cur is None:
                    cur = x
                else:
                    # upsample C
                    cur = F.interpolate(cur, size=x.shape[-2:], mode='bilinear', align_corners=False)
                    cur = torch.cat([cur, x], dim=1)
                push_tensor(f"concat_last{i}", cur)
            return result_dict

        if self.ds_first:
            # first ds each layer, then unfold
            for i in range(self.ds_steps + 1):
                push_tensor(f"ds{i}", unfold_feats_merge(x, self.patch_size))
                if i == self.ds_steps:
                    break
                x = downsample(x)
        else:
            unfolded = unfold_feats_merge(x, self.patch_size)
            for i in range(self.ds_steps + 1):
                push_tensor(f"ds{i}", unfolded)
                if i == self.ds_steps:
                    break
                unfolded = downsample(unfolded)

        return result_dict

    def name(self):
        return f"unfold_{self.patch_size}_ds{self.ds_steps}"


class PerceptualFeatures(FeatureExtractor):
    def __init__(
        self,
        input_shape,
        model_type="clip",
        extract_kwargs=dict(),
        random_init=False,
        compile_mode="none",
        **model_kwargs,
    ):
        """
        Args:
            input_shape: the shape of the input; (C, H, W)
            model_type: the type of the model; 'clip', 'vgg', 'latent_resnet'
            extract_kwargs: additional kwargs for the get_activations function
        """
        super().__init__(input_shape)
        self.model_type = model_type
        self.extract_kwargs = copy.deepcopy(extract_kwargs)
        extraction_keys = ['n_acti', 'min_res', 'use_std', 'use_mean', 'start_acti']
        for key in extraction_keys:
            if key in model_kwargs:
                if key not in self.extract_kwargs:
                    self.extract_kwargs[key] = model_kwargs.pop(key)
                else:
                    model_kwargs.pop(key) # remove from model_kwargs to avoid passing to model

        if model_type == "clip":
            self.model = EvalWrapper(RN50_clip(random_init=random_init, **model_kwargs))
        elif model_type == "vgg":
            self.model = EvalWrapper(
                vgg16_perceptual(random_init=random_init, **model_kwargs)
            )
        elif model_type == "latent_resnet":
            self.model = EvalWrapper(build_latent_resnet(**model_kwargs))
        elif model_type == "mae_resnet":
            self.model = EvalWrapper(build_mae_resnet(**model_kwargs))
        elif model_type == "mae_resnet_gn":
            print("!!", "USE MAE GN")
            self.model = EvalWrapper(build_mae_resnet_gn(**model_kwargs))
        elif model_type == "mae_convnext":
            self.model = EvalWrapper(build_mae_convnext(**model_kwargs))
        elif model_type == "mae_vit":
            self.model = EvalWrapper(build_mae_vit(**model_kwargs))
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        print(
            f"model_type: {model_type}, random_init: {random_init}, extract_kwargs: {self.extract_kwargs}"
        )

        fixed_kwargs = dict(self.extract_kwargs)
        self.acti_func = partial(self.model.model.activations, **fixed_kwargs)
        if compile_mode != "none":
            self.acti_func = torch.compile(self.acti_func, mode=compile_mode)

    def f_map(self, x):
        return self.acti_func(x)
    
    def name(self):
        return self.model_type



def build_feature_modules(
    input_shape,
    loss_on_patches=[3],
    loss_downsamples=[1],
    resnet_load_dicts=[],
    perceptual_load_dicts=[dict(n_acti=3, min_res=16)],
    unfold_feature_dicts=[],
    has_local=True,
    has_global=True,
    compile_mode="none",
    **kwargs
):
    """
    Returns a module list of FeatureExtractor objects.
    Args:
        input_shape: the shape of the input; (C, H, W)
        loss_on_patches: the patches to loss on;
        loss_downsamples: the downsamples to loss on;
        resnet_load_dicts: the load dicts for the resnet models;
        perceptual_load_dicts: the load dicts for the clip/vgg models;
    Returns:
        features: a list of FeatureExtractor objects.
        Will always contain local + global; contain groups + patch losses + downsample losses.
    """
    print("Additional kwargs:", kwargs)

    chn_divisors = []
    for v in range(1, input_shape[0] + 1):
        if input_shape[0] % v == 0:
            chn_divisors.append(v)
    feature_extractors = []
    if has_global:
        feature_extractors.append(Flatten(input_shape))
    if has_local:
        feature_extractors.append(Coordinate(input_shape))

    for perceptual_load_dict in perceptual_load_dicts:
        perceptual_f = PerceptualFeatures(input_shape, compile_mode=compile_mode, **perceptual_load_dict)
        feature_extractors.append(perceptual_f)
    for unfold_feature_dict in unfold_feature_dicts:
        unfold_f = UnfoldFeatures(input_shape, **unfold_feature_dict)
        feature_extractors.append(unfold_f)

    return nn.ModuleList(feature_extractors)


if __name__ == "__main__":
    from utils.profile import print_module_summary
    import time

    device = "cuda"
    # clipf = PerceptualFeatures(input_shape=(4, 32, 32), n_acti=30, min_res=32, use_std=False, use_mean=False, random_init=True, model_type='clip', extract_kwargs=dict(patch_group_size=2)).to(device)
    maef = PerceptualFeatures(
        input_shape=(4, 32, 32), 
        model_type='mae_resnet_gn',
        load_dict=dict(
            run_id="resnetgn50_XL_nocls_20251021_054520",
            epoch="199999",
            load_entry="ema_model",
        ), 
        num_classes=1000,
        in_channels=4,
        base_channels=320,
        layers=[3, 8, 12, 5],
        dropout_prob=0.0,
        patch_size=2,
        extract_kwargs=dict(
            patch_group_size=2,
            patch_mean_size=[2,4],
            patch_std_size=[2,4],
            transpose_goal_dims=[],
            min_res=32,
            use_mean=True,
            use_std=True,
            use_layers=[2,3,4],
            every_k_block=2,
        ),
        compile_mode="default").to(device)
    attn_dict = dict(
        kernel_type="attn_new",
        R_list=[0.2],
        transpose_aff=True,
        no_ratio=False,
    )
    import contextlib
    use_bf16 = True
    context = torch.cuda.amp.autocast(dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()
    from energy_loss import attn_loss_new
    # for i in range(2):
    #     before_time = time.time()
    #     B = 8
    #     target_shape = (32,640)
    #     recon_shape = (16,640)
    #     target = torch.randn(B, *target_shape, device=device)
    #     recon = torch.randn(B, *recon_shape, device=device)
    #     fixed_neg = torch.randn(B, *recon_shape, device=device)
    #     loss, info = attn_loss_new(recon, target, fixed_neg, target_ratio=4.0, n_sinkhorn_steps=2)
    #     print(loss, f"takes {time.time() - before_time} secs")
    

    # for i in range(2):
    #     before_time = time.time()
    #     B = 128
    #     target_shape = (32,640)
    #     recon_shape = (16,640)
    #     target = torch.randn(B, *target_shape, device=device)
    #     recon = torch.randn(B, *recon_shape, device=device)
    #     fixed_neg = torch.randn(B, *recon_shape, device=device)
    #     loss, info = attn_loss_new(recon, target, fixed_neg, target_ratio=4.0, n_sinkhorn_steps=2)
    #     print(loss, f"takes {time.time() - before_time} secs")

    with open(f"module_summary.txt", "w") as f:
        forward_dict = dict(target=torch.randn(8, 32, 4, 32, 32, device=device),
                        recon=torch.randn(8, 16, 4, 32, 32, device=device),
                        fixed_neg=torch.randn(8, 16, 4, 32, 32, device=device),
                        contra_dict=attn_dict,
                        debug=False)
        # for i in range(2):
        #     with contextlib.redirect_stdout(f):
        #         table = print_module_summary(
        #             maef, [], forward_dict
        #         )
        with context:
            for j in range(2):
                before_time = time.time()
                outputs = maef(**forward_dict)
                after_time = time.time()
                print(f"Time taken: {after_time - before_time} seconds")
                # print(outputs)
# python -m features
# test bf16 speed & projection speed
# 1st call: 10.616055965423584 seconds (without feature compilation; )
# with partial compilation: 65.07265019416809 seconds
# with full compilation:  32.687318086624146 seconds