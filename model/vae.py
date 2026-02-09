# %%
import torch
import torch.nn as nn
import copy
from utils.misc import EasyDict

from model.vit_ae import Encoder, Decoder
from model.mlp import CondEmbed
from utils.misc import EasyDict
from features import build_feature_modules
from einops import repeat, rearrange
import random
from model.LightningDiT.lightningdit import DitGen
from model.styleGAN.training.networks_stylegan3 import Generator as StyleGAN

default_clip_dict = dict(extract_kwargs=dict(min_res=64,use_mean=True,use_std=True),random_init=False, model_type='clip')
class VAE(nn.Module):
    def __init__(self, input_shape, decoder_config, loss_on_patches=[1,3,5],clip_dict=default_clip_dict,clip_dict2=dict(), feature_params=dict(laplacian_load_dict=dict(enable=False)), enable_clip=True, compile_mode="none"):
        super().__init__()
        print("Building VAE with decoder_config:", decoder_config)
        print("Compile mode:", compile_mode)

        if "class" not in decoder_config:
            decoder_class = "DitGen"
        else:
            decoder_class = decoder_config.pop("class")
            
        if decoder_class == "DitGen":
            self.decoder = DitGen(compile_mode=compile_mode, **decoder_config)
        elif decoder_class == "StyleGAN":
            self.decoder = StyleGAN(**decoder_config)
        else:
            raise ValueError(f"Invalid decoder model: {decoder_class}")

        c_dict = copy.deepcopy(default_clip_dict)
        c_dict.update(clip_dict)
        print("c_dict:", c_dict)
        percep_dicts = [c_dict]
        if clip_dict2 != dict():
            percep_dicts = [c_dict, clip_dict2]

        if not enable_clip:
            percep_dicts = []
        self.feats = build_feature_modules(input_shape, compile_mode=compile_mode, **EasyDict(loss_on_patches=loss_on_patches, loss_downsamples=[1], perceptual_load_dicts=percep_dicts), **feature_params)

    def forward(self, x, c, recon=8, cfg_scale=1.0, neg_samples=None, neg_w=0.0, attn_dict=dict(kernel_type="attn", sample_norm=True), old_samples=None, noise_dict=None, **extra_kwargs):
        '''
        Args:
            x: [B, F1, *input_shape]
            c: [B,]
            recon: int
            cfg_scale: [B] or float
            neg_samples: [B, F2, *input_shape]
            neg_w: float or [B, F2]
            attn_dict: dict
            **extra_kwargs: dict
            return_samples: bool; if true: will return x_recon [B, F, *input_shape] in info['samples']
            old_samples: [B, r, *input_shape] or None; if provided, use these as the generated samples for loss calculation.
            noise_dict: dict (where each entry: (b, r, ...)) or None; if provided, use this noise for generation.
            return_noise: bool; if true: will return the noise used for generation in info['noise'].

        Returns:
            total_loss: float
            contra_info: dict
        '''
        B = x.shape[0]
        c_rep = repeat(c, 'b ... -> (r b) ...', r=recon)
        if noise_dict is None:
            noise_dict = self.decoder.generate_noise_dict(B * recon, x.device)
        else:
            noise_dict = {k: rearrange(v, 'b r ... -> (r b) ...', r=recon) for k, v in noise_dict.items()}

        if isinstance(cfg_scale, (float, int)):
            cfg_scale_vec = torch.full((B,), float(cfg_scale), device=x.device)
        else:
            cfg_scale_vec = torch.as_tensor(cfg_scale, device=x.device).view(-1)
            if cfg_scale_vec.numel() == 1:
                cfg_scale_vec = cfg_scale_vec.expand(B)
        cfg_scale = repeat(cfg_scale_vec, 'b -> (r b)', r=recon)

        new_gen = self.decoder(c_rep, cfg_scale=cfg_scale, noise_dict=noise_dict)["samples"]
        new_gen = rearrange(new_gen, '(r b) ... -> b r ...', r=recon)
        # old_gen = old_samples if old_samples is not None else new_gen

        contra_loss = 0
        contra_info = dict()
        
        recon_w = torch.ones((B, new_gen.shape[1]), device=x.device)
        target_w = torch.ones((B, x.shape[1]), device=x.device)

        if neg_samples is not None:
            fixed_neg = neg_samples
            fixed_neg_w = neg_w
        else:
            fixed_neg, fixed_neg_w = None, None

        for i, feat in enumerate(self.feats):
            loss, info = feat(target=x, recon=new_gen, old_recon=old_samples, fixed_neg=fixed_neg, target_w=target_w, recon_w=recon_w, fixed_neg_w=fixed_neg_w, contra_dict=attn_dict)
            contra_loss = contra_loss + loss
            name = feat.name() if hasattr(feat, "name") else f"extractor_{i}"
            for k, v in info.items():
                contra_info[f"{name}/{k}"] = v
        contra_loss = contra_loss / len(self.feats)
        total_loss = contra_loss
        total_loss = total_loss.mean()
        return total_loss, contra_info

    def generate(self, c, cfg_scale=1.0, temp=1.0, noise_dict=None):
        '''
        Args:
            c: [B,]
            cfg_scale: [B] or float
            temp: float
            noise_dict: dict (where each entry: (b, ...)) or None; if provided, use this noise for generation.
        Returns:
            dict with entries:
                samples: torch.Tensor: The generated samples.
                noise: dict: The noise dictionary used for generation.
        '''
        return self.decoder(c, cfg_scale=cfg_scale, temp=temp, noise_dict=noise_dict)
