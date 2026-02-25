import os
import argparse
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers import AutoencoderKL
import torchvision.models as tv_models

import torch.utils.checkpoint as cp

from config import load_config
from utils.misc import add_weight_decay


class CombinedMuonAdamW:

    def __init__(self, model, muon_lr=1.5e-3, adamw_lr=1.5e-4,
                 weight_decay=0.01, momentum=0.95, betas=(0.9, 0.95)):
        muon_params = []
        adamw_params_decay = []
        adamw_params_nodecay = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 2:
                muon_params.append(p)
            elif p.ndim > 2:
                adamw_params_decay.append(p)
            else:
                adamw_params_nodecay.append(p)

        self.muon = torch.optim.Muon(
            muon_params, lr=muon_lr, momentum=momentum, weight_decay=weight_decay,
        )
        adamw_groups = []
        if adamw_params_decay:
            adamw_groups.append({"params": adamw_params_decay, "weight_decay": weight_decay})
        if adamw_params_nodecay:
            adamw_groups.append({"params": adamw_params_nodecay, "weight_decay": 0.0})
        self.adamw = torch.optim.AdamW(
            adamw_groups,
            lr=adamw_lr, betas=betas,
        )
        self.muon_base_lr = muon_lr
        self.adamw_base_lr = adamw_lr

        n_muon = sum(p.numel() for p in muon_params)
        n_adamw = sum(p.numel() for p in adamw_params_decay) + sum(p.numel() for p in adamw_params_nodecay)
        print(f"CombinedMuonAdamW: Muon {n_muon/1e6:.1f}M params (lr={muon_lr}), "
              f"AdamW {n_adamw/1e6:.1f}M params (lr={adamw_lr})")

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adamw.step()

    def set_lr(self, warmup_ratio):
        for pg in self.muon.param_groups:
            pg['lr'] = self.muon_base_lr * warmup_ratio
        for pg in self.adamw.param_groups:
            pg['lr'] = self.adamw_base_lr * warmup_ratio

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, d):
        self.muon.load_state_dict(d["muon"])
        self.adamw.load_state_dict(d["adamw"])
from dataset.cache_dataset import CachedFolder
from model.LightningDiT.lightningdit import DitGen
from features import FeatureExtractor
from utils.ema import EMA
from utils.ckpt_utils import save_ckpt, load_last_ckpt, get_run_name, ckpt_epoch_numbers
from utils.fid import eval_fid, visualize_imagenet_samples
from utils.logging_utils import WandbLogger
from utils.distributed_utils import is_main_process, get_rank, get_world_size


def build_moco_v2_backbone(checkpoint_path="", device="cpu"):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        backbone = tv_models.resnet50(weights=None)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("module.encoder_q."):
                new_k = k.replace("module.encoder_q.", "")
                if not new_k.startswith("fc."):
                    cleaned[new_k] = v
        missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
        print(f"MoCo v2 backbone loaded from {checkpoint_path} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print("No MoCo v2 checkpoint found — using ImageNet-supervised ResNet-50")
        backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)

    backbone.fc = nn.Identity()
    backbone.eval()
    backbone.requires_grad_(False)
    return backbone.to(device)


class MoCoV2MultiScaleFeatures(FeatureExtractor):

    STAGES = [
        ("layer1", [1, 2]),
        ("layer2", [1, 3]),
        ("layer3", [1, 3, 5]),
        ("layer4", [1, 2]),
    ]

    def __init__(self, input_shape, vae, moco_backbone, micro_batch=64):
        super().__init__(input_shape)
        self.vae = vae
        self.cache_scaling = 0.18125
        self.micro_batch = micro_batch

        if hasattr(moco_backbone, '_orig_mod'):
            self.moco = moco_backbone._orig_mod
        else:
            self.moco = moco_backbone

        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        self._feature_names = self._build_feature_names()

    def _build_feature_names(self):
        names = []
        for layer_name, extract_blocks in self.STAGES:
            for bidx in extract_blocks:
                prefix = f"{layer_name}_b{bidx}"
                names.extend([
                    f"{prefix}_loc", f"{prefix}_global",
                    f"{prefix}_p2", f"{prefix}_p4",
                    f"{prefix}_norm",
                ])
        names.append("input_x2mean")
        return names

    @staticmethod
    def _patch_stats(feat, patch_size):
        B, C, H, W = feat.shape
        p = patch_size
        Hp, Wp = H // p, W // p
        f = feat[:, :, :Hp * p, :Wp * p]
        f = f.reshape(B, C, Hp, p, Wp, p)
        f = f.permute(0, 2, 4, 1, 3, 5).reshape(B, Hp * Wp, C, p * p)
        means = f.mean(-1)
        stds = f.std(-1)
        return torch.cat([means, stds], dim=1)

    def _feature_vectors(self, feat, prefix):
        B, C, H, W = feat.shape
        out = {}

        out[f"{prefix}_loc"] = feat.reshape(B, C, H * W).permute(0, 2, 1)

        gmean = feat.mean(dim=(2, 3))
        gstd = feat.std(dim=(2, 3))
        out[f"{prefix}_global"] = torch.stack([gmean, gstd], dim=1)

        if H >= 2 and W >= 2:
            out[f"{prefix}_p2"] = self._patch_stats(feat, 2)
        else:
            out[f"{prefix}_p2"] = torch.stack([gmean, gstd], dim=1)

        if H >= 4 and W >= 4:
            out[f"{prefix}_p4"] = self._patch_stats(feat, 4)
        else:
            out[f"{prefix}_p4"] = torch.stack([gmean, gstd], dim=1)

        channel_rms = (feat ** 2).mean(dim=(2, 3)).sqrt()
        out[f"{prefix}_norm"] = channel_rms.unsqueeze(1)

        return out

    def _f_map_chunk(self, x):
        images = self.vae.decode(x / self.cache_scaling).sample
        images = ((images + 1) / 2).clamp(0, 1)
        images_normed = (images - self.img_mean) / self.img_std

        result = {}

        moco = self.moco
        h = moco.conv1(images_normed)
        h = moco.bn1(h)
        h = moco.relu(h)
        h = moco.maxpool(h)

        for layer_name, extract_blocks in self.STAGES:
            layer = getattr(moco, layer_name)
            for bidx, block in enumerate(layer):
                h = block(h)
                if bidx in extract_blocks:
                    prefix = f"{layer_name}_b{bidx}"
                    result.update(self._feature_vectors(h, prefix))

        result["input_x2mean"] = (images ** 2).mean(dim=(2, 3)).unsqueeze(1)

        return tuple(result[name] for name in self._feature_names)

    def f_map(self, x):
        B = x.shape[0]
        mb = self.micro_batch
        all_chunks = []

        for i in range(0, B, mb):
            chunk = x[i:i + mb]
            if torch.is_grad_enabled() and chunk.requires_grad:
                feat_tuple = cp.checkpoint(
                    self._f_map_chunk, chunk, use_reentrant=False
                )
            else:
                feat_tuple = self._f_map_chunk(chunk)
            all_chunks.append(feat_tuple)

        result = {}
        for idx, name in enumerate(self._feature_names):
            result[name] = torch.cat([c[idx] for c in all_chunks], dim=0)
        return result

    def name(self):
        return "moco_v2"


class ClassMemoryBank:

    def __init__(self, num_classes, bank_size, latent_shape, device):
        self.num_classes = num_classes
        self.bank_size = bank_size
        self.device = device
        self.storage = torch.zeros(
            num_classes, bank_size, *latent_shape, device=device
        )
        self.ptr = torch.zeros(num_classes, dtype=torch.long)
        self.count = torch.zeros(num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, latents, labels):
        for i in range(latents.shape[0]):
            c = labels[i].item()
            idx = self.ptr[c].item() % self.bank_size
            self.storage[c, idx].copy_(latents[i])
            self.ptr[c] += 1
            if self.count[c] < self.bank_size:
                self.count[c] += 1

    @torch.no_grad()
    def sample(self, class_label, n_samples):
        n_valid = self.count[class_label].item()
        if n_valid == 0:
            return None
        indices = torch.randint(0, n_valid, (n_samples,))
        return self.storage[class_label, indices]

    def n_valid(self, class_label):
        return self.count[class_label].item()


def get_lr(step, base_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


class ClassLabelDataset(Dataset):

    def __init__(self, num_classes=1000, samples_per_class=50):
        self.labels = []
        for c in range(num_classes):
            self.labels.extend([c] * samples_per_class)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.labels[idx], dtype=torch.long)


def build_vae_decoder(device, for_training=False):
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()
    vae.requires_grad_(False)  # freeze params; gradients still flow through ops
    return vae


def make_eval_generator(model, vae, cfg_scale, device):
    cache_scaling = 0.18125

    @torch.inference_mode()
    def generator_fn(batch):
        if isinstance(batch, (list, tuple)):
            labels = batch[0].to(device)
        else:
            labels = batch.to(device)
        output = model(labels, cfg_scale=cfg_scale)
        latents = output["samples"]
        images = vae.decode(latents / cache_scaling).sample
        images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=-1.0)
        images = ((images + 1) / 2).clamp(0, 1)
        return images

    return generator_fn


def main(args):
    cfg = load_config(args.config)

    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        is_main = (rank == 0)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rank = 0
        world_size = 1
        is_main = True

    dataset_name = cfg.dataset.get("name", "imagenet256_cache")
    run_name = args.run_name or cfg.wandb.get("name", None) or "dit_B2_clip"
    run_id = get_run_name(run_name)
    if is_main:
        print(f"Run ID: {run_id}")

    cached_path = args.cached_path
    dataset = CachedFolder(root=cached_path)
    if is_main:
        print(f"Loaded cached latent dataset: {len(dataset)} samples")

    batch_size = args.batch_size or cfg.train.get("total_batch_size", 64)

    if world_size > 1:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    import copy
    dec_cfg = copy.deepcopy(dict(cfg.model.decoder_config))
    generator = DitGen(**dec_cfg).to(device)

    n_param_tensors = sum(1 for _ in generator.parameters())
    print(f"[Rank {rank}] generator has {n_param_tensors} parameter tensors")

    compile_model = cfg.train.get("compile_model", False)
    compile_mode = cfg.train.get("compile_mode", "default")
    if compile_model:
        generator = torch.compile(generator, mode=compile_mode)
        if is_main:
            print(f"Generator compiled with torch.compile (mode={compile_mode})")

    if world_size > 1:
        dist.barrier()
        generator = DDP(generator, device_ids=[local_rank])

    if is_main:
        n_params = sum(p.numel() for p in generator.parameters()) / 1e6
        print(f"Generator parameters: {n_params:.1f}M")

    logger = WandbLogger()
    wandb_cfg = copy.deepcopy(dict(cfg))
    logger.setup_wandb(
        project=cfg.wandb.get("project", "drift-flow"),
        entity=cfg.wandb.get("entity", None) or None,
        name=run_id,
        config=wandb_cfg,
    )

    ema_decay = cfg.train.get("ema_decay", 0.999)
    gen_model_raw = generator.module if world_size > 1 else generator
    gen_model_uncompiled = (
        gen_model_raw._orig_mod
        if hasattr(gen_model_raw, '_orig_mod')
        else gen_model_raw
    )
    ema = EMA(gen_model_uncompiled, decay=ema_decay)
    if is_main:
        print(f"EMA (eval only): decay={ema_decay}")

    train_vae = build_vae_decoder(device, for_training=True)
    if is_main:
        print(f"Loaded SD-VAE decoder for MoCo v2 feature pipeline (frozen, grad-through)")

    moco_cfg = cfg.model.get("moco_v2", {})
    moco_checkpoint = moco_cfg.get("checkpoint_path", "")
    moco_backbone = build_moco_v2_backbone(moco_checkpoint, device)
    if is_main:
        moco_params = sum(p.numel() for p in moco_backbone.parameters()) / 1e6
        print(f"MoCo v2 backbone: {moco_params:.1f}M params (frozen, multi-scale)")

    input_shape = tuple(cfg.model.get("input_shape", [4, 32, 32]))
    feat_mb = cfg.train.get("feat_micro_batch", 64)
    moco_feature_extractor = MoCoV2MultiScaleFeatures(
        input_shape, vae=train_vae, moco_backbone=moco_backbone, micro_batch=feat_mb
    ).to(device)
    feature_extractors = [moco_feature_extractor]
    if is_main:
        print(f"Feature extractors: {[f.name() for f in feature_extractors]}")

    opt_cfg = cfg.get("optimizer", {})
    muon_lr = opt_cfg.get("muon_lr", 1.5e-3)
    adamw_lr = opt_cfg.get("adamw_lr", 2e-4)
    weight_decay = opt_cfg.get("weight_decay", 0.01)
    momentum = opt_cfg.get("momentum", 0.95)
    optimizer = CombinedMuonAdamW(
        gen_model_raw,
        muon_lr=muon_lr,
        adamw_lr=adamw_lr,
        weight_decay=weight_decay,
        momentum=momentum,
        betas=(opt_cfg.get("beta1", 0.9), opt_cfg.get("beta2", 0.95)),
    )

    start_step = 0
    start_epoch = 0
    load_dict = cfg.train.get("load_dict", {})
    resume_run_id = args.resume_run_id or load_dict.get("run_id", "")
    if resume_run_id:
        if is_main:
            print(f"Resuming from run: {resume_run_id}")
        ckpt = load_last_ckpt(resume_run_id)
        gen_model_raw.load_state_dict(ckpt["generator"])
        ema.model.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0)
        if is_main:
            print(f"Resumed from step {start_step}, epoch {start_epoch}")

    vae = train_vae

    num_classes = cfg.train.get("n_classes", 1000)
    eval_dataset = ClassLabelDataset(num_classes=num_classes, samples_per_class=50)

    train_cfg = cfg.train
    fwd = train_cfg.get("forward_dict", {})
    attn_cfg = fwd.get("attn_dict", {})

    Nc = train_cfg.get("n_class_labels", 8)
    N_neg = fwd.get("recon", 8)
    cfg_scale = train_cfg.get("min_cfg_scale", 1.0)
    warmup_steps = train_cfg.lr_schedule.get("warmup_steps", 5000)
    total_steps = train_cfg.get("n_steps", 30000)
    clip_grad = train_cfg.lr_schedule.get("clip_grad", 2.0)
    save_every = train_cfg.get("save_per_step", 5000)
    eval_every = train_cfg.get("eval_gen_per_step", 5000)
    eval_fid_samples = train_cfg.get("eval_fid_samples", 50000)
    eval_batch_size = train_cfg.get("eval_bsz_per_gpu", 64)

    N_pos = train_cfg.get("n_pos", 1)
    pos_bank_size = train_cfg.get("pos_bank_size", 128)

    use_bf16 = train_cfg.get("use_bf16", False)

    contra_dict = dict(
        kernel_type=attn_cfg.get("kernel_type", "attn_new"),
        sample_norm=attn_cfg.get("sample_norm", True),
        scale_dist_normed=attn_cfg.get("scale_dist_normed", True),
        R_list=attn_cfg.get("R_list", [0.02, 0.05, 0.2]),
    )

    mem_bank = ClassMemoryBank(
        num_classes=num_classes,
        bank_size=pos_bank_size,
        latent_shape=input_shape,
        device=device,
    )
    if is_main:
        bank_mem_mb = (num_classes * pos_bank_size * torch.zeros(input_shape).numel() * 4) / 1e6
        print(f"Memory bank: {num_classes} classes × {pos_bank_size} slots "
              f"(~{bank_mem_mb:.0f} MB, N_pos={N_pos})")

    global_step = start_step
    n_dataset = len(dataset)
    steps_per_epoch = n_dataset // (batch_size * world_size)

    if is_main:
        print(f"\n{'='*60}")
        print(f"Pipeline summary:")
        print(f"  Config file:          {args.config}")
        print(f"  Run ID:               {run_id}")
        print(f"  Cached latent path:   {cached_path}")
        print(f"  Latent shape:         {list(input_shape)}")
        print(f"  Dataset size:         {n_dataset}")
        print(f"  Batch size (per GPU): {batch_size}")
        print(f"  Steps per epoch:      {steps_per_epoch}")
        print(f"  Total training steps: {total_steps}")
        print(f"  Start step:           {start_step}")
        print(f"  Warmup steps:         {warmup_steps}")
        print(f"  Save every:           {save_every} steps")
        print(f"  Eval FID every:       {eval_every} steps ({eval_fid_samples} samples)")
        print(f"  Nc={Nc}, N_pos={N_pos}, N_neg={N_neg}")
        print(f"  Samples per GPU:      {Nc} × {N_neg} = {Nc * N_neg} generated")
        print(f"  Optimizer:            Muon (lr={muon_lr}) + AdamW (lr={adamw_lr})")
        print(f"  Mixed precision:      {'bf16' if use_bf16 else 'fp32'}")
        print(f"  Feat micro-batch:     {feat_mb} (VAE+MoCo sub-batch, checkpointed)")
        print(f"  CFG alpha:            {cfg_scale} (1.0 = no CFG)")
        print(f"  EMA decay (eval):     {ema_decay}")
        print(f"  Grad clip:            {clip_grad}")
        n_feat = len(moco_feature_extractor._feature_names)
        print(f"  Feature pipeline:     latents → VAE decode → MoCo ResNet-50 (multi-scale, A.5)")
        print(f"  Feature loss terms:   {n_feat} (9 maps × 5 types + 1 input)")
        print(f"  MoCo checkpoint:      {moco_checkpoint or '(ImageNet-supervised R50)'}")
        print(f"  Loss config:          {contra_dict}")
        print(f"{'='*60}\n")

    gen_model = generator.module if world_size > 1 else generator
    epoch = start_epoch if resume_run_id else 0

    while global_step < total_steps:
        if world_size > 1:
            sampler.set_epoch(epoch)

        for batch_idx, (latents, labels) in enumerate(dataloader):
            if global_step >= total_steps:
                break

            # ── Learning rate warmup ──
            warmup_ratio = min(1.0, (global_step + 1) / warmup_steps) if warmup_steps > 0 else 1.0
            optimizer.set_lr(warmup_ratio)

            latents = latents.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            mem_bank.update(latents, labels)

            if world_size > 1:
                unique_labels = labels.unique()
                selected_buf = torch.zeros(Nc, dtype=torch.long, device=device)
                n_available = torch.tensor(0, dtype=torch.long, device=device)
                if rank == 0:
                    if len(unique_labels) < Nc:
                        actual = unique_labels
                    else:
                        perm = torch.randperm(len(unique_labels), device=device)[:Nc]
                        actual = unique_labels[perm]
                    n_available.fill_(len(actual))
                    selected_buf[:len(actual)] = actual
                dist.broadcast(selected_buf, src=0)
                dist.broadcast(n_available, src=0)
                n_avail = n_available.item()
                all_selected = selected_buf[:n_avail]

                local_Nc = math.ceil(n_avail / world_size)
                start = rank * local_Nc
                end = min(start + local_Nc, n_avail)
                my_classes = all_selected[start:end]
            else:
                unique_labels = labels.unique()
                if len(unique_labels) < Nc:
                    my_classes = unique_labels
                else:
                    perm = torch.randperm(len(unique_labels), device=device)[:Nc]
                    my_classes = unique_labels[perm]

            optimizer.zero_grad()
            accum_loss = 0.0
            accum_info = {}

            pos_list = []
            gen_list = []
            valid_classes = []

            for c in my_classes:
                c_val = c.item()

                pos_samples = mem_bank.sample(c_val, N_pos)
                if pos_samples is None:
                    continue
                pos_list.append(pos_samples)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
                    class_cond = torch.full((N_neg,), c_val, dtype=torch.long, device=device)
                    gen_output = gen_model(class_cond, cfg_scale=cfg_scale)
                    generated_latents = gen_output["samples"]
                gen_list.append(generated_latents)

                valid_classes.append(c_val)

            if len(valid_classes) == 0:
                dummy = torch.zeros(1, device=device, requires_grad=True)
                dummy_out = (generator(torch.zeros(1, dtype=torch.long, device=device),
                             cfg_scale=cfg_scale)["samples"] * 0.0).sum() + dummy
                dummy_out.backward()
                optimizer.zero_grad()
                continue

            target_batch = torch.stack(pos_list, dim=0)
            gen_batch = torch.stack(gen_list, dim=0)

            local_loss = torch.zeros(len(valid_classes), device=device)
            local_info = {}

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
                for feat in feature_extractors:
                    loss, info = feat(
                        target=target_batch,
                        recon=gen_batch,
                        contra_dict=contra_dict,
                    )
                    local_loss = local_loss + loss
                    for k, v in info.items():
                        local_info[f"{feat.name()}/{k}"] = v

            avg_loss = local_loss.mean()
            avg_loss.backward()

            accum_loss = avg_loss.item()
            for k, v in local_info.items():
                accum_info[k] = v.item() if isinstance(v, torch.Tensor) else v

            grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=clip_grad)

            if not torch.isfinite(grad_norm):
                if is_main:
                    print(f"[step {global_step}] WARNING: non-finite grad norm ({grad_norm:.4f}), skipping update")
                optimizer.zero_grad()
            else:
                optimizer.step()

            ema.update(gen_model)

            global_step += 1

            log_payload = {
                "train/loss": accum_loss,
                "train/lr": optimizer.param_groups[0]['lr'],
                "train/step": global_step,
            }
            for k, v in accum_info.items():
                log_payload[f"train/{k}"] = v
            logger.log_dict(log_payload, step=global_step)
            logger.set_step(global_step)

            if is_main and global_step % 100 == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                log_str = (f"[epoch {epoch}][step {global_step}/{total_steps}] "
                           f"loss={accum_loss:.6f} lr={cur_lr:.2e}")
                print(log_str)

            if global_step % save_every == 0:
                ckpt_dict = {
                    "generator": gen_model.state_dict(),
                    "ema": ema.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                }
                save_ckpt(run_id, global_step, ckpt_dict, max_ckpts=5)
                if is_main:
                    print(f"Saved checkpoint at step {global_step}")

            if global_step % eval_every == 0:
                if is_main:
                    print(f"\n{'─'*40}")
                    print(f"FID evaluation at step {global_step}...")

                ema_gen_fn = make_eval_generator(ema.model, vae, cfg_scale, device)
                visualize_imagenet_samples(
                    generator=ema_gen_fn,
                    logger=logger,
                    log_prefix=f"step_{global_step}",
                    step=global_step,
                )

                ema_fid_result = eval_fid(
                    generator=ema_gen_fn,
                    cond_dataset=eval_dataset,
                    logger=logger,
                    total_samples=eval_fid_samples,
                    gpu_batch_size=eval_batch_size,
                    log_prefix=f"EMA_{global_step}",
                    dataset=dataset_name,
                    log_folder_name="eval",
                    step=global_step,
                )
                if is_main:
                    fid_val = ema_fid_result.get("fid", float("nan"))
                    isc_val = ema_fid_result.get("isc_mean", float("nan"))
                    prec_val = ema_fid_result.get("precision", 0)
                    recall_val = ema_fid_result.get("recall", 0)
                    logger.log_dict({
                        "eval/ema_fid": fid_val,
                        "eval/ema_isc": isc_val,
                        "eval/ema_precision": prec_val,
                        "eval/ema_recall": recall_val,
                    }, step=global_step)
                    print(f"EMA FID @ step {global_step}: {fid_val:.2f}  "
                          f"IS: {isc_val:.2f}  P: {prec_val:.3f}  R: {recall_val:.3f}")
                    print(f"{'─'*40}\n")

                if world_size > 1:
                    dist.barrier()

        if is_main:
            print(f"Epoch {epoch} complete, global_step={global_step}/{total_steps}")
        epoch += 1

    # ── Final checkpoint ──
    if global_step > 0:
        ckpt_dict = {
            "generator": gen_model.state_dict(),
            "ema": ema.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
        }
        save_ckpt(run_id, global_step, ckpt_dict, max_ckpts=5)
        if is_main:
            print(f"Saved final checkpoint at step {global_step}")

    if is_main:
        print(f"\nFinal FID evaluation at step {global_step}...")
    ema_gen_fn = make_eval_generator(ema.model, vae, cfg_scale, device)
    visualize_imagenet_samples(
        generator=ema_gen_fn,
        logger=logger,
        log_prefix=f"final_{global_step}",
        step=global_step,
    )
    final_fid = eval_fid(
        generator=ema_gen_fn,
        cond_dataset=eval_dataset,
        logger=logger,
        total_samples=eval_fid_samples,
        gpu_batch_size=eval_batch_size,
        log_prefix=f"EMA_final_{global_step}",
        dataset=dataset_name,
        log_folder_name="eval",
        step=global_step,
    )
    if is_main:
        fid_val = final_fid.get("fid", float("nan"))
        isc_val = final_fid.get("isc_mean", float("nan"))
        logger.log_dict({
            "eval/final_ema_fid": fid_val,
            "eval/final_ema_isc": isc_val,
        }, step=global_step)
        print(f"Final EMA FID: {fid_val:.2f}  IS: {isc_val:.2f}")

    logger.finish()


def get_args():
    parser = argparse.ArgumentParser("Drifting Model Training")

    parser.add_argument('--config', type=str, default='configs/dit_B2_clip.yaml',
                        help='Path to YAML config file')

    parser.add_argument('--cached_path', type=str, required=True,
                        help='Path to cached SD-VAE latents (output of cache_latent.py)')
    parser.add_argument('--batch_size', type=int, default=0,
                        help='Batch size per GPU (0 = use config value)')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resume_run_id', type=str, default='',
                        help='Run ID to resume from (overrides config load_dict.run_id)')
    parser.add_argument('--run_name', type=str, default='',
                        help='Run name (overrides config wandb.name)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
