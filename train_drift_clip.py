"""
Drifting Model Training Script for ImageNet 256x256 (Latent Space)
  — MoCo v2 pixel-space feature extractor variant

Training pipeline (per step):
=============================
    Images ──> SD-VAE encoder ──> target_latents (frozen, pre-cached)
    (class + noise) ──> DitGen ──> generated_latents (has gradients)
                                        │
                    ┌───────────────────┘
                    ▼
            SD-VAE decoder (frozen, but NO torch.no_grad on gen path —
            gradients flow through back to DitGen):
              latents → pixel-space images [3, 256, 256]
                    │
                    ▼
            MoCo v2 ResNet-50 backbone (frozen, but NO torch.no_grad
            on gen path — gradients flow through back to DitGen):
              images → 2048-d feature vectors
                    │
                    ▼
            Contrastive drifting loss (energy_loss.py)
              (old_recon defaults to detached current recon)
                    │
                    ▼
            Backprop → update DitGen only

Pre-caching (one-time):
    torchrun --nproc_per_node=NUM_GPUS dataset/cache_latent.py \\
        --data_path /path/to/imagenet --cached_path /path/to/cache --split train
    (then again with --split val)

Training (30k steps, 5k warmup):
    torchrun --nproc_per_node=NUM_GPUS train_drift_clip.py \\
        --config configs/dit_B2_clip.yaml --cached_path /path/to/cache/train
"""

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

from config import load_config
from utils.misc import add_weight_decay
from dataset.cache_dataset import CachedFolder
from model.LightningDiT.lightningdit import DitGen
from features import FeatureExtractor
from utils.ema import EMA
from utils.ckpt_utils import save_ckpt, load_last_ckpt, get_run_name, ckpt_epoch_numbers
from utils.fid import eval_fid, visualize_imagenet_samples
from utils.logging_utils import WandbLogger
from utils.distributed_utils import is_main_process, get_rank, get_world_size


# ─────────────────────────────────────────────────────────
# MoCo v2 backbone + VAE-based pixel-space feature extractor
# ─────────────────────────────────────────────────────────

def build_moco_v2_backbone(checkpoint_path="", device="cpu"):
    """Load a ResNet-50 backbone with MoCo v2 pretrained weights.

    If checkpoint_path is provided, loads the official MoCo v2 checkpoint
    (keys prefixed with 'module.encoder_q.'). Otherwise falls back to
    ImageNet-supervised ResNet-50 weights from torchvision.

    Returns a ResNet-50 with the final FC layer replaced by Identity,
    outputting 2048-d features after global average pooling.
    """
    if checkpoint_path and os.path.isfile(checkpoint_path):
        backbone = tv_models.resnet50(weights=None)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        # Strip MoCo v2 prefix 'module.encoder_q.' and skip the FC head
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

    # Replace final FC with identity so forward() returns 2048-d features
    backbone.fc = nn.Identity()
    backbone.eval()
    backbone.requires_grad_(False)  # freeze params, but gradients still flow through
    return backbone.to(device)


class MoCoV2PixelFeatures(FeatureExtractor):
    """Feature extractor that decodes latents → pixels via SD-VAE,
    then extracts MoCo v2 features.

    Operates in latent space (input_shape = (4, 32, 32)) but internally
    decodes to pixel space and runs through the MoCo v2 ResNet-50 backbone.

    Both the VAE decoder and MoCo backbone are frozen (requires_grad=False),
    but NOT wrapped in torch.no_grad() — gradients flow through them
    from the contrastive loss back to the generator for generated samples.
    For target samples, the parent FeatureExtractor.forward() wraps
    extraction in torch.no_grad() automatically.
    """

    def __init__(self, input_shape, vae, moco_backbone):
        super().__init__(input_shape)
        self.vae = vae
        self.moco = moco_backbone
        self.cache_scaling = 0.18125
        # ImageNet normalization constants (for MoCo v2 input)
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def f_map(self, x):
        """
        x: [B, 4, 32, 32] latents (scaled by 0.18125)
        Returns: dict with 'moco' → [B, 1, 2048]
        """
        # Decode latents → pixel images [-1, 1] → [0, 1]
        images = self.vae.decode(x / self.cache_scaling).sample
        images = ((images + 1) / 2).clamp(0, 1)  # [B, 3, 256, 256]
        # ImageNet normalize for MoCo v2
        images = (images - self.img_mean) / self.img_std
        # Extract MoCo v2 features (2048-d after global avg pool)
        features = self.moco(images)  # [B, 2048]
        return {"moco": features.unsqueeze(1)}  # [B, 1, 2048]

    def name(self):
        return "moco_v2"


# (No memory bank — positives come directly from the data batch)


# ─────────────────────────────────────────────────────────
# Learning rate schedule with linear warmup
# ─────────────────────────────────────────────────────────
def get_lr(step, base_lr, warmup_steps):
    """Linear warmup for warmup_steps, then constant base_lr."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


# ─────────────────────────────────────────────────────────
# Simple dataset of class labels for FID evaluation
# ─────────────────────────────────────────────────────────
class ClassLabelDataset(Dataset):
    """Returns (class_label,) for each of num_classes * samples_per_class entries.
    Used as cond_dataset for eval_fid: the generator receives class labels
    and produces images."""
    def __init__(self, num_classes=1000, samples_per_class=50):
        self.labels = []
        for c in range(num_classes):
            self.labels.extend([c] * samples_per_class)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.labels[idx], dtype=torch.long)


def build_vae_decoder(device, for_training=False):
    """Load SD-VAE decoder.

    Args:
        device: target device.
        for_training: if True, the VAE is frozen but gradients can flow
            through its operations (used in the MoCo v2 feature pipeline).
            If False, fully frozen for inference-only (FID eval).
    """
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()
    vae.requires_grad_(False)  # freeze params; gradients still flow through ops
    return vae


def make_eval_generator(model, vae, cfg_scale, device):
    """Return a function compatible with eval_fid:
        generator(batch) -> images [B, 3, 256, 256] in [0, 1]
    where batch is a tensor of class labels from ClassLabelDataset.
    """
    cache_scaling = 0.18125  # same as CachedFolder

    @torch.inference_mode()
    def generator_fn(batch):
        if isinstance(batch, (list, tuple)):
            labels = batch[0].to(device)
        else:
            labels = batch.to(device)
        # Generate latents with the model
        output = model(labels, cfg_scale=cfg_scale)
        latents = output["samples"]  # [B, 4, 32, 32]
        # Decode through SD-VAE (undo cache scaling)
        images = vae.decode(latents / cache_scaling).sample
        images = ((images + 1) / 2).clamp(0, 1)  # [B, 3, 256, 256]
        return images

    return generator_fn


# ─────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────
def main(args):
    # ── Load config ──
    cfg = load_config(args.config)

    # ── Distributed setup ──
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

    # ── Run ID ──
    dataset_name = cfg.dataset.get("name", "imagenet256_cache")
    run_name = args.run_name or cfg.wandb.get("name", None) or "dit_B2_clip"
    run_id = get_run_name(run_name)
    if is_main:
        print(f"Run ID: {run_id}")

    # ── Dataset: load cached latents ──
    # CachedFolder reads .npz files produced by cache_latent.py
    # Returns (latent * 0.18125, class_label) with random hflip
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

    # ── Generator (from config model.decoder_config) ──
    # DitGen accepts **kwargs and forwards to LightningDiT
    # Extract decoder config BEFORE any wandb/logging that could modify cfg
    import copy
    dec_cfg = copy.deepcopy(dict(cfg.model.decoder_config))
    generator = DitGen(**dec_cfg).to(device)

    # Debug: verify param count consistency across ranks before DDP
    n_param_tensors = sum(1 for _ in generator.parameters())
    print(f"[Rank {rank}] generator has {n_param_tensors} parameter tensors")

    if world_size > 1:
        dist.barrier()  # ensure all ranks constructed the model
        generator = DDP(generator, device_ids=[local_rank])

    if is_main:
        n_params = sum(p.numel() for p in generator.parameters()) / 1e6
        print(f"Generator parameters: {n_params:.1f}M")

    # ── Wandb (initialized AFTER model + DDP to avoid config mutation) ──
    logger = WandbLogger()
    wandb_cfg = copy.deepcopy(dict(cfg))  # deep copy so wandb can't modify original
    logger.setup_wandb(
        project=cfg.wandb.get("project", "drift-flow"),
        entity=cfg.wandb.get("entity", None) or None,
        name=run_id,
        config=wandb_cfg,
    )

    # ── EMA model (for evaluation / FID only) ──
    ema_decay = cfg.train.get("ema_decay", 0.999)
    gen_model_raw = generator.module if world_size > 1 else generator
    ema = EMA(gen_model_raw, decay=ema_decay)
    if is_main:
        print(f"EMA (eval only): decay={ema_decay}")

    # ── SD-VAE decoder (frozen, but gradients flow through for gen path) ──
    # Loaded eagerly because the MoCo v2 feature pipeline needs it every step
    train_vae = build_vae_decoder(device, for_training=True)
    if is_main:
        print("Loaded SD-VAE decoder for MoCo v2 feature pipeline (frozen, grad-through)")

    # ── MoCo v2 backbone (frozen, but gradients flow through for gen path) ──
    moco_cfg = cfg.model.get("moco_v2", {})
    moco_checkpoint = moco_cfg.get("checkpoint_path", "")
    moco_backbone = build_moco_v2_backbone(moco_checkpoint, device)
    if is_main:
        moco_params = sum(p.numel() for p in moco_backbone.parameters()) / 1e6
        print(f"MoCo v2 backbone: {moco_params:.1f}M params (frozen)")

    # ── Feature extractor: MoCo v2 pixel-space features ──
    # Pipeline: latents → VAE decode → pixel images → MoCo v2 → 2048-d features
    input_shape = tuple(cfg.model.get("input_shape", [4, 32, 32]))
    moco_feature_extractor = MoCoV2PixelFeatures(
        input_shape, vae=train_vae, moco_backbone=moco_backbone
    ).to(device)
    feature_extractors = [moco_feature_extractor]
    if is_main:
        print(f"Feature extractors: {[f.name() for f in feature_extractors]}")

    # ── Optimizer (from config optimizer section) ──
    # Use add_weight_decay to exclude biases and norm layers from weight decay,
    # matching the convention in config.py / build_model_dict.
    opt_cfg = cfg.get("optimizer", {})
    base_lr = opt_cfg.get("lr", 2e-4)
    weight_decay = opt_cfg.get("weight_decay", 0.01)
    param_groups = add_weight_decay(gen_model_raw, weight_decay=weight_decay, lr=base_lr)
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(opt_cfg.get("beta1", 0.9), opt_cfg.get("beta2", 0.95)),
    )

    # ── Checkpoint resume ──
    start_step = 0
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
        if is_main:
            print(f"Resumed from step {start_step}")

    # ── VAE decoder for FID evaluation (reuse the training VAE) ──
    vae = train_vae  # same frozen VAE used in the MoCo feature pipeline

    # ── Eval dataset for FID ──
    num_classes = cfg.train.get("n_classes", 1000)
    eval_dataset = ClassLabelDataset(num_classes=num_classes, samples_per_class=50)

    # ── Training config (from config train section) ──
    train_cfg = cfg.train
    fwd = train_cfg.get("forward_dict", {})
    attn_cfg = fwd.get("attn_dict", {})

    Nc = train_cfg.get("n_class_labels", 8)
    N_neg = fwd.get("recon", 8)
    cfg_scale = train_cfg.get("min_cfg_scale", 1.0)  # alpha=1.0 means no CFG
    warmup_steps = train_cfg.lr_schedule.get("warmup_steps", 5000)
    total_steps = train_cfg.get("n_steps", 30000)
    clip_grad = train_cfg.lr_schedule.get("clip_grad", 2.0)
    save_every = train_cfg.get("save_per_step", 5000)
    eval_every = train_cfg.get("eval_gen_per_step", 5000)
    eval_fid_samples = train_cfg.get("eval_fid_samples", 50000)
    eval_batch_size = train_cfg.get("eval_bsz_per_gpu", 64)

    # Loss config (from config train.forward_dict.attn_dict)
    contra_dict = dict(
        kernel_type=attn_cfg.get("kernel_type", "attn_new"),
        sample_norm=attn_cfg.get("sample_norm", True),
        scale_dist_normed=attn_cfg.get("scale_dist_normed", True),
        R_list=attn_cfg.get("R_list", [0.02, 0.05, 0.2]),
    )

    # ── Training loop (step-based, not epoch-based) ──
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
        print(f"  Nc={Nc}, N_neg={N_neg} (positives from batch)")
        print(f"  CFG alpha:            {cfg_scale} (1.0 = no CFG)")
        print(f"  EMA decay (eval):     {ema_decay}")
        print(f"  Grad clip:            {clip_grad}")
        print(f"  Feature pipeline:     latents → VAE decode → MoCo v2 (2048-d)")
        print(f"  MoCo checkpoint:      {moco_checkpoint or '(ImageNet-supervised R50)'}")
        print(f"  Loss config:          {contra_dict}")
        print(f"{'='*60}\n")

    gen_model = generator.module if world_size > 1 else generator
    epoch = 0

    while global_step < total_steps:
        if world_size > 1:
            sampler.set_epoch(epoch)

        for batch_idx, (latents, labels) in enumerate(dataloader):
            if global_step >= total_steps:
                break

            # ── Learning rate warmup ──
            lr = get_lr(global_step, base_lr, warmup_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # ════════════════════════════════════════════════
            # Step A: Images -> SD-VAE encoder -> target_latents
            #   (pre-cached; CachedFolder returns latent * 0.18125)
            # ════════════════════════════════════════════════
            latents = latents.to(device, non_blocking=True)  # [B, 4, 32, 32]
            labels = labels.to(device, non_blocking=True)    # [B]

            # ── Group batch by class, pick up to Nc classes ──
            # No memory bank: positives come directly from the data batch.
            unique_labels = labels.unique()
            if len(unique_labels) < Nc:
                selected = unique_labels
            else:
                perm = torch.randperm(len(unique_labels), device=device)[:Nc]
                selected = unique_labels[perm]

            pos_list = []
            gen_list = []
            valid_classes = []

            for c in selected:
                c_val = c.item()
                mask = (labels == c)
                target_latents = latents[mask]  # [N_c, 4, 32, 32]
                if target_latents.shape[0] == 0:
                    continue

                # Pick 1 random positive per class (no memory bank)
                rand_idx = torch.randint(0, target_latents.shape[0], (1,), device=device)
                pos_list.append(target_latents[rand_idx])  # [1, 4, 32, 32]

                # ════════════════════════════════════════════
                # Step B: (class + noise) -> DitGen -> generated_latents
                #   Class-conditioned, no CFG (alpha=1.0)
                # ════════════════════════════════════════════
                class_cond = torch.full((N_neg,), c_val, dtype=torch.long, device=device)
                gen_output = gen_model(class_cond, cfg_scale=cfg_scale)
                generated_latents = gen_output["samples"]  # [N_neg, 4, 32, 32]
                gen_list.append(generated_latents)

                valid_classes.append(c_val)

            if len(valid_classes) == 0:
                continue

            target_batch = torch.stack(pos_list, dim=0)       # [B_valid, 1, 4, 32, 32]
            gen_batch = torch.stack(gen_list, dim=0)           # [B_valid, N_neg, 4, 32, 32]

            # ════════════════════════════════════════════════
            # Step C: Feature extraction + contrastive loss
            #   Feature extractors are frozen (no learnable params)
            #   but NOT wrapped in no_grad — gradients flow through
            #   the feature transform back to the generator.
            #
            #   FeatureExtractor.forward internally:
            #     - target features extracted under torch.no_grad()
            #     - recon (generated) features keep full gradient
            #     - old_recon=None → defaults to recon.detach()
            #       (current generated samples, detached)
            #     - calls group_contra_loss from energy_loss.py
            # ════════════════════════════════════════════════
            total_loss = torch.zeros(len(valid_classes), device=device)
            all_info = {}

            for feat in feature_extractors:
                loss, info = feat(
                    target=target_batch,         # positive: real latents (no grad)
                    recon=gen_batch,              # generated: DitGen output (has grad)
                    contra_dict=contra_dict,
                    # old_recon omitted → defaults to recon.detach()
                )
                total_loss = total_loss + loss
                for k, v in info.items():
                    all_info[f"{feat.name()}/{k}"] = v

            avg_loss = total_loss.mean()

            # ════════════════════════════════════════════════
            # Step D: Backprop -> update DitGen only
            # ════════════════════════════════════════════════
            optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=clip_grad)
            optimizer.step()

            # ── Update EMA (used for evaluation only, not bootstrap) ──
            ema.update(gen_model)

            global_step += 1

            # ── Logging ──
            log_payload = {
                "train/loss": avg_loss.item(),
                "train/lr": optimizer.param_groups[0]['lr'],
                "train/step": global_step,
            }
            for k, v in all_info.items():
                if isinstance(v, torch.Tensor):
                    log_payload[f"train/{k}"] = v.item()
                else:
                    log_payload[f"train/{k}"] = v
            logger.log_dict(log_payload, step=global_step)
            logger.set_step(global_step)

            if is_main and global_step % 100 == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                log_str = (f"[epoch {epoch}][step {global_step}/{total_steps}] "
                           f"loss={avg_loss.item():.6f} lr={cur_lr:.2e}")
                print(log_str)

            # ── Checkpoint saving ──
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

            # ── FID evaluation + sample visualization ──
            if global_step % eval_every == 0:
                if is_main:
                    print(f"\n{'─'*40}")
                    print(f"FID evaluation at step {global_step}...")

                # ── Sample visualization (EMA model) ──
                ema_gen_fn = make_eval_generator(ema.model, vae, cfg_scale, device)
                visualize_imagenet_samples(
                    generator=ema_gen_fn,
                    logger=logger,
                    log_prefix=f"step_{global_step}",
                )

                # ── Evaluate EMA model FID ──
                ema_fid_result = eval_fid(
                    generator=ema_gen_fn,
                    cond_dataset=eval_dataset,
                    logger=logger,
                    total_samples=eval_fid_samples,
                    gpu_batch_size=eval_batch_size,
                    log_prefix=f"EMA_{global_step}",
                    dataset=dataset_name,
                    log_folder_name="eval",
                )
                if is_main:
                    fid_val = ema_fid_result.get("fid", float("nan"))
                    isc_val = ema_fid_result.get("isc_mean", float("nan"))
                    prec_val = ema_fid_result.get("precision", 0)
                    recall_val = ema_fid_result.get("recall", 0)
                    # Log summary metrics to wandb for easy tracking
                    logger.log_dict({
                        "eval/ema_fid": fid_val,
                        "eval/ema_isc": isc_val,
                        "eval/ema_precision": prec_val,
                        "eval/ema_recall": recall_val,
                    }, step=global_step)
                    print(f"EMA FID @ step {global_step}: {fid_val:.2f}  "
                          f"IS: {isc_val:.2f}  P: {prec_val:.3f}  R: {recall_val:.3f}")
                    print(f"{'─'*40}\n")

                # Sync all processes after eval
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

    # ── Final FID evaluation ──
    if is_main:
        print(f"\nFinal FID evaluation at step {global_step}...")
    ema_gen_fn = make_eval_generator(ema.model, vae, cfg_scale, device)
    visualize_imagenet_samples(
        generator=ema_gen_fn,
        logger=logger,
        log_prefix=f"final_{global_step}",
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

    # Config (all hyperparams live in the YAML config)
    parser.add_argument('--config', type=str, default='configs/dit_B2_clip.yaml',
                        help='Path to YAML config file')

    # Deployment-specific (override or not in config)
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
