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

import torch.utils.checkpoint as cp

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

    Memory optimizations:
    - Sub-batches the VAE+MoCo forward into chunks of `micro_batch`
    - Uses gradient checkpointing: only stores input/output per chunk,
      recomputes VAE+MoCo activations during backward (trades compute for memory).
    """

    def __init__(self, input_shape, vae, moco_backbone, micro_batch=64):
        super().__init__(input_shape)
        self.vae = vae
        self.moco = moco_backbone
        self.cache_scaling = 0.18125
        self.micro_batch = micro_batch
        # ImageNet normalization constants (for MoCo v2 input)
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _f_map_chunk(self, x):
        """Process one micro-batch: latents → VAE decode → MoCo v2 features.
        Separated so it can be wrapped by gradient checkpointing.
        """
        images = self.vae.decode(x / self.cache_scaling).sample
        images = ((images + 1) / 2).clamp(0, 1)  # [mb, 3, 256, 256]
        images = (images - self.img_mean) / self.img_std
        features = self.moco(images)  # [mb, 2048]
        return features

    def f_map(self, x):
        """
        x: [B, 4, 32, 32] latents (scaled by 0.18125)
        Returns: dict with 'moco' → [B, 1, 2048]

        Processes in micro-batches with gradient checkpointing to keep
        only one chunk's VAE+MoCo activations alive at a time.
        """
        B = x.shape[0]
        mb = self.micro_batch
        all_features = []
        for i in range(0, B, mb):
            chunk = x[i:i + mb]
            if torch.is_grad_enabled() and chunk.requires_grad:
                # Gradient checkpointing: recompute VAE+MoCo during backward
                feat = cp.checkpoint(self._f_map_chunk, chunk, use_reentrant=False)
            else:
                # No grad path (target samples) — no need to checkpoint
                feat = self._f_map_chunk(chunk)
            all_features.append(feat)
        features = torch.cat(all_features, dim=0)  # [B, 2048]
        return {"moco": features.unsqueeze(1)}  # [B, 1, 2048]

    def name(self):
        return "moco_v2"


# ─────────────────────────────────────────────────────────
# Per-class FIFO memory bank for multiple positive samples
# ─────────────────────────────────────────────────────────

class ClassMemoryBank:
    """Per-class FIFO memory bank for positive latent samples.

    A single dataloader batch (e.g. 512 samples across 1000 classes)
    rarely contains N_pos samples for any given class.  This bank
    accumulates latents across training steps so that we can draw
    N_pos diverse positives per class at every step.

    Storage is pre-allocated on GPU as a single contiguous tensor
    [num_classes, bank_size, *latent_shape] for efficiency.
    """

    def __init__(self, num_classes, bank_size, latent_shape, device):
        self.num_classes = num_classes
        self.bank_size = bank_size
        self.device = device
        # Pre-allocate storage — ~2 GB for 1000 classes × 128 × (4,32,32) in fp32
        self.storage = torch.zeros(
            num_classes, bank_size, *latent_shape, device=device
        )
        # Per-class circular write pointer and valid count (kept on CPU for indexing)
        self.ptr = torch.zeros(num_classes, dtype=torch.long)
        self.count = torch.zeros(num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, latents, labels):
        """Add a batch of latents to their respective class banks.

        Args:
            latents: [B, *latent_shape] — cached latents from dataloader.
            labels:  [B] — integer class labels.
        """
        for i in range(latents.shape[0]):
            c = labels[i].item()
            idx = self.ptr[c].item() % self.bank_size
            self.storage[c, idx].copy_(latents[i])
            self.ptr[c] += 1
            if self.count[c] < self.bank_size:
                self.count[c] += 1

    @torch.no_grad()
    def sample(self, class_label, n_samples):
        """Sample n_samples from the bank for *class_label*.

        Samples with replacement if n_samples > number of stored entries.
        Returns [n_samples, *latent_shape] on self.device, or None if empty.
        """
        n_valid = self.count[class_label].item()
        if n_valid == 0:
            return None
        indices = torch.randint(0, n_valid, (n_samples,))
        return self.storage[class_label, indices]  # already on device

    def n_valid(self, class_label):
        """Number of valid entries stored for *class_label*."""
        return self.count[class_label].item()


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

    # ── Optional torch.compile (before DDP) ──
    compile_model = cfg.train.get("compile_model", False)
    compile_mode = cfg.train.get("compile_mode", "default")
    if compile_model:
        generator = torch.compile(generator, mode=compile_mode)
        if is_main:
            print(f"Generator compiled with torch.compile (mode={compile_mode})")

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
    # NOTE: VAE decode is NOT compiled — diffusers' @apply_forward_hook
    # decorator is incompatible with torch._dynamo tracing.
    if is_main:
        print("Loaded SD-VAE decoder for MoCo v2 feature pipeline (frozen, grad-through)")

    # ── MoCo v2 backbone (frozen, but gradients flow through for gen path) ──
    moco_cfg = cfg.model.get("moco_v2", {})
    moco_checkpoint = moco_cfg.get("checkpoint_path", "")
    moco_backbone = build_moco_v2_backbone(moco_checkpoint, device)
    if compile_model:
        moco_backbone = torch.compile(moco_backbone, mode=compile_mode)
    if is_main:
        moco_params = sum(p.numel() for p in moco_backbone.parameters()) / 1e6
        compiled_tag = " [compiled]" if compile_model else ""
        print(f"MoCo v2 backbone: {moco_params:.1f}M params (frozen){compiled_tag}")

    # ── Feature extractor: MoCo v2 pixel-space features ──
    # Pipeline: latents → VAE decode → pixel images → MoCo v2 → 2048-d features
    input_shape = tuple(cfg.model.get("input_shape", [4, 32, 32]))
    feat_mb = cfg.train.get("feat_micro_batch", 64)
    moco_feature_extractor = MoCoV2PixelFeatures(
        input_shape, vae=train_vae, moco_backbone=moco_backbone, micro_batch=feat_mb
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

    # Positive sample config
    N_pos = train_cfg.get("n_pos", 1)
    pos_bank_size = train_cfg.get("pos_bank_size", 128)

    # Memory optimization config
    grad_accum_steps = train_cfg.get("grad_accum_steps", 1)
    use_bf16 = train_cfg.get("use_bf16", False)

    # Loss config (from config train.forward_dict.attn_dict)
    contra_dict = dict(
        kernel_type=attn_cfg.get("kernel_type", "attn_new"),
        sample_norm=attn_cfg.get("sample_norm", True),
        scale_dist_normed=attn_cfg.get("scale_dist_normed", True),
        R_list=attn_cfg.get("R_list", [0.02, 0.05, 0.2]),
    )

    # ── Per-class memory bank for multiple positives ──
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
        print(f"  Nc={Nc}, N_pos={N_pos}, N_neg={N_neg}")
        print(f"  Grad accum steps:     {grad_accum_steps} (micro_Nc ≈ {max(1, Nc // grad_accum_steps)})")
        print(f"  Mixed precision:      {'bf16' if use_bf16 else 'fp32'}")
        print(f"  Feat micro-batch:     {feat_mb} (VAE+MoCo sub-batch, checkpointed)")
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

            # ── Update per-class memory bank with current batch ──
            # This must happen BEFORE class selection so even on the
            # very first step, selected classes already have ≥1 entry.
            mem_bank.update(latents, labels)

            # ── Group batch by class, pick up to Nc classes ──
            unique_labels = labels.unique()
            if len(unique_labels) < Nc:
                selected = unique_labels
            else:
                perm = torch.randperm(len(unique_labels), device=device)[:Nc]
                selected = unique_labels[perm]

            # ── Split classes into micro-batches for gradient accumulation ──
            # Each micro-batch processes a subset of classes independently.
            # The contrastive loss is per-class, so accumulation is exact.
            n_selected = len(selected)
            micro_Nc = max(1, math.ceil(n_selected / grad_accum_steps))
            class_chunks = [selected[i:i+micro_Nc]
                            for i in range(0, n_selected, micro_Nc)]
            n_chunks = len(class_chunks)

            optimizer.zero_grad()
            accum_loss = 0.0
            accum_info = {}
            total_valid = 0

            for chunk_idx, class_chunk in enumerate(class_chunks):
                pos_list = []
                gen_list = []
                valid_classes = []

                for c in class_chunk:
                    c_val = c.item()

                    # Draw N_pos positives from the memory bank
                    pos_samples = mem_bank.sample(c_val, N_pos)
                    if pos_samples is None:
                        continue  # class not yet in the bank (shouldn't happen after update)
                    pos_list.append(pos_samples)  # [N_pos, 4, 32, 32]

                    # ════════════════════════════════════════════
                    # Step B: (class + noise) -> DitGen -> generated_latents
                    #   Class-conditioned, no CFG (alpha=1.0)
                    #   bf16 autocast keeps activations in half-precision
                    # ════════════════════════════════════════════
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
                        class_cond = torch.full((N_neg,), c_val, dtype=torch.long, device=device)
                        gen_output = gen_model(class_cond, cfg_scale=cfg_scale)
                        generated_latents = gen_output["samples"]  # [N_neg, 4, 32, 32]
                    gen_list.append(generated_latents)

                    valid_classes.append(c_val)

                if len(valid_classes) == 0:
                    continue

                target_batch = torch.stack(pos_list, dim=0)       # [chunk_valid, N_pos, 4, 32, 32]
                gen_batch = torch.stack(gen_list, dim=0)           # [chunk_valid, N_neg, 4, 32, 32]

                # ════════════════════════════════════════════════
                # Step C: Feature extraction + contrastive loss
                #   MoCoV2PixelFeatures uses gradient checkpointing +
                #   sub-batching internally for memory efficiency.
                #   Target features extracted under no_grad (by FeatureExtractor).
                #   Generated features keep full gradient back to DitGen.
                # ════════════════════════════════════════════════
                chunk_loss = torch.zeros(len(valid_classes), device=device)
                chunk_info = {}

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
                    for feat in feature_extractors:
                        loss, info = feat(
                            target=target_batch,
                            recon=gen_batch,
                            contra_dict=contra_dict,
                        )
                        chunk_loss = chunk_loss + loss
                        for k, v in info.items():
                            chunk_info[f"{feat.name()}/{k}"] = v

                # Scale loss for gradient accumulation and backward
                avg_chunk_loss = chunk_loss.mean() / n_chunks
                avg_chunk_loss.backward()

                # Accumulate metrics for logging
                accum_loss += chunk_loss.mean().item() / n_chunks
                for k, v in chunk_info.items():
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    accum_info[k] = accum_info.get(k, 0) + val / n_chunks
                total_valid += len(valid_classes)

            if total_valid == 0:
                continue

            # ════════════════════════════════════════════════
            # Step D: Gradient clipping + optimizer step
            #   Gradients accumulated from all class micro-batches
            # ════════════════════════════════════════════════
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=clip_grad)
            optimizer.step()

            # ── Update EMA (used for evaluation only, not bootstrap) ──
            ema.update(gen_model)

            global_step += 1

            # ── Logging ──
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
