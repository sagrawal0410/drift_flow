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


class MoCoV2MultiScaleFeatures(FeatureExtractor):
    """Multi-scale feature extractor following Section A.5 of the drifting paper.

    Pipeline: latents → VAE decode → pixel images → MoCo v2 ResNet-50

    Extracts feature maps at every 2 residual blocks + final layer in each
    stage of the ResNet-50, plus the VAE decoder output (encoder input).

    For each feature map [B, C, H, W], produces (per A.5):
      (a) H×W per-location vectors (each C-dim)
      (b) 1 global mean + 1 global std (each C-dim)
      (c) (H/2)×(W/2) means + stds from 2×2 patches (each C-dim)
      (d) (H/4)×(W/4) means + stds from 4×4 patches (each C-dim)

    For the encoder input (pixel images), computes mean(x²) per channel.

    Each feature type at each extraction point gets its own independent
    drifting loss (computed by FeatureExtractor.forward → group_contra_loss).

    ResNet-50 (bottleneck, [3,4,6,3]) extraction points:
      layer1: after block 1, block 2 (final)     → 64×64×256   (2 maps)
      layer2: after block 1, block 3 (final)     → 32×32×512   (2 maps)
      layer3: after block 1, 3, block 5 (final)  → 16×16×1024  (3 maps)
      layer4: after block 1, block 2 (final)     → 8×8×2048    (2 maps)
      Total: 9 feature maps × 4 types + 1 input  = 37 loss terms

    Memory optimizations (same as before):
      - Sub-batches VAE+MoCo forward into chunks of `micro_batch`
      - Gradient checkpointing: recompute during backward
    """

    # (layer_name, extract_after_block_indices)
    # "every 2 residual blocks + final" per stage of ResNet-50 [3,4,6,3]
    STAGES = [
        ("layer1", [1, 2]),       # 3 blocks → after block 1, 2 (final)
        ("layer2", [1, 3]),       # 4 blocks → after block 1, 3 (final)
        ("layer3", [1, 3, 5]),    # 6 blocks → after block 1, 3, 5 (final)
        ("layer4", [1, 2]),       # 3 blocks → after block 1, 2 (final)
    ]

    def __init__(self, input_shape, vae, moco_backbone, micro_batch=64):
        super().__init__(input_shape)
        self.vae = vae
        self.cache_scaling = 0.18125
        self.micro_batch = micro_batch

        # Unwrap torch.compile wrapper if present — we need direct layer access
        if hasattr(moco_backbone, '_orig_mod'):
            self.moco = moco_backbone._orig_mod
        else:
            self.moco = moco_backbone

        # ImageNet normalization constants (for MoCo v2 input)
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # Precompute ordered feature names (needed for tuple↔dict in checkpointing)
        self._feature_names = self._build_feature_names()

    def _build_feature_names(self):
        """Build ordered list of all feature names returned by f_map."""
        names = []
        for layer_name, extract_blocks in self.STAGES:
            for bidx in extract_blocks:
                prefix = f"{layer_name}_b{bidx}"
                names.extend([
                    f"{prefix}_loc", f"{prefix}_global",
                    f"{prefix}_p2", f"{prefix}_p4",
                ])
        names.append("input_x2mean")
        return names

    @staticmethod
    def _patch_stats(feat, patch_size):
        """Compute mean and std over non-overlapping patches.

        Args:
            feat: [B, C, H, W]
            patch_size: int
        Returns:
            [B, 2 * N_patches, C]  where N_patches = (H//p) * (W//p)
            First N_patches entries are means, next N_patches are stds.
        """
        B, C, H, W = feat.shape
        p = patch_size
        Hp, Wp = H // p, W // p
        # Reshape into patches: [B, C, Hp, p, Wp, p] → [B, Hp*Wp, C, p²]
        f = feat[:, :, :Hp * p, :Wp * p]
        f = f.reshape(B, C, Hp, p, Wp, p)
        f = f.permute(0, 2, 4, 1, 3, 5).reshape(B, Hp * Wp, C, p * p)
        means = f.mean(-1)   # [B, N, C]
        stds = f.std(-1)     # [B, N, C]
        return torch.cat([means, stds], dim=1)   # [B, 2N, C]

    def _feature_vectors(self, feat, prefix):
        """Extract (a)–(d) feature vectors from one feature map.

        Args:
            feat: [B, C, H, W]
        Returns:
            dict  {name: [B, F, C]}   (4 entries)
        """
        B, C, H, W = feat.shape
        out = {}

        # (a) Per-location vectors: [B, H*W, C]
        out[f"{prefix}_loc"] = feat.reshape(B, C, H * W).permute(0, 2, 1)

        # (b) Global mean + std: [B, 2, C]
        gmean = feat.mean(dim=(2, 3))  # [B, C]
        gstd = feat.std(dim=(2, 3))    # [B, C]
        out[f"{prefix}_global"] = torch.stack([gmean, gstd], dim=1)

        # (c) 2×2 patch means + stds
        if H >= 2 and W >= 2:
            out[f"{prefix}_p2"] = self._patch_stats(feat, 2)
        else:
            out[f"{prefix}_p2"] = torch.stack([gmean, gstd], dim=1)

        # (d) 4×4 patch means + stds
        if H >= 4 and W >= 4:
            out[f"{prefix}_p4"] = self._patch_stats(feat, 4)
        else:
            out[f"{prefix}_p4"] = torch.stack([gmean, gstd], dim=1)

        return out

    def _f_map_chunk(self, x):
        """Process one micro-batch: latents → VAE → MoCo ResNet-50 → multi-scale features.

        Returns a tuple of tensors in self._feature_names order
        (torch.utils.checkpoint requires tensor/tuple output).
        """
        # ── Decode latents → pixel images ──
        images = self.vae.decode(x / self.cache_scaling).sample
        images = ((images + 1) / 2).clamp(0, 1)        # [mb, 3, 256, 256]
        images_normed = (images - self.img_mean) / self.img_std

        result = {}

        # ── Run ResNet-50 stem ──
        moco = self.moco
        h = moco.conv1(images_normed)
        h = moco.bn1(h)
        h = moco.relu(h)
        h = moco.maxpool(h)  # [mb, 64, 64, 64]

        # ── Run each stage, extracting feature maps at specified blocks ──
        for layer_name, extract_blocks in self.STAGES:
            layer = getattr(moco, layer_name)
            for bidx, block in enumerate(layer):
                h = block(h)
                if bidx in extract_blocks:
                    prefix = f"{layer_name}_b{bidx}"
                    result.update(self._feature_vectors(h, prefix))

        # ── Encoder input feature: mean of x² per channel → [mb, 1, 3] ──
        result["input_x2mean"] = (images ** 2).mean(dim=(2, 3)).unsqueeze(1)

        # Return as ordered tuple for checkpoint compatibility
        return tuple(result[name] for name in self._feature_names)

    def f_map(self, x):
        """
        x: [B, 4, 32, 32] latents (scaled by 0.18125)
        Returns: dict {name: [B, F, D]} with 37 entries (9 maps × 4 types + 1 input).

        Processes in micro-batches with gradient checkpointing.
        """
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

        # Concatenate along batch dim and rebuild dict
        result = {}
        for idx, name in enumerate(self._feature_names):
            result[name] = torch.cat([c[idx] for c in all_chunks], dim=0)
        return result

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
        # Early in training the generator can produce extreme latents that
        # decode to NaN/Inf — replace before clamping so FID eval doesn't crash.
        images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=-1.0)
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
    if is_main:
        print(f"Loaded SD-VAE decoder for MoCo v2 feature pipeline (frozen, grad-through)")

    # ── MoCo v2 backbone (frozen, but gradients flow through for gen path) ──
    moco_cfg = cfg.model.get("moco_v2", {})
    moco_checkpoint = moco_cfg.get("checkpoint_path", "")
    moco_backbone = build_moco_v2_backbone(moco_checkpoint, device)
    # Note: MoCo backbone is NOT compiled because the multi-scale feature
    # extractor needs direct layer-by-layer access to extract intermediate maps.
    if is_main:
        moco_params = sum(p.numel() for p in moco_backbone.parameters()) / 1e6
        print(f"MoCo v2 backbone: {moco_params:.1f}M params (frozen, multi-scale)")

    # ── Feature extractor: MoCo v2 multi-scale pixel-space features (A.5) ──
    # Pipeline: latents → VAE decode → pixels → MoCo ResNet-50 layer-by-layer
    #   → 9 feature maps (every 2 blocks + final per stage) × 4 feature types
    #   + 1 input-level feature = 37 independent drifting loss terms
    input_shape = tuple(cfg.model.get("input_shape", [4, 32, 32]))
    feat_mb = cfg.train.get("feat_micro_batch", 64)
    moco_feature_extractor = MoCoV2MultiScaleFeatures(
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
        n_feat = len(moco_feature_extractor._feature_names)
        print(f"  Feature pipeline:     latents → VAE decode → MoCo ResNet-50 (multi-scale, A.5)")
        print(f"  Feature loss terms:   {n_feat} (9 maps × 4 types + 1 input)")
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
                #   MoCoV2MultiScaleFeatures uses gradient checkpointing +
                #   sub-batching internally for memory efficiency (A.5).
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
            grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=clip_grad)

            # Skip update if gradients are NaN/Inf (early training instability)
            if not torch.isfinite(grad_norm):
                if is_main:
                    print(f"[step {global_step}] WARNING: non-finite grad norm ({grad_norm:.4f}), skipping update")
                optimizer.zero_grad()
            else:
                optimizer.step()

──
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
