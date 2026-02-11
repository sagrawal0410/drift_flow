# ── Checkpoint storage ──
# Root directory where training checkpoints are saved/loaded by ckpt_utils.py.
# Structure: <ckpt_root>/<run_id>/checkpoints/checkpoint-<step>.pth
ckpt_root = "/data/scratch-oc40/shaurya10/drift_flow/checkpoints/"

# ── FID reference statistics ──
# Precomputed Inception statistics (.npz) for FID computation.
# Used by fid.py -> metrics_dataset() to compare generated samples against.
dataset_to_fid_path = {
    "imagenet256": "/data/scratch-oc40/shaurya10/drift_flow/utils/fid_stats/adm_in256_stats.npz",
}

# ── Precision / Recall reference path ──
# Directory of real images used for precision/recall metrics (optional).
# Only needed if eval_prc_recall=True. Empty string disables it.
prc_recall_path = ""

# ── CLIP-FID cache ──
# Directory where CLIP feature caches are stored for CLIP-FID evaluation.
# Used by custom_fid.py. Empty string or any writable path works.
clip_fid_cache_path = "/data/scratch-oc40/shaurya10/drift_flow/cache/clip_fid"

# ── Dataset paths ──
# Root of the standard ImageNet dataset (with train/ and val/ subdirectories).
# Used by dataset.py and custom_fid.py for loading real images.
imagenet_root = "/data/infolab/aia/imagenet_pytorch"

# Root of cached VAE latents, keyed by split.
# Used by dataset.py when loading cached latents without tar.
imagenet_cache_root = {
    "train": "/data/scratch-oc40/shaurya10/cache_latents/train",
    "val": "/data/scratch-oc40/shaurya10/cache_latents/val",
}

# Download directory for small datasets (MNIST, CIFAR-10).
# Used by dataset.py for torchvision dataset downloads.
dataset_download_path = "/data/scratch-oc40/shaurya10/drift_flow/data"

# ── Tar-based dataset paths (optional, for SLURM local-disk caching) ──
# Path to tarball of ImageNet images. Only used if tar_local=True.
imagenet_tar_path = ""

# Paths to tarballs of cached latents, keyed by split. Only used if tar_local=True.
imagenet_cache_tar_path = {
    "train": "",
    "val": "",
}

# Number of leading path components to strip when untarring. Only used if tar_local=True.
strip_imagenet_tar = 0
strip_imagenet_cache_tar = 0

# Temporary folder for untarring. Only used if tar_local=True.
tmp_folder = "/tmp/drift_flow"

# ── torch.compile mode ──
# Controls torch.compile behavior in utils/misc.py.
# Options: "none" (no compilation), "default", "reduce-overhead", "max-autotune"
# "none" is safest and avoids compilation overhead during development.
compile_mode = "none"
