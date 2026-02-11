#!/bin/bash
#SBATCH --job-name=train_clip
#SBATCH --partition=vision-he-h200
#SBATCH --account=vision-he
#SBATCH --qos=vision-he-low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_clip_%j.out
#SBATCH --error=logs/train_clip_%j.err

set -e

mkdir -p logs

# ── Conda activation ──
source /data/scratch-oc40/shaurya10/miniconda3/etc/profile.d/conda.sh
conda activate drift_flow

# ── Wandb API key (compute nodes may not have login credentials) ──
export WANDB_API_KEY="wandb_v1_X76mfMNkN3pcus42Lt1vTCKpKua"

# ── Distributed rendezvous ──
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(( 29500 + RANDOM % 1000 ))
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"

# ── Pre-download SD-VAE (single process, avoids HF cache race condition) ──
# The VAE decoder is needed every step (latents → pixels → MoCo v2 features)
# and also for FID evaluation. MoCo v2 loads from local checkpoint, no download needed.
echo "Pre-downloading SD-VAE..."
python -c "
from diffusers import AutoencoderKL
AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema')
print('SD-VAE cached.')
"

# ── Train ──
torchrun --nproc_per_node=8 train_drift_clip.py \
    --config configs/dit_B2_clip.yaml \
    --cached_path /data/scratch-oc40/shaurya10/cache_latents/train
