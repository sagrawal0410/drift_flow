#!/bin/bash
#SBATCH --job-name=cache_latents
#SBATCH --partition=vision-he-h200
#SBATCH --account=vision-he
#SBATCH --qos=vision-he-low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cache_latents_%j.out
#SBATCH --error=logs/cache_latents_%j.err

set -e  # exit on first error

mkdir -p logs

# ── Conda activation (required in SLURM batch scripts) ──
source /data/scratch-oc40/shaurya10/miniconda3/etc/profile.d/conda.sh
conda activate drift_flow

# ── Distributed rendezvous (required for torchrun inside SLURM) ──
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(( 29500 + RANDOM % 1000 ))
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"

# Auto-detect GPU count from SLURM allocation
NGPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
echo "Using $NGPUS GPUs"

# Cache train split
torchrun --nproc_per_node=$NGPUS dataset/cache_latent.py \
    --data_path /data/infolab/aia/imagenet_pytorch \
    --cached_path /data/scratch-oc40/shaurya10/cache_latents/train \
    --split train \
    --batch_size 128

# Cache val split
torchrun --nproc_per_node=$NGPUS dataset/cache_latent.py \
    --data_path /data/infolab/aia/imagenet_pytorch \
    --cached_path /data/scratch-oc40/shaurya10/cache_latents/val \
    --split val \
    --batch_size 128
