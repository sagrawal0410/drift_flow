#!/bin/bash
#SBATCH --job-name=cache_latents
#SBATCH --partition=vision-he-h200
#SBATCH --account=vision-he
#SBATCH --qos=vision-he-low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cache_latents_%j.out
#SBATCH --error=logs/cache_latents_%j.err

# ── Paths (edit these) ──
DATA_PATH=/path/to/imagenet
CACHE_PATH=/path/to/cache

mkdir -p logs

conda activate drift

# Cache train split
python dataset/cache_latent.py \
    --data_path "$DATA_PATH" \
    --cached_path "${CACHE_PATH}/train" \
    --split train \
    --batch_size 128

# Cache val split
python dataset/cache_latent.py \
    --data_path "$DATA_PATH" \
    --cached_path "${CACHE_PATH}/val" \
    --split val \
    --batch_size 128
