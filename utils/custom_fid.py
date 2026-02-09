

import os
import torch
import cv2
import numpy as np
import pickle
import scipy.linalg
from typing import Any
from utils.distributed_utils import init_distributed_mode

from torch_fidelity import calculate_metrics
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import sys
sys.path.append('..')
from utils.distributed_utils import get_rank, get_world_size, broadcast_object
from utils.misc import TemporalSeed
import time
import requests 
from tqdm import tqdm
import clip
from utils.global_path import clip_fid_cache_path
# from utils import torch_utils
def custom_fid(real_dir, gen_dir, gen_is_train=False):
    """
    Compute the FID score between real and generated images.
    Args:
        real_dir (str): Path to the directory containing real images.
        gen_dir (str): Path to the directory containing generated images.
        gen_is_train (bool): Whether the generated images are from train set
            if true: the folder will contains lots of a/b.jpeg; should only sample 50 per folder. 
    Returns:
        fid_score (float): The FID score between real and generated images.
    """
    import os
    import numpy as np
    import torch
    import scipy.linalg
    from tqdm import tqdm
    import clip
    from utils.distributed_utils import get_rank, get_world_size

    model, preprocess = clip.load("ViT-B/32", device="cuda")

    def aggr_features(dir_path, is_train=False):
        """Aggregate CLIP features from all images in the directory."""
        import glob
        from PIL import Image

        if is_train:
            img_files = []
            subdirs = sorted([d.path for d in os.scandir(dir_path) if d.is_dir()])
            for subdir in subdirs:
                # The docstring mentions "a/b.jpeg" which implies one level of subdirectories
                subdir_files = sorted(
                    glob.glob(os.path.join(subdir, "*.png"))
                    + glob.glob(os.path.join(subdir, "*.jpg"))
                    + glob.glob(os.path.join(subdir, "*.jpeg"))
                    + glob.glob(os.path.join(subdir, "*.JPEG"))
                )
                img_files.extend(subdir_files[:50])
        else:
            img_files = sorted(
                glob.glob(os.path.join(dir_path, "**", "*.png"), recursive=True)
                + glob.glob(os.path.join(dir_path, "**", "*.jpg"), recursive=True)
                + glob.glob(os.path.join(dir_path, "**", "*.jpeg"), recursive=True)
                + glob.glob(os.path.join(dir_path, "**", "*.JPEG"), recursive=True)
            )

        world_size = get_world_size()
        rank = get_rank()
        if world_size > 1:
            img_files = [img for i, img in enumerate(img_files) if (i % world_size) == rank]

        print(f"Processing {len(img_files)} images on rank {rank}")

        features = []
        batch_size = 64  # adjust as needed for GPU memory

        # Read and process images in batches
        for i in tqdm(range(0, len(img_files), batch_size)):
            batch_files = img_files[i : i + batch_size]
            ims = []
            with torch.no_grad():
                for fname in batch_files:
                    img = Image.open(fname).convert("RGB")
                    arr = preprocess(img)
                    ims.append(arr)
                ims_torch = torch.stack(ims, dim=0).cuda()
                feats = model.encode_image(ims_torch)
                features.append(feats)

        features = torch.cat(features, dim=0)

        if world_size > 1:
            # gather features from all ranks
            gathered = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered, features.cpu())
            return torch.cat(gathered, dim=0)
        else:
            return features

    def compute_fid(real_features: np.ndarray, gen_features: np.ndarray) -> float:
        """Compute the Fréchet Inception Distance (FID)."""
        assert real_features.ndim == 2 and gen_features.ndim == 2
        # Feature statistics
        mu_real = np.mean(real_features, axis=0)
        mu_gen = np.mean(gen_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        sigma_gen = np.cov(gen_features, rowvar=False)

        # FID
        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)
        fid = np.real(m + np.trace(sigma_gen + sigma_real - 2 * s))
        return fid

    feature_gen = aggr_features(gen_dir, is_train=gen_is_train).cpu().numpy()

    # Caching for real features
    rank = get_rank()
    world_size = get_world_size()

    feature_real = None
    cache_exists = os.path.exists(clip_fid_cache_path)

    if world_size > 1:
        # Broadcast cache existence from rank 0 to all other processes
        cache_exists_tensor = torch.tensor(int(cache_exists), dtype=torch.int).cuda()
        torch.distributed.broadcast(cache_exists_tensor, src=0)
        cache_exists = bool(cache_exists_tensor.item())

    if cache_exists:
        if rank == 0:
            print(f"Loading real image features from cache: {clip_fid_cache_path}")
            with open(clip_fid_cache_path, "rb") as f:
                feature_real = pickle.load(f)
        if world_size > 1:
            feature_real = broadcast_object(feature_real)
    else:
        feature_real_tensor = aggr_features(real_dir)
        feature_real = feature_real_tensor.cpu().numpy()
        if rank == 0:
            print(f"Saving real image features to cache: {clip_fid_cache_path}")
            os.makedirs(os.path.dirname(clip_fid_cache_path), exist_ok=True)
            with open(clip_fid_cache_path, "wb") as f:
                pickle.dump(feature_real, f)
    
    if world_size > 1:
        torch.distributed.barrier()

    fid = compute_fid(feature_real, feature_gen)
    return fid


from utils.global_path import imagenet_root
def fid_clip(gen_dir, gen_is_train=False):
    val_dir = os.path.join(imagenet_root, "val")
    print("VAL DIR", val_dir)
    fid = custom_fid(val_dir, gen_dir, gen_is_train=gen_is_train)
    return fid

if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dist_url = "env://"
    init_distributed_mode(args)

    train_dir_path=os.path.join(imagenet_root, "train")
    print("TRAIN DIR", train_dir_path)
    print("Train FID", fid_clip(train_dir_path, gen_is_train=True))
    

    gen_dir_Lnocls = "/checkpoint/flows/mingyangd/nn_flow/runs/L_resnetXL100_nocls_ev4_20251020_031402/samples/6f72ad417ae82940e9981c82c636f11f/EMA_50000"
    print("LNOCLS", fid_clip(gen_dir_Lnocls))

    gen_dir = "/checkpoint/flows/mingyangd/nn_flow/runs/L_resnetL_finetune_ev4_bf16_20251021_054229/samples/2717c5da6f19da360a6176e795fa6a9e/EMA_50000"
    fid = fid_clip(gen_dir)
    print("FINETUNE", fid)
    gen_dir_nocls = "/checkpoint/flows/mingyangd/nn_flow/runs/L_resnetB_nocls_ev4_bf16_20251020_001637/samples/e47151518304b9a172c5a9828c6da1d9/EMA_50000"
    fid_nocls = fid_clip(gen_dir_nocls)
    print("NOCLS", fid_nocls)
# torchrun --nproc_per_node=8 -m utils.custom_fid
# L_resnetL_finetune_ev4_bf16_20251021_054229 /checkpoint/flows/mingyangd/nn_flow/runs/L_resnetL_finetune_ev4_bf16_20251021_054229/samples/2717c5da6f19da360a6176e795fa6a9e/EMA_50000
# L_resnetB_nocls_ev4_bf16_20251020_001637 /checkpoint/flows/mingyangd/nn_flow/runs/L_resnetB_nocls_ev4_bf16_20251020_001637/samples/e47151518304b9a172c5a9828c6da1d9/EMA_50000
#  python -m utils.custom_fid 
# FINETUNE 4.15047543435238
# NOCLS 4.287063432312976