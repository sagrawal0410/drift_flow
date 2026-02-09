import torch
# from numpy._core.multiarray import _reconstruct
import numpy as np

torch.serialization.add_safe_globals([
    # _reconstruct,
    np.ndarray,
    np.dtype,
    np.generic,
    np.float32,
    np.float64,  # 如有需要也可添加
])
import os
import torch
import cv2
import numpy as np
import pickle
import scipy.linalg
from typing import Any

from torch_fidelity import calculate_metrics
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import sys
sys.path.append('..')
from utils.distributed_utils import get_rank, get_world_size, broadcast_object
from utils.misc import TemporalSeed
import time
from tqdm import tqdm
from utils.distributed_utils import get_wandb_id
from utils.global_path import dataset_to_fid_path
from utils.pixel_fid import mnist_pixel_fid
from utils.custom_fid import fid_clip
from utils.global_path import prc_recall_path
class ImageFolder:
    def __init__(self, path, world_size=1, global_rank=0, mode="clean"):
        self.path = path
        self.mode = mode
        self.world_size = world_size
        self.global_rank = global_rank
        self.counter = 0
        self.clean()

    def id_to_global_id(self, n):
        return n * self.world_size + self.global_rank

    def remove_all_images(self):
        if not os.path.exists(self.path):
            return
        for file in os.listdir(self.path):
            os.remove(os.path.join(self.path, file))

    def clean(self):
        self.counter = 0
        if os.path.exists(self.path):
            return
        os.makedirs(self.path, exist_ok=True)

    def get_image_based_on_id(self, n):
        img = cv2.imread(
            os.path.join(self.path, f"img_{n}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        # Convert BGR to RGB
        if img.shape[-1] == 3:
            img = img[..., ::-1]
        img = torch.from_numpy(np.array(img))
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        else:
            img = img.permute(2, 0, 1)
        return img

    def get_first_n_images_uint8(self, n):
        """
        Returns a batch of images from the folder.
        Args:
            n (int): The number of images to return.
        Returns:
            images (B, C, H, W): A batch of images in RGB format.
        """
        list_of_images = [self.get_image_based_on_id(i) for i in range(n)]
        return torch.stack(list_of_images)

    def get_first_n_images_01(self, n):
        return self.get_first_n_images_uint8(n) / 255

    def process_and_save(self, imgs):
        """
        Process and save a batch of images.
        Args:
            imgs (Tensor): A batch of images (B, C, H, W); range should be [0, 1]
        """
        batch_size = imgs.size(0)

        # Assert that the image tensor values are within the range [0, 1]
        assert torch.all(imgs >= -1e-3) and torch.all(
            imgs <= 1 + 1e-3
        ), "Images should be in the range [0, 1]"

        # Process each image, convert to [0, 255], and save
        for i in range(batch_size):
            img_np = (
                imgs[i].permute(1, 2, 0).cpu().numpy()
            )  # Convert from (C, H, W) to (H, W, C)
            img_np = (img_np * 255).astype(np.uint8)  # Convert to range [0, 255]

            # If image has 3 channels (RGB), use [::-1] to reverse RGB to BGR for cv2, else for grayscale, leave it as is
            if img_np.shape[-1] == 3:
                img_np = img_np[:, :, ::-1]  # Reverse the last channel (RGB to BGR)

            # Save the image
            img_filename = f"img_{self.id_to_global_id(self.counter)}.png"
            full_path = os.path.join(self.path, img_filename)
            cv2.imwrite(full_path, img_np)
            self.counter += 1



def fid_between_folders(real_dir, gen_dir):
    """
    Computes the FID score between real and generated images.
    Args:
        real_dir (str): Path to the directory containing real images.
        gen_dir (str): Path to the directory containing generated images.
    Returns:
        fid_score (float): The FID score between real and generated images.
    """
    # fid_score = fid.compute_fid(real_dir, gen_dir)
    # return fid_score
    metrics = calculate_metrics(
        input1=real_dir,
        input2=gen_dir,
        fid_statistics_file=None,
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=False,
    )
    fid_score = metrics["frechet_inception_distance"]
    return fid_score


def fid_dataset(gen_dir, dataset):
    """
    Computes the FID score between generated images and a dataset.
    Args:
        gen_dir (str): Path to the directory containing generated images.
        dataset (str): dataset name. Example: "cifar10-train", "imagenet-val".
    Returns:
        fid_score (float): The FID score between real and generated images.
    """
    metrics_dict = metrics_dataset(gen_dir, dataset)
    fid_score = metrics_dict["frechet_inception_distance"]
    return fid_score


def metrics_dataset(gen_dir, dataset, eval_prc_recall=False):
    """
    Compute metrics (FID and Inception Score) between generated images and a dataset.
    Returns a dictionary from torch_fidelity with keys including:
      - 'frechet_inception_distance'
      - 'inception_score_mean'
      - 'inception_score_std'
      - 'precision' (conditionally for ImageNet)
      - 'recall' (conditionally for ImageNet)
    """
    if dataset == "mnist":
        # Inception score is not meaningful for MNIST here; only return FID via pixel metric.
        fid_score = mnist_pixel_fid(gen_dir)
        return {
            "frechet_inception_distance": fid_score,
            "inception_score_mean": 0,
            "inception_score_std": 0,
            "precision": 0,
            "recall": 0,

        }
    elif 'cifar' in dataset:
        if dataset == "cifar" or dataset == "cifar10":
            dataset = "cifar10-train"
        metrics_dict = calculate_metrics(
            input1=gen_dir,
            input2=dataset,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=True,
            verbose=False,
        )
        return metrics_dict
    else:
        dataset_name = dataset
        if dataset_name == "imagenet256_cache":
            dataset_name = "imagenet256"
        metrics_dict = calculate_metrics(
            input1=gen_dir,
            input2=None,
            fid_statistics_file=dataset_to_fid_path[dataset_name],
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        if 'imagenet256' in dataset_name and eval_prc_recall:
            prec_recall_dict = calculate_metrics(
                input1=prc_recall_path,
                input2=gen_dir,
                cuda=True,
                isc=False,
                fid=False,
                prc=True,
                verbose=False,
            )
            metrics_dict["precision"] = prec_recall_dict["precision"]
            metrics_dict["recall"] = prec_recall_dict["recall"]
        return metrics_dict


class TemporalFolders:
    def __init__(self, world_size=1, global_rank=0, path=""):
        self.folders = {}  # Initialize a dictionary to store ImageFolder instances
        self.world_size = world_size
        self.global_rank = global_rank
        self.path = path

    def get(self, folder_name):
        if folder_name not in self.folders:
            self.folders[folder_name] = ImageFolder(
                f"{self.path}/{folder_name}",
                world_size=self.world_size,
                global_rank=self.global_rank,
            )
            self.folders[folder_name].clean()
        return self.folders[folder_name]

    def clean(self, folder_name):
        self.get(folder_name).clean()

    def add(self, folder_name, imgs):
        self.get(folder_name).process_and_save(imgs)

    def fid_dataset(self, folder_name, dataset):
        return fid_dataset(self.get(folder_name).path, dataset)

    def fid_between(self, folder_name1, folder_name2):
        return fid_between_folders(
            self.get(folder_name1).path, self.get(folder_name2).path
        )


def visualize_imagenet_samples(generator, logger, class_list=[207,360,388,113,355,980,323,979], samples_per_class=8, log_prefix=""):
    '''
    Args:
        generator: a function, takes in a batch of classes, returns a batch of samples (B, C, H, W) within range [0, 1]
        logger: a logger, to log the samples
        class_list: a list of class ids
        samples_per_class: the number of samples to generate per class
        log_prefix: the prefix for logging
    '''
    if get_rank() > 0:
        return 
    with torch.inference_mode():
        id_tensor = torch.tensor([x for x in class_list for _ in range(samples_per_class)], device='cuda')
        gen_samples = generator(id_tensor)
        logger.log_image(f"viz/{log_prefix}", gen_samples)

def prepare_val_prec_recall():
    '''
    Prepare the precision and recall dataset for ImageNet. 
    Will store the images in the path prc_recall_path.
    '''
    from utils.global_path import prc_recall_path
    # remember to first wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
    npz_path = "/checkpoint/flows/mingyangd/VIRTUAL_imagenet256_labeled.npz"
    goal_path = prc_recall_path
    
    if not os.path.exists(goal_path):
        os.makedirs(goal_path)
    
    # Load images from NPZ file
    data = np.load(npz_path)
    images = data['arr_0']  # Shape: (10000, 256, 256, 3)
    
    # Save images as PNG files
    for img_idx in tqdm(range(len(images))):
        img = images[img_idx]  # Shape: (256, 256, 3)
        # Ensure image is in [0, 255] range
        print(img.max(), img.min())
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        # Convert RGB to BGR for cv2
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        dst_filename = f"{img_idx:04d}.png"
        dst = os.path.join(goal_path, dst_filename)
        cv2.imwrite(dst, img_bgr)


def eval_fid(
    generator,
    cond_dataset,
    logger,
    total_samples=5000,
    gpu_batch_size=768,
    log_prefix="",
    dataset="cifar10-train",
    log_folder_name="",
    eval_clip=False, 
    eval_prc_recall=False,
):
    """
    Generate and evaluate FID for a given generator and dataset.
    Args:
        generator: a function, takes in a batch in dataset, returns a batch of samples (B, C, H, W) within range [0, 1]
        cond_dataset: a dataset, each index returns a batch of cond.
        logger: a logger, to log the FID
        n_samples: the number of samples to generate
        batch_size: the batch size for generating samples
        log_prefix: the prefix for logging
    Returns:
        fid: the FID score
    """

    distributed = torch.distributed.is_initialized()
    if distributed:
        torch.distributed.barrier()
    start_time = time.time()

    # Setup for distributed evaluation
    rank = get_rank() if distributed else 0
    world_size = get_world_size() if distributed else 1

    # Create a sampler for distributed data loading
    context = TemporalSeed(rank)
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            cond_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        loader = DataLoader(cond_dataset, batch_size=gpu_batch_size, sampler=sampler)
    else:
        loader = DataLoader(cond_dataset, batch_size=gpu_batch_size, shuffle=True)

    # context and folder
    folder = ImageFolder(
        f"samples/{get_wandb_id()}/{log_prefix}_{total_samples}", world_size=get_world_size(), global_rank=get_rank()
    )

    
    if distributed:
        torch.distributed.barrier()

    print("Removal time:", time.time() - start_time)

    # generate samples
    current_samples = 0
    samples_per_gpu = total_samples // get_world_size()
    loader_iter = tqdm(loader) if rank == 0 else loader

    with torch.inference_mode():
        for batch in loader_iter:
            sample_size = min(gpu_batch_size, samples_per_gpu - current_samples)
            if sample_size <= 0:
                break

            if sample_size < gpu_batch_size:
                if isinstance(batch, tuple) or isinstance(batch, list):
                    batch = [x[:sample_size] for x in batch]
                else:
                    assert isinstance(batch, torch.Tensor)
                    batch = batch[:sample_size]

            gen = generator(batch)
            assert (gen >= 0).all() and (gen <= 1).all()
            folder.process_and_save(gen)

            current_samples += sample_size
            if current_samples >= samples_per_gpu:
                break
    
    # log fid
    if distributed:
        torch.distributed.barrier()
    # log samples
    if rank == 0:
        first_samples = folder.get_first_n_images_01(64)
        for i in range(64):
            print(first_samples[i].shape)
            print(first_samples[i].mean(), first_samples[i].std())
        logger.log_image(f"{log_folder_name}_viz/{log_prefix}", first_samples)

    
    print("Generating samples time:", time.time() - start_time)

    if rank == 0:
        metrics = metrics_dataset(folder.path, dataset, eval_prc_recall=eval_prc_recall)
        fid = metrics["frechet_inception_distance"]
        isc_mean = metrics["inception_score_mean"]
        isc_std = metrics["inception_score_std"]
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)
        print(f"Found {log_prefix} FID:", fid)
        print(f"Found {log_prefix} Inception Score (mean±std): {isc_mean:.4f} ± {isc_std:.4f}")
        log_payload = {
            f"{log_folder_name}/{log_prefix}_fid{total_samples}": fid,
            f"fid_time/{log_folder_name}/{log_prefix}_{total_samples}": time.time() - start_time,
            f"{log_folder_name}/{log_prefix}_isc{total_samples}": float(isc_mean),
            # f"{log_folder_name}/{log_prefix}_isc_std": float(isc_std),        
            f"{log_folder_name}/{log_prefix}_precision": float(precision),
            f"{log_folder_name}/{log_prefix}_recall": float(recall),
        }
        
        logger.log_dict(log_payload)
        result_dict = {
            "fid": fid,
            "isc_mean": float(isc_mean),
            "isc_std": float(isc_std),
            "precision": float(precision),
            "recall": float(recall),
        }
    else:
        result_dict = dict()

    if eval_clip and "imagenet256" in dataset:
        clip_fid = fid_clip(folder.path)
        if rank == 0:
            logger.log_dict({f"{log_folder_name}/{log_prefix}_clip_fid{total_samples}": clip_fid})

    if distributed:
        result_dict = broadcast_object(result_dict)
    context.resume()
    if distributed:
        torch.distributed.barrier()
    return result_dict

from utils.distributed_utils import init_distributed_mode
if __name__ == "__main__":
    # prepare_val_prec_recall()

    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dist_url = "env://"
    init_distributed_mode(args)
    folder = "/checkpoint/flows/mingyangd/nn_flow/runs/L_resnetXL100_nocls_ev4_20251020_031402/samples/6f72ad417ae82940e9981c82c636f11f/EMA_50000"

    # print(fid_clip(folder))
    # print(metrics_dataset(folder, "imagenet256", eval_prc_recall=True))
    print(calculate_metrics(
                input1=prc_recall_path,
                input2=folder,
                cuda=True,
                isc=False,
                fid=False,
                prc=True,
                verbose=False,
            ))
#torchrun --nproc_per_node=8 -m utils.fid