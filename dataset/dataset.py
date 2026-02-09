# %%
import numpy as np
import torch
import os

from torchvision.datasets import MNIST,  ImageFolder, CIFAR10
from torchvision import transforms
from utils.global_path import imagenet_root, imagenet_cache_root, dataset_download_path, imagenet_tar_path, tmp_folder, imagenet_cache_tar_path
from .cache_dataset import CachedFolder
from .imagenet256_latent import ImageNet256LatentDataset
from utils.distributed_utils import init_distributed_mode, get_world_size
import datetime

dataset_root = None
def set_dataset_root(path):
    global dataset_root
    if dataset_root is None:
        dataset_root = path
    else:
        raise ValueError("dataset_root is already set")

import os
import shutil
import time
import tarfile
import sys
import subprocess # New import
from pathlib import Path
from utils.distributed_utils import get_local_rank

def copy_and_untar(source_tar_path: str, extract_dir: str = "/tmp", temp_file_path: str = "/tmp/tmp.tar", N_strips=0):
    """
    Copies a .tar file to a destination directory (default /tmp)
    and then extracts its contents into that same directory.

    **MODIFIED VERSION:**
    1. Only runs on LOCAL_RANK 0 (assumes distributed environment).
    2. Uses system 'cp' and 'tar' commands via subprocess for max speed.

    Args:
        source_tar_path (str): The full path to the source .tar file.
        extract_dir (str): The directory to copy the file to and
                               extract its contents into. Defaults to "/tmp".
    """
    local_rank = get_local_rank()
    if int(local_rank) == 0:
        # if not exist
        print("copying and untarring")
        time_start = time.time()
        if not os.path.exists(temp_file_path):
            subprocess.run(
                ["cp", source_tar_path, temp_file_path], 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
        print(f"Copied, takes {time.time() - time_start} seconds")
        time_start = time.time()
        # if extract_dir not exist
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir, exist_ok=True)
            
            # 在这里添加 --strip-components=N
            subprocess.run(
                ["tar", 
                 "-xf", 
                 temp_file_path, 
                 "-C", 
                 extract_dir, 
                 f"--strip-components={N_strips}"],  # <--- 添加这一行
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
        print(f"Untarred, takes {time.time() - time_start} seconds")
    
    if get_world_size() > 1:
        torch.distributed.barrier()

class MNISTDataset:
    def __init__(self):
        # Load MNIST dataset

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        )

        self.dataset_train = MNIST(
            root=dataset_download_path, train=True, download=True, transform=transform
        )
        self.dataset_val = MNIST(
            root=dataset_download_path, train=False, download=True, transform=transform
        )

    def sample(self, batch_size, is_train=True, return_labels=False):
        """Sample a batch of images randomly from the dataset; values in [-1, 1]"""
        """
        Args:
            batch_size: the number of images to sample
            is_train: whether to sample from the training dataset
        Returns:
            batch: [batch_size, 1, 28, 28], range in [-1, 1]
            (optional) labels: [batch_size], range in [0, 9]
        """
        dataset = self.dataset_train if is_train else self.dataset_val
        indices = torch.randint(0, len(dataset), (batch_size,))
        batch = torch.stack(
            [dataset[i][0] for i in indices]
        )  # Get just the images, not labels
        if return_labels:
            labels = torch.tensor([dataset[i][1] for i in indices])
            return batch, labels
        else:
            return batch

    def __len__(self, is_train=True):
        return len(self.dataset_train) if is_train else len(self.dataset_val)


class CIFAR10Dataset:
    def __init__(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.dataset_train = CIFAR10(
            root=dataset_download_path, train=True, download=True, transform=transform
        )
        self.dataset_val = CIFAR10(
            root=dataset_download_path, train=False, download=True, transform=transform
        )

    def sample(self, batch_size, is_train=True, return_labels=False):
        """Sample a batch of images randomly from the dataset; values in [-1, 1]"""
        """
        Args:
            batch_size: the number of images to sample
            is_train: whether to sample from the training dataset
        Returns:
            batch: [batch_size, 3, 32, 32], range in [-1, 1]
            (optional) labels: [batch_size], range in [0, 9]
        """
        dataset = self.dataset_train if is_train else self.dataset_val
        indices = torch.randint(0, len(dataset), (batch_size,))
        batch = torch.stack(
            [dataset[i][0] for i in indices]
        )  # Get just the images, not labels
        if return_labels:
            labels = torch.tensor([dataset[i][1] for i in indices])
            return batch, labels
        else:
            return batch

    def __len__(self, is_train=True):
        return len(self.dataset_train) if is_train else len(self.dataset_val)



class ImageNetDataset:
    def __init__(self, path, img_size=32):

        transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        )
        self.dataset_train = ImageFolder(
            root = os.path.join(path, "train"),
            transform=transform
        )
        self.dataset_val = ImageFolder(
            root = os.path.join(path, "val"),
            transform=transform
        )

    def sample(self, batch_size, is_train=True, return_labels=False):
        """Sample a batch of images randomly from the dataset; values in [-1, 1]"""
        """
        Args:
            batch_size: the number of images to sample
            is_train: whether to sample from the training dataset
        Returns:
            batch: [batch_size, 3, 32, 32], range in [-1, 1]
            (optional) labels: [batch_size], range in [0, 9]
        """
        dataset = self.dataset_train if is_train else self.dataset_val
        indices = torch.randint(0, len(dataset), (batch_size,))
        batch = torch.stack(
            [dataset[i][0] for i in indices]
        )  # Get just the images, not labels
        if return_labels:
            labels = torch.tensor([dataset[i][1] for i in indices])
            return batch, labels
        else:
            return batch

    def __len__(self, is_train=True):
        return len(self.dataset_train) if is_train else len(self.dataset_val)

from PIL import Image

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def postprocess(x, label_list=None):
    return (x * 0.5 + 0.5).clamp(0, 1)


def hist_diff(x, y):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()
    x = x.flatten()
    y = y.flatten()
    bins = np.linspace(min(x.min(), y.min()), max(x.max(), y.max()), 501)
    x_hist, _ = np.histogram(x, bins=bins, density=True)
    y_hist, _ = np.histogram(y, bins=bins, density=True)
    x_hist = x_hist / x_hist.sum()
    y_hist = y_hist / y_hist.sum()
    x_cdf = np.cumsum(x_hist)
    y_cdf = np.cumsum(y_hist)
    emd = np.abs(x_cdf - y_cdf).mean()
    return emd


from torch.utils.data import Dataset

def tar_imagenet(tar_local=False):
    from utils.global_path import strip_imagenet_tar
    if tar_local:
        copy_and_untar(os.path.join(imagenet_tar_path), os.path.join(tmp_folder, "imagenet"), os.path.join(tmp_folder, "imagenet.tar"), N_strips=strip_imagenet_tar)
        return os.path.join(tmp_folder, "imagenet")
    else:
        return imagenet_root

def tar_imagenet_cache(tar_local=False, split="train"):
    from utils.global_path import strip_imagenet_cache_tar
    if tar_local:
        tar_path = imagenet_cache_tar_path[split]
        copy_and_untar(tar_path, os.path.join(tmp_folder, f"imagenet_{split}_cache"), os.path.join(tmp_folder, f"imagenet_{split}_cache.tar"), N_strips=strip_imagenet_cache_tar)
        return os.path.join(tmp_folder, f"imagenet_{split}_cache")
    else:
        return imagenet_cache_root[split]


def get_dataset(dataset_name, split="train", tar_local=False):
    """
    Args:
        dataset_name: the name of the dataset; "mnist" or "cifar10" or "imagenet32"
        split: the split to get; "train" or "val"
    Returns:
        dataset: the dataset; consists of (image, label) pairs; every image is within the range [-1, 1]
    """
    assert dataset_name in ["mnist", "cifar10", "imagenet32", "imagenet256", "imagenet64", "imagenet128", "imagenet256_cache", "imagenet256_latent"]
    assert split in ["train", "val"]
    if dataset_name == "mnist":
        return (
            MNISTDataset().dataset_train
            if split == "train"
            else MNISTDataset().dataset_val
        )
    elif dataset_name == "cifar10":
        return (
            CIFAR10Dataset().dataset_train
            if split == "train"
            else CIFAR10Dataset().dataset_val
        )
    elif dataset_name in ["imagenet32", "imagenet256", "imagenet64", "imagenet128"]:
        resolution = int(dataset_name.split("imagenet")[1])
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, resolution)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        )
        assert split in ["train", "val"]
        folder = tar_imagenet(tar_local)
        return ImageFolder(
            root = os.path.join(folder, split),
            transform=transform
        )
    elif dataset_name == "imagenet256_cache":
        assert split in ["train", "val"]
        folder = tar_imagenet_cache(tar_local, split)
        return CachedFolder(folder)
    elif dataset_name == "imagenet256_latent":
        assert split in ["train", "val"]
        folder = tar_imagenet(tar_local)
        return ImageNet256LatentDataset(data_path=folder, split=split)


if __name__ == "__main__":
    # init_distributed_mode()
    get_dataset("imagenet256", "val", tar_local=True)
    # get_dataset("imagenet256_cache", "val", tar_local=True)
    # dataset = get_dataset("cifar10", "train")
    # print(len(dataset))
    # print(dataset[0])
# python -m dataset.dataset