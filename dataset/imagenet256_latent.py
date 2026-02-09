import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
# VAE is no longer used in the dataset, so we can remove this import
# from diffusers.models import AutoencoderKL 
import numpy as np


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


class ImageNet256LatentDataset(Dataset):
    def __init__(self, data_path, split='train', input_size=256):
        self.data_path = data_path
        self.split = split
        self.input_size = input_size

        # Train uses stronger augmentation; Val keeps deterministic center-crop
        if self.split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
            )
        else:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        print("Path: ", os.path.join(self.data_path, self.split))
        self.image_folder = datasets.ImageFolder(os.path.join(self.data_path, self.split))

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, index):
        while True:
            path, target = self.image_folder.samples[index]
            try:
                sample = self.image_folder.loader(path)
                break
            except Exception:
                # if image is corrupt, load another one
                index = (index + 1) % len(self)

        image_tensor = self.transform(sample)
        return image_tensor, target
