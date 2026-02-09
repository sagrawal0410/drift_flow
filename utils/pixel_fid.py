import os
import numpy as np
import torch
from scipy.linalg import sqrtm
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from PIL import Image
from tqdm import tqdm
from utils.global_path import dataset_download_path

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """PyTorch implementation of the Frechet Distance."""
    diff = mu1 - mu2

    # We use SciPy's sqrtm function for the matrix square root, as torch.linalg.sqrtm
    # is only for symmetric positive semi-definite matrices, and sigma1 @ sigma2 is not
    # guaranteed to be symmetric.
    # We convert to numpy for this operation.
    sigma1_np = sigma1.cpu().numpy()
    sigma2_np = sigma2.cpu().numpy()
    covmean, _ = sqrtm(sigma1_np @ sigma2_np, disp=False)

    if not np.isfinite(covmean).all():
        msg = f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates"
        print(msg)
        offset = np.eye(sigma1_np.shape[0]) * eps
        covmean = sqrtm((sigma1_np + offset) @ (sigma2_np + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    covmean = torch.from_numpy(covmean).to(mu1.device)

    tr_covmean = torch.trace(covmean)

    return (diff @ diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean

_mnist_stats_cache = {}
def get_mnist_pixel_stats(data_dir=dataset_download_path, device='cpu'):
    cache_key = f"stats_torch_{device}"
    if cache_key in _mnist_stats_cache:
        return _mnist_stats_cache[cache_key]
    
    dataset = MNIST(root=data_dir, train=True, download=True, transform=ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, num_workers=4)
    
    images = [batch[0].flatten(start_dim=1) for batch in tqdm(dataloader, desc="Calculating MNIST pixel stats (torch)")]
    images = torch.cat(images, dim=0).to(device)
    
    mu = torch.mean(images, dim=0)
    sigma = torch.cov(images.T)
    
    _mnist_stats_cache[cache_key] = (mu, sigma)
    return mu, sigma

def get_gen_pixel_stats(gen_dir, device='cpu'):
    files = [os.path.join(gen_dir, f) for f in os.listdir(gen_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    images_list = []
    for f in files:
        img = Image.open(f).convert('L')
        img_tensor = torch.tensor(np.array(img), dtype=torch.float32).flatten()
        images_list.append(img_tensor)
    
    if not images_list:
        raise ValueError(f"No images found in {gen_dir}")
        
    images = torch.stack(images_list).to(device)
    images = images / 255.0

    mu = torch.mean(images, dim=0)
    sigma = torch.cov(images.T)
    return mu, sigma

def mnist_pixel_fid(gen_dir, device='cpu'):
    mu_real, sigma_real = get_mnist_pixel_stats(device=device)
    mu_gen, sigma_gen = get_gen_pixel_stats(gen_dir, device=device)
    return calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen) 

if __name__ == "__main__":
    from utils.fid import fid_dataset
    from utils.fid import ImageFolder
    folder = ImageFolder(path="/private/home/mingyangd/dmy/test_mnist", world_size=1, global_rank=0, mode="clean")
    # folder.process_and_save(torch.randn(1000, 1, 28, 28).clamp(0, 1))
    # save 1000 images from mnist
    dataset = MNIST(root=dataset_download_path, train=True, download=True, transform=ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, num_workers=8)
    for i, (images, labels) in enumerate(dataloader):
        folder.process_and_save(images)
        if i > 10:
            break
    print(fid_dataset(folder.path, dataset="mnist"))