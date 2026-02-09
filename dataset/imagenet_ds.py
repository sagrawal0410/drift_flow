from hae import eval_fid
from dataset.dataset import get_dataset
from utils.distributed_utils import get_rank, get_world_size, init_distributed_mode
import torch.nn.functional as F
from utils.logging_utils import WandbLogger

from utils.misc import EasyDict
if __name__ == "__main__":
    init_distributed_mode()
    rank = get_rank()
    world_size = get_world_size()
    logger = WandbLogger()
    logger.setup_wandb(config=EasyDict(wandb=EasyDict(project="imagenet-ds", entity="mit-hair", name="imagenet-ds")))
    def postprocess(x):
        return (x * 0.5 + 0.5).clamp(0, 1)
    
    def gen(x):
        imgs = x[0]
        imgs = F.interpolate(imgs, size=16, mode="bicubic")
        imgs = F.interpolate(imgs, size=32, mode="bicubic")
        imgs = postprocess(imgs)
        return imgs
    
    def identity(x):
        return postprocess(x[0])
    
    def bilinear(x):
        imgs = x[0]
        imgs = F.interpolate(imgs, size=16, mode="bilinear")
        imgs = F.interpolate(imgs, size=32, mode="bilinear")
        imgs = postprocess(imgs)
        return imgs

    def nearest(x):
        imgs = x[0]
        imgs = F.interpolate(imgs, size=16, mode="nearest")
        imgs = F.interpolate(imgs, size=32, mode="nearest")
        imgs = postprocess(imgs)
        return imgs
    
    dataset = get_dataset("imagenet", "train")
    eval_fid(gen, dataset, logger, dataset="imagenet32", log_prefix="bicubic")
    eval_fid(identity, dataset, logger, dataset="imagenet32", log_prefix="identity")
    eval_fid(bilinear, dataset, logger, dataset="imagenet32", log_prefix="bilinear")
    eval_fid(nearest, dataset, logger, dataset="imagenet32", log_prefix="nearest")
    
# bash run.sh -t imagenet_ds.py -m "imagenet"