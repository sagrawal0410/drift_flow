import torch
import math
from torch.optim.lr_scheduler import _LRScheduler
from typing import Union
from pathlib import Path
from torch import Tensor
import torch.nn as nn
from typing import Any
import random
import numpy as np
from utils.global_path import compile_mode

def custom_compile(**args):
    def wrapper(function):
        if compile_mode == "none":
            return function
        kwargs = {
            "mode": compile_mode,
        }
        kwargs.update(args)
        return torch.compile(function, **kwargs)
    return wrapper


class EvalWrapper(nn.Module):
    '''
    Wrapper for a model, to keep it frozen & always in eval mode.
    '''
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def train(self, mode=True):
        self.model.eval()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
        
def add_weight_decay(
    model,
    weight_decay=1e-5,
    skip_list=(),
    eps: Union[float, None] = None,
    lr: Union[float, None] = None,
):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)  # no weight decay on bias, norm and diffloss
        else:
            decay.append(param)
    result_list = [
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
        {"params": decay, "weight_decay": weight_decay, "lr": lr},
    ]

    if lr is not None:
        for param_group in result_list:
            param_group["lr"] = lr
    if eps is not None:
        for param_group in result_list:
            param_group["eps"] = eps
    return result_list


def sg(x):
    return x.clone().detach()


def make_folder(folder_path: Path):

    print(f"Current working directory: {Path.cwd()}")
    print("Making folder", folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)


class TemporalSeed:
    def __init__(self, seed: int):
        # Store original states
        self.original_states = {
            "torch_state": torch.get_rng_state(),
            "torch_cuda_state": (
                torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            ),
            "numpy_state": np.random.get_state(),
            "random_state": random.getstate(),
        }

        # Set new seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  # Only set for current device
        np.random.seed(seed)
        random.seed(seed)

    def resume(self):
        # Restore original states
        torch.set_rng_state(self.original_states["torch_state"])
        if (
            torch.cuda.is_available()
            and self.original_states["torch_cuda_state"] is not None
        ):
            torch.cuda.set_rng_state(self.original_states["torch_cuda_state"])
        np.random.set_state(self.original_states["numpy_state"])
        random.setstate(self.original_states["random_state"])


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]



class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(
        self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5
    ):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1



def LinearWarmupCosineDecayLR(kimg, lr_schedule=EasyDict(
        lr=2e-4, 
        warmup_kimg=10000,
        total_kimg=200000,
    ),):
    if kimg <= lr_schedule.warmup_kimg:
        return lr_schedule.lr * kimg / lr_schedule.warmup_kimg
    else:
        progress = (kimg - lr_schedule.warmup_kimg) / (lr_schedule.total_kimg - lr_schedule.warmup_kimg)
        return lr_schedule.lr * 0.5 * (1 + math.cos(math.pi * progress))
    

def dict_to_easydict(d):
    if not isinstance(d, dict):
        return d
    return EasyDict({k: dict_to_easydict(v) for k, v in d.items()})

def easydict_to_dict(d):
    if not isinstance(d, EasyDict):
        return d
    return {k: easydict_to_dict(v) for k, v in d.items()}

class AdaWeighting(nn.Module):
    def __init__(self, requires_kl, lr=0.01, momentum=0.95):
        """
        Adaptive weighting using log-space updates for stability.

        Args:
            requires_kl: (B,)
            lr: learning rate for adapting log_weight
            momentum: EMA factor for smoothing KL observations
        """
        super().__init__()
        requires_kl = torch.tensor(requires_kl)
        self.requires_kl = requires_kl
        self.log_weight = nn.Parameter(torch.zeros(requires_kl.shape[0]), requires_grad=False)  # log space for multiplicative control
        self.ema_kl = None
        self.lr = lr
        self.momentum = momentum

    def update_weight(self, current_kl):
        '''
        Args:
            current_kl: (B,)
        '''
        current_kl = current_kl.to(self.requires_kl.device)
        if self.ema_kl is None:
            self.ema_kl = current_kl
        else:
            self.ema_kl = self.momentum * self.ema_kl + (1 - self.momentum) * current_kl
        error = self.ema_kl - self.requires_kl
        self.log_weight.data += self.lr * error / (self.requires_kl + 1e-6)

        # Clamp to avoid instability
        self.log_weight.data = self.log_weight.data.clamp(min=-10.0, max=10.0)

    def weight(self):
        return self.log_weight.exp()