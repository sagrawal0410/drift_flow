import wandb

import torch.distributed as dist

import torch

import os

from pathlib import Path

import copy
import tempfile
import shutil
import atexit
import datetime
import builtins
import time
import io

import wandb
import threading
import sys


class Watchdog:
    def __init__(self, timeout_seconds=300):
        self.timeout_seconds = timeout_seconds
        self.timer = None

    def _timeout_handler(self):
        print(f"Watchdog timeout! init_distributed_mode did not complete in {self.timeout_seconds} seconds. Terminating.", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    def start(self):
        print(f"Starting watchdog for init_distributed_mode with {self.timeout_seconds}s timeout.")
        self.timer = threading.Timer(self.timeout_seconds, self._timeout_handler)
        self.timer.start()

    def stop(self):
        if self.timer and self.timer.is_alive():
            self.timer.cancel()
            print("Watchdog stopped, init_distributed_mode completed in time.")


def get_wandb_id():
    obj = wandb.run.id if is_main_process() else None
    if get_world_size() == 1:
        return obj
    obj = broadcast_object(obj)
    return obj


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ["LOCAL_RANK"])

def is_main_process():
    return get_rank() == 0


def is_rank_zero():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ["LOCAL_RANK"])


def init_distributed_mode(args):
    '''
    This function initializes the distributed mode.
    Requires:
        args.dist_url: the url of the distributed training
    Will add:
        args.rank: the rank of the process
        args.world_size: the number of processes
        args.gpu: the gpu of the process
    '''
    # 3. 注册清理钩子
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ or 'SLURM_PROCID' in os.environ
    if not is_distributed:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)
        args.distributed = False
        args.gpu=0
        return

    dist_timeout = getattr(args, 'dist_timeout', 2000)
    watchdog = Watchdog(timeout_seconds=dist_timeout)
    watchdog.start()
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = "env://"
    elif 'SLURM_PROCID' in os.environ:

        print("Using SLURM_PROCID")
        if getattr(args, 'rank', None) is None:
            args.rank = int(os.environ['SLURM_PROCID'])
        if getattr(args, 'world_size', None) is None:
            args.world_size = int(os.environ.get('SLURM_NTASKS', 1))
        if getattr(args, 'gpu', None) is None:
            local_rank = int(os.environ.get('SLURM_LOCALID', args.rank % torch.cuda.device_count()))
            args.gpu = local_rank
        os.environ["LOCAL_RANK"] = str(args.gpu)
    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(seconds=dist_timeout)
    )
    torch.distributed.barrier()    
    print("Distributed initialized", flush=True)

    setup_for_distributed(args.rank == 0)

    watchdog.stop()


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == float("inf"):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or name in skip_list
            or "diffloss" in name
        ):
            no_decay.append(param)  # no weight decay on bias, norm and diffloss
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x    


def broadcast_object(obj):
    """
    Broadcast a Python object from rank 0 to all processes more efficiently.
    Uses cuda tensors when available and reduces memory copies.
    """
    if dist.get_rank() == 0:
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = buffer.getvalue()
        length_tensor = torch.LongTensor([len(data)]).cuda()
        data_tensor = torch.frombuffer(data, dtype=torch.uint8).cuda()
    else:
        length_tensor = torch.LongTensor([0]).cuda()

    dist.broadcast(length_tensor, src=0)
    print(f"Rank {dist.get_rank()} broadcasting length tensor {length_tensor}")

    if dist.get_rank() != 0:
        data_tensor = torch.empty(length_tensor.item(), dtype=torch.uint8).cuda()

    dist.broadcast(data_tensor, src=0)

    if dist.get_rank() != 0:
        buffer = io.BytesIO(data_tensor.cpu().numpy().tobytes())
        obj = torch.load(buffer, weights_only=False)

    return obj

def gather_object(obj):
    import io, torch, torch.distributed as dist

    rank       = dist.get_rank()
    world_size = dist.get_world_size()

    # serialize to bytes
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    data = buffer.getvalue()

    # 1) gather sizes (on CPU)
    len_tensor       = torch.tensor([len(data)], dtype=torch.long)
    size_list        = [torch.zeros(1, dtype=torch.long) for _ in range(world_size)] if rank == 0 else None
    dist.gather(len_tensor, gather_list=size_list, dst=0)

    if rank == 0:
        sizes   = [int(x.item()) for x in size_list]
        max_size = max(sizes)
    else:
        max_size = None
    # broadcast max_size to all ranks
    max_size = torch.tensor([max_size or 0], dtype=torch.long)
    dist.broadcast(max_size, src=0)
    max_size = int(max_size.item())
    data_tensor = torch.zeros(max_size, dtype=torch.uint8)
    data_tensor[:len(data)] = torch.tensor(list(data), dtype=torch.uint8)
    gather_list = [torch.empty(max_size, dtype=torch.uint8) for _ in range(world_size)] if rank == 0 else None
    dist.gather(data_tensor, gather_list=gather_list, dst=0)

    if rank == 0:
        objects = []
        for sz, buf_tensor in zip(sizes, gather_list):
            buf = io.BytesIO(bytes(buf_tensor[:sz].tolist()))
            objects.append(torch.load(buf, weights_only=False))
        return objects
    else:
        return None

def sum_info_across_gpus(info, device):
    if get_world_size() > 1:
        keys = list(info.keys())
        values = []
        for key in keys:
            value = info[key]
            values.append(torch.tensor(float(value), device=device))
        all_values = torch.stack(values)
        dist.all_reduce(all_values, op=dist.ReduceOp.SUM)
        aggregated_info = {}
        for i, key in enumerate(keys):
            aggregated_info[key] = all_values[i].item()
        return aggregated_info
    else:
        return info
def aggregate_info_across_gpus(info, device):
    """Aggregate info dictionary across all GPUs by taking the mean."""
    x = sum_info_across_gpus(info, device)
    if get_world_size() > 1:
        x = {k: v / get_world_size() for k, v in x.items()}
    return x