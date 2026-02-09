# logging_utils.py
# If you need builtin generics on Python <3.9, uncomment the next line:
# from __future__ import annotations

import os
import io
import json
import math
import hashlib
import warnings
import atexit
from pathlib import Path
from typing import Union, Any, Sequence, Optional, Dict

import numpy as np
import torch
import torch.distributed as dist
import einops
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image
import wandb
from omegaconf import DictConfig, OmegaConf

from utils.distributed_utils import is_rank_zero


# -------------------------
# Small helpers & plotting
# -------------------------

def stable_run_id(cfg_str: str) -> str:
    """Deterministic 32-char id for a config string (W&B run id)."""
    return hashlib.md5(cfg_str.encode()).hexdigest()


def plt_to_image(dpi: int = 120) -> Image.Image:
    """Render current matplotlib figure to a detached PIL image."""
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()  # copy() to detach from ephemeral buffer


def plot_scatter(
    x: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Image.Image:
    """Scatter plot -> PIL image. Smaller/fewer points get larger alpha/size."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    if labels is None:
        plt.scatter(x, y)
        plt.tight_layout()
        return plt_to_image()

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    unique_labels, label_counts = np.unique(labels, return_counts=True)
    order = np.argsort(label_counts)[::-1]  # largest first; we’ll draw them first
    sorted_labels = unique_labels[order]
    sorted_counts = label_counts[order]

    max_size, min_size = 100.0, 20.0
    max_alpha, min_alpha = 0.9, 0.4

    for tp, count in zip(sorted_labels, sorted_counts):
        if isinstance(tp, str):
            positions = np.array([a == tp for a in labels])
        else:
            positions = np.array(labels) == tp

        size = max(min_size, max_size * (1.0 / max(count, 1)))
        alpha = min_alpha + (max_alpha - min_alpha) * (1.0 / max(count, 1))

        plt.scatter(x[positions], y[positions], label=str(tp), s=size, alpha=alpha)

    plt.legend()
    plt.tight_layout()
    return plt_to_image()


def plot_heatmap(
    data: Union[np.ndarray, torch.Tensor],
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    x_ticklabels: Optional[Sequence] = None,
    y_ticklabels: Optional[Sequence] = None,
) -> Image.Image:
    """Heatmap -> PIL image, with wrapped labels and safe tick relabeling."""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    if data.ndim == 1:
        data = data[None, :]  # (1, W)

    vmin = float(np.min(data))
    vmax = float(np.max(data))

    fig, ax = plt.subplots()
    cax = ax.imshow(data, cmap="hot", aspect="auto", vmin=vmin, vmax=vmax)

    if x_label is not None:
        ax.set_xlabel("\n".join(_wrap_text(x_label, width=40)))
    if y_label is not None:
        ax.set_ylabel("\n".join(_wrap_text(y_label, width=40)))

    x_locator = MaxNLocator(integer=True, prune="both")
    y_locator = MaxNLocator(integer=True, prune="both")
    ax.xaxis.set_major_locator(x_locator)
    ax.yaxis.set_major_locator(y_locator)

    x_ticks = ax.get_xticks().astype(int)
    y_ticks = ax.get_yticks().astype(int)

    if x_ticklabels is not None:
        xt_adj = [x_ticklabels[i] for i in x_ticks if 0 <= i < len(x_ticklabels)]
        ax.set_xticklabels(xt_adj, rotation=45, ha="right")

    if y_ticklabels is not None:
        yt_adj = [y_ticklabels[i] for i in y_ticks if 0 <= i < len(y_ticklabels)]
        ax.set_yticklabels(yt_adj)

    cbar = fig.colorbar(cax)
    ticks = np.linspace(vmin, vmax, 10)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

    fig.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.2)
    plt.tight_layout()
    return plt_to_image()


def plot_plots(
    list_x_coords: Union[np.ndarray, torch.Tensor, Sequence],
    list_y_coords: Union[np.ndarray, torch.Tensor, Sequence],
    list_labels: Union[Sequence[str], str],
) -> Image.Image:
    """Multiple line plots -> PIL image."""
    if not isinstance(list_labels, list):
        list_labels = [list_labels]

    if isinstance(list_x_coords, torch.Tensor):
        list_x_coords = list_x_coords.detach().cpu().numpy()
    if isinstance(list_y_coords, torch.Tensor):
        list_y_coords = list_y_coords.detach().cpu().numpy()

    list_x_coords = np.array(list_x_coords, dtype=object)
    list_y_coords = np.array(list_y_coords, dtype=object)

    N = len(list_labels)
    if list_x_coords.ndim == 1:
        list_x_coords = np.array([list_x_coords] * N, dtype=object)
    if list_y_coords.ndim == 1:
        list_y_coords = np.array([list_y_coords] * N, dtype=object)

    for i in range(N):
        plt.plot(np.array(list_x_coords[i], dtype=float),
                 np.array(list_y_coords[i], dtype=float),
                 label=str(list_labels[i]))
    plt.legend()
    plt.tight_layout()
    return plt_to_image()


def _wrap_text(s: str, width: int = 40) -> Sequence[str]:
    """Simple word wrap without importing textwrap in many places."""
    import textwrap as _tw
    return _tw.wrap(s, width=width)


# -------------------------
# W&B logger with offline
# -------------------------

class WandbLogger:
    """
    A thin W&B wrapper that:
      * logs only on rank-0,
      * supports offline mode,
      * buffers scalar dicts across steps/ranks and flushes them periodically via set_step,
      * dumps local JSON (averaged) and images in offline mode.

    Offline files layout:
      <offline_dir>/
        images/<folder>/<entry>_steps{step}.jpg
        json/<folder>.json            # JSONL; each line is {"step": step, <entry>: avg, ...}
    """
    def __init__(self) -> None:
        self.step: int = 0
        self.log_every_k: int = 1
        self.can_log: bool = True
        self.online: bool = True
        self.last_dump: Optional[int] = None

        # offline dump state
        self.offline_dir = Path("wandb_offline")
        # key -> {"sum": float, "count": int}
        self._dict_buffer: Dict[str, Dict[str, float | int]] = {}
        atexit.register(self.finish)

    # ---------- basic controls ----------

    def set_can_log(self, can_log: bool) -> None:
        self.can_log = can_log

    def set_step(self, step: int) -> None:
        """
        Update internal step and (only here) decide whether to flush buffered dicts.
        We flush when the last dump is "far enough" (>= log_every_k steps ago).
        """
        self.step = int(step)
        if self.last_dump is None:
            # First time we get a step, prime the pump but do not force flush
            self.last_dump = self.step
            return

        if (self.step - self.last_dump) >= self.log_every_k:
            # Flush (online/offline) and record last_dump
            self.flush_dicts(self.step)
            self.last_dump = self.step

    def is_rank_zero(self) -> bool:
        return is_rank_zero()

    # ---------- setup ----------

    def setup_wandb(
        self,
        project: str,
        config: Optional[Union[dict, DictConfig]] = None,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        online: bool = True,
        offline_dir: str = "wandb_offline",
        log_every_k: int = 1,
    ) -> None:
        """
        Setup the logger.
        """
        self.log_every_k = int(log_every_k)
        print(f"[WandbLogger] log_every_k: {self.log_every_k}")
        self.online = bool(online)

        if not self.is_rank_zero():
            # still set offline_dir for consistency
            if not self.online:
                self.offline_dir = Path(offline_dir)
            return

        if config is not None and isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)  # type: ignore[assignment]

        # prepare offline directory if needed
        if not self.online:
            self.offline_dir = Path(offline_dir)
            (self.offline_dir / "images").mkdir(parents=True, exist_ok=True)
            (self.offline_dir / "json").mkdir(parents=True, exist_ok=True)

        extra_kwargs = {}
        extra_kwargs["id"] = stable_run_id(name)
        extra_kwargs["resume"] = "allow"

        init_kwargs: Dict[str, Any] = dict(
            project=project,
            config=config,
            mode="online" if self.online else "offline",
            **extra_kwargs,
        )
        # Optional fields
        if entity:
            init_kwargs["entity"] = entity
        if name:
            init_kwargs["name"] = name

        wandb.init(**init_kwargs)

    # ---------- logging scalars/dicts ----------
    def log_table(self, elts: dict, step: Optional[int] = None) -> None:
        """
        Log a table to W&B.
        """
        if not self.is_rank_zero() or not self.can_log:
            return
        wandb.log(elts, step=self._resolve_step(step))

    def log_dict(self, elts: dict, step: Optional[int] = None) -> None:
        """
        Buffer a dictionary for later aggregation/logging.
        Flushing happens ONLY in `set_step` via `flush_dicts`.
        """
        if not self.can_log:
            return

        # Allow callers to override the internal step for this buffer add,
        # but actual flush cadence is still controlled by set_step().
        current_step = int(step if step is not None else self.step)
        _ = current_step  # step isn't used directly here; flushing is time-based via set_step

        # accumulate sums/counts in buffer
        for k, v in elts.items():
            if isinstance(v, torch.Tensor):
                val = v.detach()
            else:
                val = float(v)

            if k not in self._dict_buffer:
                self._dict_buffer[k] = {"sum": 0.0, "count": 0}

            self._dict_buffer[k]["sum"] = val + self._dict_buffer[k]["sum"]  # type: ignore
            self._dict_buffer[k]["count"] = 1 + self._dict_buffer[k]["count"]  # type: ignore

    def log_dict_dir(self, elts: dict, dir_name: str, step: Optional[int] = None) -> None:
        """Prefix every key with 'dir_name/' before buffering."""
        new_dict = {f"{dir_name}/{k}": v for k, v in elts.items()}
        self.log_dict(new_dict, step=step)

    # ---------- logging images ----------

    def log_single_image(self, key: str, image: Image.Image, step: Optional[int] = None) -> None:
        if not self.is_rank_zero() or not self.can_log:
            return
        if not self.online:
            self._save_offline_image(key, [image], step)
        wandb.log({key: wandb.Image(image)}, step=self._resolve_step(step))

    def log_scatter(
        self,
        key: str,
        x: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        step: Optional[int] = None,
    ) -> None:
        if not self.is_rank_zero() or not self.can_log:
            return
        image = plot_scatter(x, y, labels)
        self.log_single_image(key, image, step=step)

    def log_image(
        self,
        key: str,
        image: Union[np.ndarray, torch.Tensor, Image.Image, Sequence[Image.Image]],
        mean: Optional[Union[np.ndarray, torch.Tensor, Sequence, float]] = None,
        std: Optional[Union[np.ndarray, torch.Tensor, Sequence, float]] = None,
        step: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Log image(s) to W&B. Also dumps locally in offline mode:
          images/<folder>/<entry>_steps{step}.jpg  (grid with 8 rows per column).
        """
        if not self.is_rank_zero() or not self.can_log:
            return

        # Single PIL image
        if isinstance(image, Image.Image):
            if not self.online:
                self._save_offline_image(key, [image], step)
            wandb.log({key: wandb.Image(image)}, step=self._resolve_step(step))
            return

        # List/Tuple of PIL images
        if isinstance(image, (list, tuple)) and len(image) and isinstance(image[0], Image.Image):
            pil_imgs = list(image)
            if not self.online:
                self._save_offline_image(key, pil_imgs, step)
            # Log per-image list
            wandb.log({key: [wandb.Image(img) for img in pil_imgs]}, step=self._resolve_step(step))
            # Also log a grid image under <key>_grid
            grid_img = self._make_grid_image(pil_imgs)
            wandb.log({f"{key}_grid": wandb.Image(grid_img)}, step=self._resolve_step(step))
            return

        # Otherwise: tensor/ndarray or list of arrays -> normalize to (B,H,W,C) uint8
        if isinstance(image, list):
            # list of arrays/tensors
            tensors = [torch.as_tensor(img).detach().cpu() for img in image]
            image = torch.stack(tensors, dim=0)

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        arr = np.asarray(image)

        # If (H,W,C) or (C,H,W), add batch dim
        if arr.ndim == 3:
            arr = arr[None, ...]

        # If channel-first, move to channel-last
        if arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
            # (B,C,H,W) -> (B,H,W,C)
            arr = einops.rearrange(arr, "b c h w -> b h w c")

        # If grayscale, duplicate channels to 3
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        # Unnormalize
        if std is not None:
            std_vec = _to_3(std)
            arr = arr * np.array(std_vec, dtype=float).reshape(1, 1, 1, 3)
        if mean is not None:
            mean_vec = _to_3(mean)
            arr = arr + np.array(mean_vec, dtype=float).reshape(1, 1, 1, 3)

        # Convert to uint8
        if arr.dtype != np.uint8:
            arr = np.clip(arr, a_min=0.0, a_max=1.0) * 255.0
            arr = arr.astype(np.uint8)

        # Local offline dump: make a grid
        pil_imgs = [Image.fromarray(img) for img in arr]
        if not self.online:
            self._save_offline_image(key, pil_imgs, step)

        # Log to W&B
        wandb_imgs = [wandb.Image(img) for img in pil_imgs]
        wandb.log({key: wandb_imgs}, step=self._resolve_step(step))
        # Also log a grid image under <key>_grid
        grid_img = self._make_grid_image(pil_imgs)
        wandb.log({f"{key}_grid": wandb.Image(grid_img)}, step=self._resolve_step(step))

    def _save_offline_image(
        self,
        key: str,
        images: Sequence[Image.Image],
        step: Optional[int] = None,
    ) -> None:
        """Save an image grid to <offline_dir>/images/<folder>/<entry>_steps{step}.jpg."""
        parts = str(key).split("/", 1)
        folder = parts[0] if len(parts) == 2 else "images"
        entry = parts[1] if len(parts) == 2 else parts[0]

        out_dir = self.offline_dir / "images" / folder
        out_dir.mkdir(parents=True, exist_ok=True)

        curr_step = self._resolve_step(step)
        if len(images) == 0:
            return
        grid = self._make_grid_image(images)

        out_path = out_dir / f"{entry}_steps{curr_step}.jpg"
        grid.save(out_path, format="JPEG")

    def _make_grid_image(self, images: Sequence[Image.Image], rows: int = 8) -> Image.Image:
        """Create a fixed-rows grid image from a sequence of PIL images.
        Pads with black tiles to fill the last column. Normalizes all tiles to the
        first image's size.
        """
        num = len(images)
        rows = max(1, int(rows))
        cols = max(1, int(math.ceil(num / rows)))

        # Normalize all images to first one's size
        w, h = images[0].size
        norm_imgs = [img.resize((w, h)) for img in images]
        total = rows * cols
        if len(norm_imgs) < total:
            blank = Image.new("RGB", (w, h), color=(0, 0, 0))
            norm_imgs += [blank] * (total - len(norm_imgs))

        grid = Image.new("RGB", (cols * w, rows * h))
        for idx, img in enumerate(norm_imgs):
            r = idx % rows
            c = idx // rows
            grid.paste(img, (c * w, r * h))
        return grid

    # ---------- logging heatmaps ----------

    def log_heatmap(
        self,
        key: str,
        data: Union[np.ndarray, torch.Tensor],
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        x_ticklabels: Optional[Sequence] = None,
        y_ticklabels: Optional[Sequence] = None,
        step: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if not self.is_rank_zero() or not self.can_log:
            return
        image = plot_heatmap(data, x_label, y_label, x_ticklabels, y_ticklabels)
        self.log_single_image(key, image, step=step)

    # ---------- logging videos ----------

    def log_video(
        self,
        key: str,
        video: Union[np.ndarray, torch.Tensor],
        mean: Optional[Union[np.ndarray, torch.Tensor, Sequence, float]] = None,
        std: Optional[Union[np.ndarray, torch.Tensor, Sequence, float]] = None,
        fps: int = 12,
        format: str = "mp4",
        step: Optional[int] = None,
    ) -> None:
        """
        Log video to W&B. Accepts (T,C,H,W) or (B,T,C,H,W).
        Applies optional mean/std (scalar or 3-vector). Converts to uint8.
        """
        if not self.is_rank_zero() or not self.can_log:
            return

        if isinstance(video, torch.Tensor):
            video = video.detach().cpu().numpy()
        arr = np.array(video)

        # Use first in batch if provided
        if arr.ndim == 5:  # (B,T,C,H,W)
            arr = arr[0]

        # (T,C,H,W) -> (T,H,W,C)
        if arr.ndim == 4 and arr.shape[1] in (1, 3):
            arr = np.transpose(arr, (0, 2, 3, 1))

        # grayscale -> RGB
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        if std is not None:
            arr = arr * np.array(_to_3(std), dtype=float).reshape(1, 1, 1, 3)
        if mean is not None:
            arr = arr + np.array(_to_3(mean), dtype=float).reshape(1, 1, 1, 3)

        if arr.dtype != np.uint8:
            arr = np.clip(arr, a_min=0.0, a_max=1.0) * 255.0
            arr = arr.astype(np.uint8)

        wandb.log({key: wandb.Video(arr, fps=fps, format=format)}, step=self._resolve_step(step))

    # ---------- dict flushing (online & offline) ----------

    def flush_dicts(self, step: Optional[int] = None) -> None:
        """
        Aggregate buffered dicts across all ranks and:
          - if online: wandb.log averaged metrics at `step`
          - if offline: append JSONL records to <offline_dir>/json/<folder>.json
        Handles incomplete per-rank buffers by summing only present keys.
        Rank-0 performs logging/writing; all ranks clear local buffers.
        """
        curr_step = self._resolve_step(step)
        local_buf_cpu = {}
        for k, v in self._dict_buffer.items():
            s = float(v["sum"]) 
            c = float(v["count"])
            local_buf_cpu[k] = {"sum": s, "count": c}
        
        gathered = [None]  # type: ignore[var-annotated]
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered = [None for _ in range(world_size)]  # type: ignore[list-item]
            try:
                dist.all_gather_object(gathered, local_buf_cpu)
            except Exception:
                # If all_gather_object fails for any reason, fall back to local-only
                gathered = [local_buf]
        else:
            gathered = [local_buf]

        # Merge: key-wise sum of sums/counts, but only from ranks that have the key
        merged: Dict[str, Dict[str, float]] = {}
        for rank_buf in gathered:
            if not isinstance(rank_buf, dict):
                continue
            for k, v in rank_buf.items():
                if k not in merged:
                    merged[k] = {"sum": 0.0, "count": 0.0}
                merged[k]["sum"] = v['sum'] + merged[k]["sum"]
                merged[k]["count"] = v['count'] + merged[k]["count"]

        # Clear local buffer on every rank (we've consumed it)
        self._dict_buffer.clear()

        # If nothing to log after merge, return
        if not merged:
            return

        # Compute averages only for entries with positive count
        avg_elts = {k: v["sum"] / v["count"] for k, v in merged.items() if v["count"] > 0.0}
        if not avg_elts:
            return

        # Rank-0 performs the actual write/log
        if self.is_rank_zero():
            if self.online:
                # Online: single wandb.log with averaged metrics
                wandb.log(avg_elts, step=curr_step)
            else:
                # Offline: regroup by folder and append JSONL
                by_folder: Dict[str, Dict[str, float]] = {}
                for k, val in avg_elts.items():
                    parts = str(k).split("/", 1)
                    folder = parts[0] if len(parts) == 2 else "root"
                    entry = parts[1] if len(parts) == 2 else parts[0]
                    if folder not in by_folder:
                        by_folder[folder] = {}
                    by_folder[folder][entry] = float(val)

                for folder, data in by_folder.items():
                    rec = {"step": int(curr_step), **data}
                    json_path = self.offline_dir / "json" / f"{folder}.json"
                    json_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(json_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")

    # ---------- internals ----------

    def _resolve_step(self, step: Optional[int]) -> int:
        return int(step if step is not None else self.step)

    def finish(self) -> None:
        """Flush buffered metrics and finish W&B run."""
        try:
            self.flush_dicts(self.step)
        except Exception:
            pass
        try:
            if self.is_rank_zero():
                wandb.finish()
        except Exception:
            pass


# -------------------------
# Small utility
# -------------------------

def _to_3(v: Union[Sequence, float, int, torch.Tensor, np.ndarray]) -> Sequence[float]:
    """
    Convert (scalar | 1-elem | 3-elem) mean/std to a length-3 list[float].
    """
    if isinstance(v, (float, int)):
        return [float(v), float(v), float(v)]
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    v = list(np.asarray(v).flatten().tolist())
    if len(v) == 1:
        return [float(v[0])] * 3
    assert len(v) == 3, "mean/std must be scalar, length-1, or length-3"
    return [float(x) for x in v]
