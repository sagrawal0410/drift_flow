import torch
import torch.distributed as dist
from datetime import datetime
from pathlib import Path
import wandb
from utils.distributed_utils import is_main_process, save_on_master, broadcast_object
from utils.misc import make_folder
from typing import List
import os
import shutil
from utils.misc import EasyDict
from utils.global_path import ckpt_root
import math

"""

This files support the following functionality:

    save_ckpt(run_id, epoch, dicts): save the dicts for (run_id, epoch). 
    load_ckpt_epoch(run_id, epoch, wandb_info=EasyDict(project="", entity="")): 
        return the dicts for (run_id, epoch);
        if wandb_info not set: will use the project & entity for the current run. 

    load_last_ckpt(run_id, wandb_info=EasyDict(project="", entity="")): load (run_id, epoch) where epoch is the latest epoch. 
    ckpt_epoch_numbers(run_id, wandb_info=EasyDict(project="", entity="")): return the sorted list of epochs available in the run. 

Automatically, the models will be uploaded to wandb and downloaded from wandb (if not locally available).

"""

def rm_ckpt(run_id: str, epoch: int):
    """
    Remove the checkpoint for the given run_id and epoch.
    """
    ckpt_path = run_to_ckpt_path(run_id, epoch, ckpt_root)
    if ckpt_path.exists():
        os.remove(ckpt_path)
        print(f"Removed checkpoint {ckpt_path}")
    else:
        print(f"Checkpoint {ckpt_path} not found")



def used_folder(run_name: str, root: str) -> Path:
    return ckpt_folder(run_name, root) / "used"


def used_marker_path(run_name: str, epoch: int, root: str) -> Path:
    return used_folder(run_name, root) / f"epoch-{int(epoch)}"


def mark_as_used(run_id: str, epoch: int):
    """
    Create a marker file to indicate a checkpoint epoch is used and should not be pruned.
    The marker will be placed under <ckpt_root>/<run_id>/checkpoints/used/epoch-<epoch>
    """
    if not is_main_process():
        return
    marker = used_marker_path(run_id, epoch, ckpt_root)
    make_folder(marker.parent)
    marker.touch(exist_ok=True)


def get_used(run_id: str) -> List[int]:
    """
    Return sorted list of epochs marked as used for a run.
    """
    folder = used_folder(run_id, ckpt_root)
    if not folder.exists():
        return []
    used_epochs = []
    for path in folder.glob("epoch-*"):
        used_epochs.append(int(path.name.split("-")[-1]))
    return sorted(set(used_epochs))


def prune_old_ckpts(run_id: str, max_ckpts: float, persist_every: int):
    """
    Keep only the latest `max_ckpts` checkpoints plus any explicitly used ones.
    """
    if (isinstance(max_ckpts, float) and math.isinf(max_ckpts)):
        return
    k = int(max_ckpts)
    if k < 0:
        return
    all_epochs = ckpt_epoch_numbers(run_id)
    used_epochs = get_used(run_id)
    all_epochs = sorted(all_epochs)
    if not all_epochs:
        return
    if k == 0:
        # delete all except explicitly used or persistence cadence
        for epoch in all_epochs:
            if epoch not in used_epochs and epoch % persist_every != 0:
                rm_ckpt(run_id, epoch)
        return
    if k >= len(all_epochs):
        return  # nothing to prune
    keep_threshold = all_epochs[-k]
    for epoch in all_epochs:
        if epoch not in used_epochs and epoch < keep_threshold and epoch % persist_every != 0:
            rm_ckpt(run_id, epoch)


def all_runs() -> List[str]:
    """Return a sorted list of all run ids under the checkpoint root."""
    root_path = Path(ckpt_root)
    if not root_path.exists():
        return []
    run_ids = [p.name for p in root_path.iterdir() if p.is_dir()]
    return sorted(run_ids)

def save_ckpt(run_id: str, epoch: int, dicts: dict, max_ckpts: float = float("inf"), persist_every: int = 10**9):
    """
    Save the checkpoint for the given run_id and epoch.
    Args:
        run_id: the run id to save the checkpoint to (should be current run id).
        epoch: the epoch to save the checkpoint to.
        dicts: the dictionary to save.
        max_ckpts: keep at most this many most-recent checkpoints; always keep those marked as used. Default: infinite (no pruning).
    """
    if not is_main_process():
        return
    ckpt_path = run_to_ckpt_path(run_id, epoch, ckpt_root)
    save_ckpt_path(ckpt_path, dicts)
    prune_old_ckpts(run_id, max_ckpts, persist_every)


def ckpt_epoch_numbers(run_id: str, wandb_info=EasyDict(entity="", project="")) -> List[int]:
    """
    Return the sorted list of epochs available in the run.
    Args:
        run_id: the run id to get the epochs from.
        wandb_info: the wandb info to use to get the epochs from wandb (if not locally available); if not set, will use the project & entity for the current run.
    Returns:
        the sorted list of epochs available in the run.
    """
    try:
        local_epochs = sorted(
            int(x.stem.split("-")[-1])
            for x in ckpt_folder(run_id, ckpt_root).glob("checkpoint-*.pth")
        )
    except (FileNotFoundError, ValueError):
        local_epochs = []
    return local_epochs

    # try:
    #     wandb_epochs = ckpt_epoch_numbers_wandb(run_id, wandb_info)
    # except Exception as e:
    #     print(f"Warning: Could not fetch wandb checkpoints: {e}")
    #     wandb_epochs = []

    # return sorted(set(local_epochs + wandb_epochs))


def load_ckpt_epoch(run_id: str, epoch: int, wandb_info=EasyDict(entity="", project=""), mark_used=True):
    """
    Load the checkpoint for the given run_id and epoch.
    Args:
        run_id: the run id to load the checkpoint from.
        epoch: the epoch to load the checkpoint from.
        wandb_info: the wandb info to use to load the checkpoint from wandb (if not locally available); if not set, will use the project & entity for the current run.
    Returns:
        the checkpoint dictionary.
    """
    ckpt_path = run_to_ckpt_path(run_id, epoch, ckpt_root)
    if ckpt_path.exists():
        loaded = load_ckpt(ckpt_path)
        if mark_used:
            mark_as_used(run_id, epoch)
        return loaded

    print(
        f"Checkpoint not found locally at {ckpt_path}, attempting to download from wandb..."
    )
    try:
        download_ckpt_wandb(run_id, epoch, wandb_info)
        loaded = load_ckpt(ckpt_path)
        try:
            mark_as_used(run_id, epoch)
        except Exception as e:
            print(f"Warning: failed to mark checkpoint as used: {e}")
        return loaded
    except Exception as e:
        raise FileNotFoundError(f"Could not load checkpoint from local or wandb: {e}")


def load_last_ckpt(run_id: str, wandb_info=EasyDict(entity="", project=""), mark_used=True):
    """
    Load the latest checkpoint for the given run_id.
    Args:
        run_id: the run id to load the checkpoint from.
        wandb_info: the wandb info to use to load the checkpoint from wandb (if not locally available); if not set, will use the project & entity for the current run.
    """
    ckpt_epochs = ckpt_epoch_numbers(run_id, wandb_info)
    if not ckpt_epochs:
        raise FileNotFoundError(f"No checkpoints found for run_id '{run_id}' locally or on wandb")
    latest = ckpt_epochs[-1]
    print("Latest ckpt id: ", latest)
    return load_ckpt_epoch(run_id, latest, wandb_info, mark_used)


"""
Everything below are internal utils. Only the functions above are public.
"""

def ckpt_path(ckpt_folder: Path, epoch: int) -> Path:
    return ckpt_folder / f"checkpoint-{epoch}.pth"


def run_to_ckpt_path(run_name: str, epoch: int, root: str) -> Path:
    return ckpt_path(ckpt_folder(run_name, root), epoch)


def load_ckpt(path: Path) -> dict:
    return torch.load(path, weights_only=False, map_location="cpu")


def find_latest_ckpt_id(ckpt_folder: Path) -> int:
    return int(
        max(
            ckpt_folder.glob("checkpoint-*.pth"),
            key=lambda x: int(x.stem.split("-")[-1]),
        ).stem.split("-")[-1]
    )


def save_ckpt_path(ckpt_path: Path, dicts: dict):
    if not is_main_process():
        return
    print(f"Saving checkpoint to {ckpt_path}")
    make_folder(ckpt_path.parent)
    save_on_master(dicts, ckpt_path)


def get_run_name(run_name: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_run_name = f"({timestamp})_{run_name}"
    if dist.is_initialized():
        full_run_name = broadcast_object(full_run_name)
    return full_run_name


def run_folder(run_name: str, root: str) -> Path:
    return Path(root) / run_name


def ckpt_folder(run_name: str, root: str) -> Path:
    return run_folder(run_name, root) / "checkpoints"


def is_run_id(run_id: str) -> bool:
    return len(run_id) == 8 and run_id.isalnum()


def version_to_int(artifact) -> int:
    return int(artifact.version[1:])


def get_wandb_api_run(
    run_id: str, wandb_info=EasyDict(entity="", project="")
) -> "wandb.apis.public.Run":
    """Get the wandb run object (from live run or API)."""
    if wandb.run is not None and wandb.run.id == run_id:
        return wandb.Api().run(f"{wandb.run.entity}/{wandb.run.project}/{run_id}")

    entity = os.environ.get("WANDB_ENTITY") or wandb.Settings().entity
    project = os.environ.get("WANDB_PROJECT") or wandb.Settings().project
    if wandb_info.entity:
        entity = wandb_info.entity
    if wandb_info.project:
        project = wandb_info.project

    print("entity: ", entity, "project: ", project)

    if not entity or not project:
        raise ValueError(
            "Set WANDB_ENTITY and WANDB_PROJECT env vars or init a wandb run."
        )

    return wandb.Api().run(f"{entity}/{project}/{run_id}")


def get_wandb_run_path(run_id: str, wandb_info=EasyDict(entity="", project="")) -> str:
    """(Legacy) Get full run path."""
    return get_wandb_api_run(run_id, wandb_info).path


def get_ckpt_artifact(run, epoch: int) -> "wandb.Artifact":
    """Get the checkpoint artifact for a given epoch."""
    epoch = int(epoch)
    for artifact in run.logged_artifacts():
        if artifact.type == "model" and artifact.state == "COMMITTED":
            if (
                "epoch" in artifact.metadata
                and int(artifact.metadata["epoch"]) == epoch
            ):
                return artifact
    raise FileNotFoundError(f"No committed checkpoint artifact for epoch {epoch}")


def ckpt_epoch_numbers_wandb(
    run_id: str, wandb_info=EasyDict(entity="", project="")
) -> List[int]:
    """List all committed checkpoint epochs available for a run."""
    run = get_wandb_api_run(run_id, wandb_info)
    epochs = []
    for artifact in run.logged_artifacts():
        print(artifact.name)
        if artifact.type == "model" and artifact.state == "COMMITTED":
            metadata = artifact.metadata
            if "epoch" in metadata:
                try:
                    epochs.append(int(metadata["epoch"]))
                except ValueError:
                    continue
    return sorted(epochs)


def download_ckpt_wandb(run_id: str, epoch: int, wandb_info=EasyDict(entity="", project="")) -> Path:
    """Download checkpoint-{epoch}.pth from W&B and place it in the local run path."""
    run = get_wandb_api_run(run_id, wandb_info)
    artifact = get_ckpt_artifact(run, epoch)

    tmp_dir = Path("tmp_wandb_dl") / f"{run_id}-{epoch}"
    make_folder(tmp_dir)
    print(f"⬇️ Downloading artifact to: {tmp_dir}")
    artifact_dir = artifact.download(root=tmp_dir)

    # Search recursively for checkpoint file
    expected_file = f"checkpoint-{epoch}.pth"
    found_path = None
    for path in Path(artifact_dir).rglob("*"):
        if path.name == expected_file:
            found_path = path
            break

    if found_path is None:
        raise FileNotFoundError(f"Could not find {expected_file} inside artifact")

    final_path = run_to_ckpt_path(run_id, epoch, ckpt_root)
    make_folder(final_path.parent)
    shutil.move(str(found_path), final_path)
    print(f"✅ Moved checkpoint to: {final_path}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return final_path


def upload_ckpt_wandb(run_id: str, epoch: int):
    """Upload checkpoint-{epoch}.pth to W&B as artifact named model-{epoch}."""
    if wandb.run is None:
        raise RuntimeError("wandb run is not initialized. Call wandb.init() first.")

    ckpt_path = run_to_ckpt_path(run_id, epoch, ckpt_root)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    artifact = wandb.Artifact(
        name=f"model_{run_id}_epoch{epoch}", type="model", metadata={"epoch": epoch}
    )
    artifact.add_file(ckpt_path, name=f"checkpoint-{epoch}.pth")

    try:
        wandb.run.log_artifact(artifact)
        print(f"✅ Uploaded checkpoint for epoch {epoch} to wandb run {run_id}")
    except Exception as e:
        print(f"❌ Failed to upload artifact: {e}")

def clean_up(keep_latest: int=1, must_keep: List[int]=[190000,192000], dry_run: bool=False):
    runs = all_runs()
    for run_id in runs:
        ckpt_epochs = ckpt_epoch_numbers(run_id)
        # print(ckpt_epochs)
        to_keep = set()
        for v in must_keep:
            to_keep.add(v)
        for s in ckpt_epochs[-keep_latest:]:
            to_keep.add(s)
        for s in get_used(run_id):
            to_keep.add(s)
        if len(ckpt_epochs) == 1 and ckpt_epochs[0] == 0:
            # print("!!", ckpt_epochs)
            to_keep = set()
        
        to_del = []
        for epoch in ckpt_epochs:
            if epoch not in to_keep:
                if not dry_run:
                    rm_ckpt(run_id, epoch)
                else:
                    to_del.append(epoch)
        if dry_run:
            if len(to_del) > 0:
                print(f"Would delete {to_del} checkpoints for run {run_id}")
            if len(to_keep) > 0:
                print(f"Would keep {to_keep} checkpoints for run {run_id}")
            # print(f"Would keep {to_keep} checkpoints for run {run_id}")

if __name__ == "__main__":
    clean_up(keep_latest=1, must_keep=[190000,192000], dry_run=False)
    # wandb.init(project="nn_flow", entity="mit-hair")
    # run_epoch_pairs = [
    #     # ("dit_B2_cache256_20250813_052344", 99999),
    #     # ("styleGAN_L_1e-3_20250811_173041", 40000),
    #     # ("sanity_dit_B2_baseline_20250822_031035", 40000), 
    #     # ("mae_b_0.7_20250816_205044", 99999)
        
    # ]
    # for run_id, epoch in run_epoch_pairs:
    #     upload_ckpt_wandb(run_id, epoch)
# python -m utils.ckpt_utils 