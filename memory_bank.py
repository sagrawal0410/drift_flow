import torch
from collections import deque, defaultdict
import torch
from collections import deque, defaultdict
from typing import Dict, Tuple, Any
import random 

def _is_tensor_leaf(x):
    return isinstance(x, torch.Tensor)

def _flatten_tree(tree: Dict[str, Any], prefix: Tuple[str, ...] = ()):
    """
    Flattens a nested dict-of-dicts tree into { key_path_tuple: tensor }.
    Leaves must be tensors.
    """
    flat = {}
    for k, v in tree.items():
        kp = prefix + (k,)
        if isinstance(v, dict):
            flat.update(_flatten_tree(v, kp))
        else:
            if not _is_tensor_leaf(v):
                raise TypeError(f"Leaf at {kp} must be a torch.Tensor, got {type(v)}")
            flat[kp] = v
    return flat

def _unflatten_tree(flat_map: Dict[Tuple[str, ...], Any]) -> Dict[str, Any]:
    """
    Reconstructs a nested dict from { key_path_tuple: value }.
    """
    root: Dict[str, Any] = {}
    for path, value in flat_map.items():
        d = root
        for key in path[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[path[-1]] = value
    return root

class MemoryBank:
    def __init__(self, num_classes=1000, max_size=100, device="cpu"):
        self.num_classes = num_classes
        self.max_size = max_size
        self.bank = defaultdict(lambda: deque(maxlen=self.max_size))  # per-class -> deque of flat dicts
        self.device = device
        self.sample_shapes = None     # { key_path: shape_without_batch }
        self.sample_dtypes = None     # { key_path: dtype }
        self.flat_keys = None         # list of key_paths for quick iteration

    def add(self, samples_nested: Dict[str, Any], labels: torch.Tensor):
        """
        Add a batch of samples to the memory bank.
        Args:
            samples_nested: 
                nested dict; leaves are tensors shaped [B, *leaf_shape]
            labels: LongTensor of shape [B] (int class indices)
        """
        # Flatten & device/detach
        flat = _flatten_tree(samples_nested)
        if self.device == "cpu":
            flat = {k: v.detach().cpu().contiguous().pin_memory() for k, v in flat.items()}
        else:
            flat = {k: v.detach().to(self.device) for k, v in flat.items()}

        # Capture shapes/dtypes and key set on first add
        if self.sample_shapes is None:
            self.sample_shapes = {k: v.shape[1:] for k, v in flat.items()}
        if self.sample_dtypes is None:
            self.sample_dtypes = {k: v.dtype for k, v in flat.items()}
        if self.flat_keys is None:
            self.flat_keys = list(flat.keys())

        # Basic checks
        B = labels.shape[0]
        for k in self.flat_keys:
            if flat[k].shape[0] != B:
                raise ValueError(f"All leaves must share the same batch size B; leaf {k} has {flat[k].shape[0]} vs {B}")

        labels = labels.detach().to(self.device)

        # Split the batch into individual samples (still flattened)
        for i in range(B):
            sample_flat = {k: flat[k][i] for k in self.flat_keys}
            self.bank[int(labels[i])].append(sample_flat)

    def sample(self, labels: torch.Tensor, n_samples: int) -> Dict[str, Any]:
        """
        For each label in the batch, sample n_samples from the bank.
        Args:
            labels: LongTensor of shape [B] (int class indices)
            n_samples: number of samples to draw for each label
        Returns:
            Nested dict matching the input structure; each leaf has shape [B, n_samples, *leaf_shape]
        """
        if self.sample_shapes is None or self.sample_dtypes is None or self.flat_keys is None:
            raise RuntimeError("MemoryBank is empty. Add samples before sampling.")

        labels = labels.detach().to(self.device)
        B = labels.shape[0]

        # Collect per-item sampled tensors (still flattened)
        per_item_flat_samples = []  # length B; each is { key_path: Tensor[n_samples, *shape] }
        for b in range(B):
            label = int(labels[b])
            dq = self.bank[label]

            if not dq:
                # No data for this label -> zeros
                flat_samples = {
                    k: torch.zeros((n_samples,) + self.sample_shapes[k],
                                   device=self.device, dtype=self.sample_dtypes[k])
                    for k in self.flat_keys
                }
            else:
                if len(dq) >= n_samples:
                    gathered = random.sample(dq, n_samples)
                else:
                    gathered = random.choices(dq, k=n_samples)
                stacks = {}
                for k in self.flat_keys:
                    stacks[k] = torch.stack([g[k] for g in gathered], dim=0).to(self.device)
                flat_samples = stacks

            per_item_flat_samples.append(flat_samples)

        # Now merge across batch -> shape [B, n_samples, ...] per leaf
        flat_batched = {k: [] for k in self.flat_keys}
        for b in range(B):
            for k in self.flat_keys:
                flat_batched[k].append(per_item_flat_samples[b][k])
        flat_batched = {k: torch.stack(v_list, dim=0) for k, v_list in flat_batched.items()}  # [B, n_samples, *]

        # Reconstruct nested structure and return
        return _unflatten_tree(flat_batched)

    def state_dict(self):
        """
        Serialize MemoryBank state for checkpointing.
        Tensors are moved to CPU for portability.
        """
        # Serialize deques -> lists, and tuple keys -> joined strings for JSON-ability
        bank_serialized = {}
        for cls_idx, dq in self.bank.items():
            items = []
            for sample_flat in dq:
                items.append({"|".join(k): v.detach().cpu() for k, v in sample_flat.items()})
            bank_serialized[int(cls_idx)] = items

        # Dtype names for robust restore
        dtypes_serialized = None
        if self.sample_dtypes is not None:
            dtypes_serialized = {"|".join(k): v.name if hasattr(v, "name") else str(v) for k, v in self.sample_dtypes.items()}

        flat_keys_serialized = None
        if self.flat_keys is not None:
            flat_keys_serialized = ["|".join(k) for k in self.flat_keys]

        shapes_serialized = None
        if self.sample_shapes is not None:
            shapes_serialized = {"|".join(k): tuple(v) for k, v in self.sample_shapes.items()}

        return {
            "num_classes": self.num_classes,
            "max_size": self.max_size,
            "device": str(self.device),
            "sample_shapes": shapes_serialized,
            "sample_dtypes": dtypes_serialized,
            "flat_keys": flat_keys_serialized,
            "bank": bank_serialized,
        }

    def load_state_dict(self, state):
        """
        Restore MemoryBank from a state dict produced by state_dict().
        Existing contents are replaced.
        """
        # Update basic configs
        self.num_classes = int(state.get("num_classes", self.num_classes))
        saved_max_size = int(state.get("max_size", self.max_size))
        if saved_max_size != self.max_size:
            self.max_size = saved_max_size
        # Recreate bank with correct maxlen
        self.bank = defaultdict(lambda: deque(maxlen=self.max_size))

        # Restore metadata
        shapes_serialized = state.get("sample_shapes")
        if shapes_serialized is not None:
            self.sample_shapes = {tuple(k.split("|")): tuple(v) for k, v in shapes_serialized.items()}
        else:
            self.sample_shapes = None

        dtypes_serialized = state.get("sample_dtypes")
        if dtypes_serialized is not None:
            def to_dtype(name: str):
                # Expect names like "float32", "int64"; fallback to torch.<name> if present
                short = name.replace("torch.", "")
                return getattr(torch, short)
            self.sample_dtypes = {tuple(k.split("|")): to_dtype(v) for k, v in dtypes_serialized.items()}
        else:
            self.sample_dtypes = None

        flat_keys_serialized = state.get("flat_keys")
        if flat_keys_serialized is not None:
            self.flat_keys = [tuple(k.split("|")) for k in flat_keys_serialized]
        else:
            self.flat_keys = None

        # Restore bank contents
        bank_serialized = state.get("bank", {})
        for cls_idx_str, items in bank_serialized.items():
            cls_idx = int(cls_idx_str)
            dq = deque(maxlen=self.max_size)
            for sample_flat_ser in items:
                sample_flat = {tuple(k.split("|")): v.to(self.device) for k, v in sample_flat_ser.items()}
                dq.append(sample_flat)
            self.bank[cls_idx] = dq

        # If keys weren't present but we have shapes, derive flat_keys
        if self.flat_keys is None and self.sample_shapes is not None:
            self.flat_keys = list(self.sample_shapes.keys())

        return self


import tqdm
if __name__ == "__main__":
    device = "cuda"
    memory_bank = MemoryBank(num_classes=1000, max_size=128,device=device)
    for i in tqdm.tqdm(range(20000)):
        # Use a single data tensor as requested
        samples_dict = {
            'data': torch.randn(16, 4, 32, 32),
        }
        labels = torch.randint(0, 1000, (16,))
        memory_bank.add(samples_dict, labels)
        y_dict = memory_bank.sample(labels, n_samples=4)
        
        if i > 100 and y_dict:
            # Shape is [B, n_samples, *leaf_shape] -> [16, 4, 4, 32, 32]
            assert 'data' in y_dict and y_dict['data'].shape == (16, 4, 4, 32, 32)

    print("Initial sampling test passed. Testing state_dict save/load.")

    # Get state dict from the original bank
    state1 = memory_bank.state_dict()

    # --- Save and measure size ---
    import os
    save_path = "memory_bank_test_state.pt"
    print(f"Saving state_dict to {save_path}...")
    torch.save(state1, save_path)

    # Check file size
    file_size_bytes = os.path.getsize(save_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"State dict file size: {file_size_mb:.2f} MB")
    # -----------------------------

    # Create a new bank and load the state from file
    print("Loading state_dict from file...")
    loaded_state = torch.load(save_path)
    new_memory_bank = MemoryBank(num_classes=1, max_size=1, device=memory_bank.device)
    new_memory_bank.load_state_dict(loaded_state)

    # Get state dict from the new bank for verification
    state2 = new_memory_bank.state_dict()

    # Compare metadata
    assert state1["num_classes"] == state2["num_classes"]
    assert state1["max_size"] == state2["max_size"]
    assert state1["device"] == state2["device"]
    assert state1["sample_shapes"] == state2["sample_shapes"]
    assert state1["sample_dtypes"] == state2["sample_dtypes"]
    assert state1["flat_keys"] == state2["flat_keys"]

    # Compare bank contents
    bank1 = state1["bank"]
    bank2 = state2["bank"]
    assert bank1.keys() == bank2.keys()

    for cls_idx in bank1.keys():
        items1 = bank1[cls_idx]
        items2 = bank2[cls_idx]
        assert len(items1) == len(items2)
        for sample1, sample2 in zip(items1, items2):
            assert sample1.keys() == sample2.keys()
            for key in sample1.keys():
                assert torch.equal(
                    sample1[key], sample2[key]
                ), f"Tensor mismatch for key {key}"

    print("State dict save/load test passed.")
    print("All tests passed.")

# python -m memory_bank