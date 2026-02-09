import torch
from typing import Any
import wandb
import time

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


def extra_flops(module, inputs, skipped_classes):
    if isinstance(module, torch.nn.Linear):
        return inputs[0].numel() * module.out_features
    elif isinstance(module, torch.nn.Conv1d):
        return inputs[0].numel() * module.out_channels * module.kernel_size[0]
    elif isinstance(module, torch.nn.Conv2d):
        base = (
            inputs[0].numel()
            * module.out_channels
            * module.kernel_size[0]
            * module.kernel_size[1]
        )
        base /= module.stride[0] * module.stride[1]
        if module.groups > 1:
            base /= module.groups
        return base
    elif isinstance(module, torch.nn.GroupNorm):
        return inputs[0].numel()
    elif hasattr(module, "extra_flops"):
        return module.extra_flops(*inputs)
    else:
        skipped_classes.add(type(module))
        return 0
    return 0


def list_all_submodules(module):
    """
    Returns a set of all submodules (recursively) in the given module.
    Args:
        module (nn.Module): The PyTorch module to analyze

    Returns:
        set[nn.Module]: A set containing all submodules including the input module
    """
    submodules = {module}
    for child in module.children():
        submodules.update(list_all_submodules(child))
    return submodules


def format_flops(flops):
    """Convert FLOPs number to human readable format."""
    if flops == 0 or flops == "-":
        return "-"

    flops = float(flops)
    for unit in ["", "K", "M", "G", "T"]:
        if abs(flops) < 1000.0:
            return f"{flops:.1f}{unit}"
        flops /= 1000.0
    return f"{flops:.1f}P"


def format_norm(norm, median_abs=None):
    """Format norm value to human readable string."""
    if norm == 0 or norm == "-":
        return "-"

    if median_abs is None:
        return f"{norm:.4f}"
    else:
        return f"{norm:.4f} / {median_abs:.4f}"


def print_module_summary(
    module,
    inputs=[],
    kwargs=dict(),
    max_nesting=3,
    skip_redundant=True,
    include_extra_inputs=False,
    include_extra_outputs=False,
    print_table=True
):
    """
    This prints the summary of the module, including the number of parameters, buffers, input and output shapes, and the FLOPs.

    Requires:
        For each nn.module in the model, should register an `extra_flops` method, takes in the inputs to the module and returns the number of extra FLOPs.

        Here, extra FLOPs are the FLOPs that are not counted by submodules (so it's 0 for most non-leaf modules).

        If the `extra_flops` method is not defined, we assume extra FLOPs are 0 for that module.

        An example:
            class Linear(torch.nn.Module):
                def __init__(
                    self,
                    in_features,
                    out_features,
                    bias=True,
                    init_mode="kaiming_normal",
                    init_weight=1,
                    init_bias=0,
                ):
                    super().__init__()
                    self.in_features = in_features
                    self.out_features = out_features
                    init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
                    self.weight = torch.nn.Parameter(
                        weight_init([out_features, in_features], **init_kwargs) * init_weight
                    )
                    self.bias = (
                        torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias)
                        if bias
                        else None
                    )

                def forward(self, x):
                    x = x @ self.weight.to(x.dtype).t()
                    if self.bias is not None:
                        x = x.add_(self.bias.to(x.dtype))
                    return x

                def extra_flops(self, x):
                    return x.numel() * self.out_features


    Args:
        module (torch.nn.Module): The module to profile.
        inputs (tuple or list): The inputs to the module.
        max_nesting (int): The maximum nesting level to profile.
        skip_redundant (bool): Whether to skip redundant entries.
        print_table (bool): Whether to print the table.


    Returns:    
        A wandb.Table object
    """
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    with torch.inference_mode():
        entries = []
        nesting = [0]
        skipped_classes = set()
        total_flops = [0]
        flops_stack = []
        time_stack = []
        time_start = time.time()

        def pre_hook(_mod, _inputs):
            nesting[0] += 1
            flops_stack.append(total_flops[0])
            time_stack.append(time.time())
            # print(f"Pre-hook: {_mod.__class__.__name__} at nesting level {nesting[0]}")

        def post_hook(mod, _inputs, outputs):
            # print(f"Post-hook: {mod.__class__.__name__} at nesting level {nesting[0]}")
            nesting[0] -= 1
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]

            inputs = list(_inputs) if isinstance(_inputs, (tuple, list)) else [_inputs]
            inputs = [t for t in inputs if isinstance(t, torch.Tensor)]

            # Calculate input and output norms and median absolute values
            input_norms = [t.float().std() if t.numel() > 0 else 0.0 for t in inputs]
            output_norms = [t.float().std() if t.numel() > 0 else 0.0 for t in outputs]

            e_flops = extra_flops(mod, inputs, skipped_classes)
            total_flops[0] += e_flops
            module_flops = total_flops[0] - flops_stack[-1]
            flops_stack.pop()
            module_time = time.time() - time_stack[-1]
            time_stack.pop()
            
            entries.append(
                EasyDict(
                    mod=mod,
                    inputs=inputs,
                    outputs=outputs,
                    extra_flops=e_flops,
                    total_flops=module_flops,
                    time=module_time,
                    nesting=nesting[0],
                    input_norms=input_norms,
                    output_norms=output_norms,
                )
            )

        hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
        hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

        # Run module.
        outputs = module(*inputs, **kwargs)
        for hook in hooks:
            hook.remove()

        module_extra_flops = dict()
        for elt in entries:
            if elt.mod not in module_extra_flops:
                module_extra_flops[elt.mod] = elt.extra_flops
            else:
                module_extra_flops[elt.mod] += elt.extra_flops

        # Identify unique outputs, parameters, and buffers.

        # Construct table.
        rows = [
            [
                type(module).__name__,
                "Parameters",
                "Buffers",
                "Input shape",
                "Output shape",
                "Datatype",
                "FLOPs",
                "Extra FLOPs",
                "Input std",
                "Output std",
                "Time",
            ]
        ]
        rows += [["---"] * len(rows[0])]
        param_total = 0
        buffer_total = 0
        submodule_names = {mod: name for name, mod in module.named_modules()}

        for e in entries:
            if e.nesting > max_nesting:
                continue
            name = "<top-level>" if e.mod is module else submodule_names[e.mod]
            param_size = sum(t.numel() for t in e.mod.parameters())
            buffer_size = sum(t.numel() for t in e.mod.buffers())
            output_shapes = [str(list(t.shape)) for t in e.outputs]
            input_shapes = [str(list(t.shape)) for t in e.inputs]
            output_dtypes = [str(t.dtype).split(".")[-1] for t in e.outputs]
            input_dtypes = [str(t.dtype).split(".")[-1] for t in e.inputs]

            rows += [
                [
                    name
                    + (
                        ":0"
                        if (include_extra_inputs and len(e.inputs) >= 2)
                        or (include_extra_outputs and len(e.outputs) >= 2)
                        else ""
                    ),
                    format_flops(param_size) if param_size else "-",
                    format_flops(buffer_size) if buffer_size else "-",
                    (input_shapes + ["-"])[0],
                    (output_shapes + ["-"])[0],
                    (output_dtypes + ["-"])[0],
                    format_flops(e.total_flops),
                    format_flops(e.extra_flops),
                    format_norm(float((e.input_norms + [0])[0])),
                    format_norm(float((e.output_norms + [0])[0])),
                    f"{e.time:.6f}s",
                ]
            ]

            # Additional rows for extra inputs (if any)
            if include_extra_inputs:
                for idx in range(1, len(e.inputs)):
                    rows += [
                        [
                            name + f":i{idx}",
                            "-",
                            "-",
                            input_shapes[idx],
                            "-",
                            "-",
                            "-",
                            "-",
                            format_norm(float(e.input_norms[idx])),
                            "-",
                            "-",
                        ]
                    ]

            # Additional rows for extra outputs (if any)
            if include_extra_outputs:
                for idx in range(1, len(e.outputs)):
                    rows += [
                        [
                            name + f":o{idx}",
                            "-",
                            "-",
                            "-",
                            output_shapes[idx],
                            output_dtypes[idx],
                            "-",
                            "-",
                            "-",
                            format_norm(float(e.output_norms[idx])),
                            "-",
                        ]
                    ]

        param_total = sum(t.numel() for t in module.parameters())
        buffer_total = sum(t.numel() for t in module.buffers())
        rows += [["---"] * len(rows[0])]
        rows += [
            [
                "Total",
                format_flops(param_total),
                str(buffer_total),
                "-",
                "-",
                "-",
                format_flops(total_flops[0]),
                "-",
                "-",
                "-",
                f"{time.time() - time_start:.6f}s",
            ]
        ]

        # Print table.
        if print_table:
            widths = [max(len(cell) for cell in column) for column in zip(*rows)]
            print()
            for row in rows:
                print(
                    "  ".join(
                        cell + " " * (width - len(cell)) for cell, width in zip(row, widths)
                    )
                )
            print()

            print(
                "Warning: the following classes don't have an `extra_flops` method, so we assume extra FLOPs are 0 for them:"
            )
            for cls in skipped_classes:
                print(cls)

            
        del hooks
        del outputs
        del entries
        torch.cuda.empty_cache()  # Clear any GPU cache if using CUDA
        columns = rows[0]
        table = wandb.Table(columns=columns, rows=rows[1:])
        
    return table


def test():
    import torch.nn as nn

    class A(nn.Module):
        def __init__(self):
            super().__init__()
            self.params = nn.Parameter(torch.randn(72, 1024))

        def forward(self, x):
            return x @ self.params

        def extra_flops(self, x):
            return x.numel() * self.params.shape[-1]

    class B(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = nn.Sequential(A(), nn.Linear(1024, 1024))

        def forward(self, x):
            return self.seq(x)

    # b = B()
    from edm.model import construct_unet

    model = construct_unet(unet_type="ddpmpp")
    x = torch.randn(1, 3, 32, 32)
    noisy_labels = torch.rand(1)
    class_labels = torch.rand(1, 10)
    print_module_summary(model, (x, noisy_labels, class_labels), max_nesting=3)

    b = B()
    print_module_summary(b, (torch.randn(1024, 72),), max_nesting=2)


if __name__ == "__main__":
    test()
