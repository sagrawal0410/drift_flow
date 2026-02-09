# %%
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def init_zero_linear(in_dim, out_dim):
    """
    Create a linear layer with weights initialized to zero.

    Args:
        in_dim: Input dimension
        out_dim: Output dimension

    Returns:
        nn.Linear module with weights initialized to zero
    """
    layer = nn.Linear(in_dim, out_dim)
    nn.init.normal_(layer.weight, mean=0.0, std=1e-5)
    nn.init.normal_(layer.bias, mean=0.0, std=1e-5)
    return layer


class ResMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, depth, use_ln=False):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.blocks = nn.ModuleList()
        # self.scale = nn.Parameter(torch.ones(1) / (depth**0.5))
        for _ in range(depth):
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),  # this is changed!
                    init_zero_linear(hidden_dim, hidden_dim),
                ) if not use_ln else 
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),  # this is changed!
                    init_zero_linear(hidden_dim, hidden_dim),
                )
            )

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim]

        Returns:
            output: [batch_size, output_dim]
        """
        x = self.input(x)
        for block in self.blocks:
            x = x + block(x)
        x = self.output(x)
        return x

    def trajectory(self, x):
        """
        Args:
            x: [batch_size, input_dim]

        Returns:
            A list of [batch_size, hidden_dim]
        """
        x = self.input(x)
        trajectory = [x]
        for block in self.blocks:
            blk = block(x)
            x = x + blk
            trajectory.append(x)
        return trajectory


class AdaLNMLP(nn.Module):
    def __init__(self, cond_dim, input_dim, hidden_dim, depth, output_dim):
        """
        AdaLNMLP is a conditional MLP that uses AdaLN to condition the MLP layers.

        Forward:
            x: [batch_size, input_dim]
            cond: [batch_size, cond_dim]

        Returns:
            output: [batch_size, output_dim]
        """
        super().__init__()
        self.cond_dim = cond_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.proj_in = nn.Linear(input_dim, hidden_dim)

        # Main MLP layers
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(init_zero_linear(hidden_dim, hidden_dim))

        self.final = nn.Linear(hidden_dim, output_dim)

        # Conditional scale and shift parameters for AdaLN
        self.scale_shifts = nn.ModuleList()
        for _ in range(depth):
            self.scale_shifts.append(
                nn.Sequential(
                    nn.Linear(cond_dim, hidden_dim * 2),
                    nn.SiLU(),
                    init_zero_linear(hidden_dim * 2, hidden_dim * 2),
                )
            )

        self.proj_scale = nn.Parameter(torch.ones(1) / (depth**0.5))

    def forward(self, x, cond=None):
        """
        Args:
            x: [batch_size, input_dim]
            cond: [batch_size, cond_dim]

        Returns:
            output: [batch_size, output_dim]
        """
        h = self.proj_in(x)
        for i, layer in enumerate(self.layers):
            old = h
            if cond is not None:
                scale_shift = self.scale_shifts[i](cond)
                scale, shift = scale_shift.chunk(2, dim=-1)
                h = h * (1 + scale) + shift
            h = layer(nn.SiLU()(h)) * self.proj_scale + old

        return self.final(h)


class SinusoidalTimeEmbedding(nn.Module):
    # maximum frequency: 10000 x; minimum frequency: 1 x
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: [batch_size]

        Returns:
            embeddings: [batch_size, dim]
        """
        assert t.ndim == 1, f"found shape: {t.shape}"
        device = t.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Embed(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.model = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, dim * mlp_ratio),
            nn.SiLU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size]

        Returns:
            output: [batch_size, dim]
        """
        return self.model(x)


class CondEmbed(nn.Module):
    def __init__(
        self,
        dim,
        emb_type_shapes=[("vec", 10), ("class", 9), ("pos",)],
        mlp_layers=3,
        normalize_output=False,
    ):
        """
        An embedding layer. Maps a list of labels to a vector of dimensionality dim.

        The list consists of features, each with batch size B, as specified in emb_type_shapes.

        Types include:
            ("class", n_classes): where the label has shape (B,) or (B, 1), and within range [0, n_classes)
            ("vec", vec_dim): where the label has shape (B, vec_dim)
            ("pos"): where the label has shape (B, 1) or (B).
        """
        super().__init__()

        # Create embeddings for each type
        self.embeddings = nn.ModuleList()
        self.emb_type_shapes = emb_type_shapes

        for emb_spec in emb_type_shapes:
            emb_type = emb_spec[0]

            if emb_type == "class":
                n_classes = emb_spec[1]
                self.embeddings.append(nn.Linear(n_classes, dim, bias=False))
            elif emb_type == "vec":
                vec_dim = emb_spec[1]
                self.embeddings.append(nn.Linear(vec_dim, dim))
            elif emb_type == "pos":
                self.embeddings.append(SinusoidalTimeEmbedding(dim))
            else:
                raise ValueError(f"Unknown embedding type: {emb_type}")

        self.mlp = ResMLP(dim, dim, dim, mlp_layers, use_ln=normalize_output)
        self.ln = nn.LayerNorm(dim) if normalize_output else nn.Identity()
        self.dim = dim

    def forward(self, inputs):
        """
        Args:
            inputs: List of tensors, each corresponding to an embedding type
                   as specified in emb_type_shapes.
            Each element either has shape (B,) or (B, 1) or (B, D).
            If not torch.tensor: will be converted to torch.tensor. If B=1, will be broadcasted to the batch size.

        Returns:
            embedded: [batch_size, dim]; if inputs is empty: batch_size will be 1.
        """
        device = next(self.parameters()).device
        if len(inputs) == 0:
            return torch.zeros(1, self.dim, device=device)

        if len(inputs) != len(self.emb_type_shapes):
            raise ValueError(
                f"Expected {len(self.emb_type_shapes)} inputs, got {len(inputs)}"
            )

        inputs = [
            (torch.tensor(x, device=device) if not isinstance(x, torch.Tensor) else x)
            for x in inputs
        ]

        combined = torch.zeros(1, self.dim, device=device)

        for i, (inp, emb_spec) in enumerate(zip(inputs, self.emb_type_shapes)):
            emb_type = emb_spec[0]

            if emb_type == "class":
                n_classes = emb_spec[1]
                if inp.ndim == 2 and inp.shape[1] == 1:
                    inp = inp.squeeze(1)
                if inp.ndim == 0:
                    inp = inp.reshape(1)
                one_hot = torch.eye(n_classes, device=device)[inp.long()] * (
                    n_classes**0.5
                )
                embedded = self.embeddings[i](one_hot)

            elif emb_type == "vec":
                embedded = self.embeddings[i](inp)

            elif emb_type == "pos":
                if inp.ndim == 2 and inp.shape[1] == 1:
                    inp = inp.squeeze(1)
                if inp.ndim == 0:
                    inp = inp.reshape(1)
                
                embedded = self.embeddings[i](inp)
            combined = combined + embedded
        output = self.ln(self.mlp(combined))

        return output
