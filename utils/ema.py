# %%
import torch
import torch.nn as nn
from copy import deepcopy
from utils.persistence import persistent_class

class EMA(nn.Module):
    def __init__(self, model, decay=0.999):
        super().__init__()
        self.model = deepcopy(model)
        self.decay = decay
        self.model.eval()
        self.model.to(next(model.parameters()).device)

    def forward(self, x):
        return self.model(x)

    def update(self, model, decay=None):
        if decay is None:
            decay = self.decay
        with torch.no_grad():
            # Get device of source model
            device = next(model.parameters()).device

            # Ensure EMA model is on same device
            if next(self.model.parameters()).device != device:
                self.model.to(device)

            # Update parameters
            for ema_param, model_param in zip(
                self.model.parameters(), model.parameters()
            ):
                ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)
                
            # Update buffers (running_mean, running_var, etc.)
            for ema_buffer, model_buffer in zip(
                self.model.buffers(), model.buffers()
            ):
                if ema_buffer.data.dtype == torch.long or ema_buffer.data.dtype == torch.int:
                    ema_buffer.copy_(model_buffer)
                else:
                    ema_buffer.data.mul_(decay).add_(model_buffer.data, alpha=1 - decay)

    def to(self, device):
        self.model.to(device)
        return self

class EDMEMA(nn.Module):
    def __init__(self, model, percent=0.10):
        super().__init__()
        self.model = deepcopy(model)
        self.schedule = EDM2Schedule(percent)
        self.model.eval()
        self.model.to(next(model.parameters()).device)
        self.step = 0

    def forward(self, x):
        return self.model(x)

    def set_step(self, step):
        self.step = step

    def update(self, model):
        self.step += 1
        decay = self.schedule.decay_scale(self.step)
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.model.parameters(), model.parameters()
            ):
                ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)
            for ema_buffer, model_buffer in zip(
                self.model.buffers(), model.buffers()
            ):
                if ema_buffer.data.dtype == torch.long or ema_buffer.data.dtype == torch.int:
                    ema_buffer.copy_(model_buffer)
                else:
                    ema_buffer.data.mul_(decay).add_(model_buffer.data, alpha=1 - decay)

class EDM2Schedule:
    def __init__(self, percent=0.10):
        # (r+1)^{1/2}(r+2)^{-1}(r+3)^{-1/2}=percent, solve for r
        self.percent = percent
        self.r = self._solve_for_r(percent)

    def _solve_for_r(self, percent):
        def equation(r):
            return ((r + 1) ** 0.5) * ((r + 2) ** -1) * ((r + 3) ** -0.5) - percent

        r_min, r_max = 0.0, 1000.0
        while r_max - r_min > 1e-10:
            r_mid = (r_min + r_max) / 2
            if equation(r_mid) > 0:
                r_min = r_mid
            else:
                r_max = r_mid

        return (r_min + r_max) / 2

    def decay_scale(self, t):
        """
        Calculate the decay scale given timestep t.

        Args:
            t: The timestep

        Returns:
            The decay scale (1-1/t)^(r+1)
        """
        if t <= 1:
            return 0.0
        return (1 - 1 / t) ** (self.r + 1)
