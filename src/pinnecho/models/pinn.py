"""PINN wrapper mapping physical ``(x, y, t)`` to physical ``(u, v, p)``.

The network operates on *non-dimensional* coordinates and outputs so that all
quantities are O(1) during optimisation, but the public interface is fully
dimensional: callers pass physical coordinates and receive physical velocity /
pressure. Because the non-dimensionalisation is an affine transform, autograd
through :meth:`forward` yields correct physical derivatives for the PDE
residuals.
"""

import torch
import torch.nn as nn

from ..config import ModelConfig
from ..data.dataset import Scales
from .mlp import MLP


class PINN(nn.Module):
    def __init__(self, cfg: ModelConfig, scales: Scales):
        super().__init__()
        out_dim = 3 if cfg.output_pressure else 2
        self.output_pressure = cfg.output_pressure
        self.net = MLP(
            in_dim=3,
            out_dim=out_dim,
            width=cfg.hidden_width,
            depth=cfg.hidden_depth,
            activation=cfg.activation,
            fourier_features=cfg.fourier_features,
            fourier_scale=cfg.fourier_scale,
        )
        # Store reference scales as buffers so they move with .to()/dtype casts.
        self.register_buffer("length", torch.tensor(scales.length))
        self.register_buffer("time", torch.tensor(scales.time))
        self.register_buffer("velocity", torch.tensor(scales.velocity))
        self.register_buffer("pressure", torch.tensor(scales.pressure))
        self.register_buffer("center", torch.tensor([scales.center_x, scales.center_y]))

    def _normalise(self, X: torch.Tensor) -> torch.Tensor:
        xy = (X[:, 0:2] - self.center) / self.length
        t = X[:, 2:3] / self.time
        return torch.cat([xy, t], dim=1)

    def forward(self, X: torch.Tensor):
        """Return ``(u, v, p)`` in physical units (each shape ``(N, 1)``)."""
        out = self.net(self._normalise(X))
        u = out[:, 0:1] * self.velocity
        v = out[:, 1:2] * self.velocity
        if self.output_pressure:
            p = out[:, 2:3] * self.pressure
        else:
            p = torch.zeros_like(u)
        return u, v, p

    def field_dict(self, X: torch.Tensor):
        """Convenience: physical ``u, v, p`` as a detached dict for evaluation."""
        u, v, p = self.forward(X)
        return {"u": u, "v": v, "p": p}
