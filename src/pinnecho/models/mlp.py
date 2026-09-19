"""MLP building blocks, with an optional Fourier-feature embedding.

A plain tanh MLP (``fourier_features = 0``) is the recommended default here:
its outputs are smooth and, crucially, so are its *derivatives*, which matters
because this project reports velocity-gradient quantities (vorticity, wall shear
stress). Random Fourier features (Tancik et al. 2020) fix the spectral bias of
plain MLPs and speed up fitting of high-frequency *values*, but at moderate/high
feature scales they inject high-frequency wiggle that fits the velocity while
badly corrupting its curl -- empirically, even a *fully supervised* fit with
Fourier features (scale 2-5) recovered vorticity/WSS an order of magnitude worse
than a plain tanh MLP. Fourier features are therefore kept available (with a
small scale) but disabled by default. See the README for the diagnostic.
"""

from typing import Callable

import torch
import torch.nn as nn


_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "sin": None,  # handled specially below
}


class Sine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class FourierFeatures(nn.Module):
    """Random Fourier feature embedding ``x -> [sin(2*pi*B x), cos(2*pi*B x)]``.

    ``B`` is a fixed (non-trainable) Gaussian matrix. Output dimension is
    ``2 * num_features``.
    """

    def __init__(self, in_dim: int, num_features: int, scale: float):
        super().__init__()
        if num_features <= 0:
            self.register_buffer("B", torch.empty(in_dim, 0))
            self.out_dim = in_dim
            self._identity = True
        else:
            B = torch.randn(in_dim, num_features) * scale
            self.register_buffer("B", B)
            self.out_dim = 2 * num_features
            self._identity = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._identity:
            return x
        proj = 2.0 * torch.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class MLP(nn.Module):
    """Fourier-feature MLP with configurable width/depth/activation."""

    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int,
                 activation: str = "tanh", fourier_features: int = 0,
                 fourier_scale: float = 1.0):
        super().__init__()
        self.embed = FourierFeatures(in_dim, fourier_features, fourier_scale)

        def act() -> nn.Module:
            if activation == "sin":
                return Sine()
            if activation not in _ACTIVATIONS or _ACTIVATIONS[activation] is None:
                raise ValueError(f"Unknown activation '{activation}'")
            return _ACTIVATIONS[activation]()

        layers: list[nn.Module] = [nn.Linear(self.embed.out_dim, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.embed(x))
