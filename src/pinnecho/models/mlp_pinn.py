"""Spec-aligned PINN network: multi-scale Fourier features + MLP + field heads.

Architecture (identical for Model A and Model B -- the *only* difference between
backbones is the boundary-condition / forcing term, never the network, so the
ablation stays clean):

    (x, y, [z], t)
        -> multi-scale random Fourier-feature encoding
           B_sigma ~ N(0, sigma^2), sigma in {1, 10, 100} concatenated
           (Tancik et al. 2020; multi-scale per Patel/Kreusser/Fraser 2025)
        -> MLP (default 8 layers x 256 units, tanh or sine activation)
        -> output heads (u, v, [w], p, c)

``c`` is the residence-time scalar. For a 2D run the outputs are ``(u, v, p, c)``
(4 heads); for 3D they are ``(u, v, w, p, c)`` (5 heads).

Input normalisation
-------------------
Fourier features assume inputs are ``O(1)``: with physical coordinates (metres,
seconds) the ``sigma=100`` band would barely oscillate. Pass ``input_lows`` /
``input_highs`` (per-column domain bounds incl. time) and the module maps inputs
to ``[-1, 1]`` before encoding. If omitted, inputs are used as-is (appropriate
when the caller already non-dimensionalises).

Empirical caveat (Stage-1 finding, see README): at moderate/high Fourier scales
the encoding fits velocity *values* well but injects high-frequency wiggle that
corrupts the *curl* (vorticity) and wall-shear-stress -- which are the very
gradient quantities this project reports. ``fourier_scales`` therefore accepts
an empty list to fall back to a plain tanh MLP. Both modes are kept so the
Fourier design requested in the spec can be run and compared head-to-head.
"""

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn


class Sine(nn.Module):
    """``sin`` activation (SIREN-style)."""

    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class MultiScaleFourierFeatures(nn.Module):
    """Concatenated random Fourier features at several Gaussian scales.

    For each ``sigma`` in ``scales`` a fixed matrix ``B_sigma ~ N(0, sigma^2)``
    of shape ``(in_dim, num_features)`` is drawn, and the encoding is

        [ sin(2 pi X B_sigma), cos(2 pi X B_sigma) ]  concatenated over sigma.

    Output dimension is ``2 * num_features * len(scales)``. If ``scales`` is
    empty the layer is the identity (plain MLP), with ``out_dim == in_dim``.
    """

    def __init__(self, in_dim: int, num_features: int, scales: Sequence[float],
                 seed: Optional[int] = None):
        super().__init__()
        self.in_dim = in_dim
        self.scales = list(scales)
        if not self.scales or num_features <= 0:
            self.out_dim = in_dim
            self._identity = True
            self.register_buffer("B", torch.empty(in_dim, 0))
            return

        self._identity = False
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        blocks = []
        for sigma in self.scales:
            B = torch.randn(in_dim, num_features, generator=gen) * float(sigma)
            blocks.append(B)
        # Store one stacked buffer (in_dim, num_features * n_scales).
        self.register_buffer("B", torch.cat(blocks, dim=1))
        self.out_dim = 2 * num_features * len(self.scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._identity:
            return x
        proj = 2.0 * torch.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class PINNNet(nn.Module):
    """Multi-scale-Fourier MLP mapping ``(x, y, [z], t)`` to field heads.

    Parameters
    ----------
    spatial_dim
        2 or 3. Determines the number of coordinate inputs and velocity heads.
    width, depth
        Hidden width and number of hidden layers (default 256 x 8).
    activation
        ``"tanh"`` or ``"sin"``.
    fourier_features
        Number of Fourier features *per scale* (0 disables Fourier encoding).
    fourier_scales
        Gaussian scales (sigmas) to concatenate, e.g. ``(1.0, 10.0, 100.0)``.
    predict_scalar
        If True, adds the residence-time head ``c``.
    input_lows, input_highs
        Optional per-column domain bounds (length ``spatial_dim + 1``) used to
        normalise inputs to ``[-1, 1]`` before Fourier encoding.
    """

    def __init__(
        self,
        spatial_dim: int = 2,
        width: int = 256,
        depth: int = 8,
        activation: str = "tanh",
        fourier_features: int = 0,
        fourier_scales: Sequence[float] = (1.0, 10.0, 100.0),
        predict_scalar: bool = True,
        input_lows: Optional[Sequence[float]] = None,
        input_highs: Optional[Sequence[float]] = None,
        velocity_scale: float = 1.0,
        pressure_scale: float = 1.0,
        scalar_scale: float = 1.0,
        fourier_seed: Optional[int] = 0,
    ):
        super().__init__()
        if spatial_dim not in (2, 3):
            raise ValueError("spatial_dim must be 2 or 3")
        self.spatial_dim = spatial_dim
        self.predict_scalar = predict_scalar
        in_dim = spatial_dim + 1  # spatial coords + time

        # Output heads: velocity (spatial_dim) + pressure (1) + scalar (0/1).
        self.out_dim = spatial_dim + 1 + (1 if predict_scalar else 0)

        # Output scales keep the raw network O(1) while heads carry physical
        # magnitudes (u ~ velocity_scale, p ~ pressure_scale, c ~ scalar_scale).
        # The transform is affine, so autograd through fields() gives correct
        # physical derivatives for the residuals.
        self.register_buffer("velocity_scale", torch.as_tensor(float(velocity_scale)))
        self.register_buffer("pressure_scale", torch.as_tensor(float(pressure_scale)))
        self.register_buffer("scalar_scale", torch.as_tensor(float(scalar_scale)))

        self._register_normalisation(in_dim, input_lows, input_highs)

        self.embed = MultiScaleFourierFeatures(
            in_dim, fourier_features, fourier_scales, seed=fourier_seed
        )

        act = self._activation_factory(activation)
        layers: List[nn.Module] = [nn.Linear(self.embed.out_dim, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, self.out_dim)]
        self.net = nn.Sequential(*layers)
        self._activation = activation
        self._init_weights()

    def _register_normalisation(self, in_dim, lows, highs):
        if lows is None or highs is None:
            self._normalise = False
            self.register_buffer("in_lo", torch.zeros(in_dim))
            self.register_buffer("in_hi", torch.ones(in_dim))
            return
        lo = torch.as_tensor(lows, dtype=torch.get_default_dtype())
        hi = torch.as_tensor(highs, dtype=torch.get_default_dtype())
        if lo.numel() != in_dim or hi.numel() != in_dim:
            raise ValueError(f"input bounds must have length {in_dim}")
        self._normalise = True
        self.register_buffer("in_lo", lo)
        self.register_buffer("in_hi", hi)

    @staticmethod
    def _activation_factory(activation: str):
        if activation == "tanh":
            return nn.Tanh
        if activation == "sin":
            return Sine
        raise ValueError(f"Unknown activation '{activation}' (use 'tanh' or 'sin')")

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                if self._activation == "sin":
                    # SIREN-style init keeps pre-activations well-scaled.
                    fan_in = m.weight.shape[1]
                    bound = (6.0 / fan_in) ** 0.5
                    nn.init.uniform_(m.weight, -bound, bound)
                else:
                    nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def _normalise_inputs(self, X: torch.Tensor) -> torch.Tensor:
        if not self._normalise:
            return X
        return 2.0 * (X - self.in_lo) / (self.in_hi - self.in_lo) - 1.0

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Raw network output, shape ``(N, out_dim)``.

        Use :meth:`fields` for a labelled dict of the individual heads.
        """
        return self.net(self.embed(self._normalise_inputs(X)))

    def fields(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return named field predictions, each shape ``(N, 1)``.

        Keys: ``u``, ``v``, (``w`` if 3D), ``p``, and ``c`` (if ``predict_scalar``).
        """
        out = self.forward(X)
        result: Dict[str, torch.Tensor] = {}
        idx = 0
        names = ["u", "v", "w"][: self.spatial_dim]
        for name in names:
            result[name] = out[:, idx:idx + 1] * self.velocity_scale
            idx += 1
        result["p"] = out[:, idx:idx + 1] * self.pressure_scale
        idx += 1
        if self.predict_scalar:
            result["c"] = out[:, idx:idx + 1] * self.scalar_scale
        return result

    def velocity(self, X: torch.Tensor) -> List[torch.Tensor]:
        """Convenience: list of velocity components for the physics residuals."""
        f = self.fields(X)
        return [f[k] for k in (["u", "v", "w"][: self.spatial_dim])]
