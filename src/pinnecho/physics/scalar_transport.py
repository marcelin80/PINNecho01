"""Passive-scalar transport residual for blood residence time.

Residence-time convention (Bosi et al. / Garcia-Villalba et al. style): a scalar
``c`` is advected by the flow and grows at unit rate everywhere, so that ``c``
measures the time a fluid parcel has spent inside the cavity since it entered:

    dc/dt + (u . grad) c = D lap(c) + 1

with

* ``c = 0`` reinitialised on the inflow (mitral valve) boundary at every time
  step -- freshly entering blood has zero residence time; and
* ``D`` small (numerical diffusion only, ``~1e-6`` in non-dimensional terms) to
  keep the field regularisable without materially smearing the transport.

The unit source term ``+1`` is what turns a generic passive scalar into a
residence-time clock: integrating ``Dc/Dt = 1`` along a pathline gives elapsed
time.

Coordinate convention matches :mod:`pinnecho.physics.ns_residual`: ``X`` is a
single leaf tensor with columns ``(x, y, t)`` (2D) or ``(x, y, z, t)`` (3D), and
``X.requires_grad_(True)`` must be set by the caller.
"""

from typing import Sequence

import torch

from . import operators as ops
from .ns_residual import _laplacian


def scalar_transport_residual(
    c: torch.Tensor,
    velocity: Sequence[torch.Tensor],
    X: torch.Tensor,
    diffusivity: float = 1e-6,
    source: float = 1.0,
) -> torch.Tensor:
    """Residence-time transport residual (shape ``(N, 1)``).

    Computes ``dc/dt + (u . grad) c - D lap(c) - source`` which should be 0
    where the PDE holds.

    Parameters
    ----------
    c
        Predicted scalar field, shape ``(N, 1)``.
    velocity
        Sequence of ``spatial_dim`` velocity components, each ``(N, 1)``.
    X
        Coordinate tensor, columns ``(x, y, [z], t)``; ``requires_grad=True``.
    diffusivity
        Scalar diffusivity ``D`` (numerical diffusion only).
    source
        Constant source rate (``1`` for the residence-time clock; set to ``0``
        for a plain passive tracer).
    """
    spatial_dim = len(velocity)
    time_col = X.shape[1] - 1

    g_c = ops.grad(c, X)
    c_t = g_c[:, time_col:time_col + 1]

    advection = torch.zeros_like(c)
    for j in range(spatial_dim):
        advection = advection + velocity[j] * g_c[:, j:j + 1]

    lap_c = _laplacian(c, X, spatial_dim)

    return c_t + advection - diffusivity * lap_c - source


def inflow_reinit_residual(
    c_pred: torch.Tensor, c_inflow: float = 0.0
) -> torch.Tensor:
    """Dirichlet residual enforcing ``c = c_inflow`` on the inflow boundary.

    Blood entering through the mitral valve is reinitialised to zero residence
    time, so this is used as a soft boundary penalty on inflow collocation
    points: ``mean((c_pred - c_inflow)^2)``. Returns the raw per-point residual
    ``c_pred - c_inflow`` (shape ``(N, 1)``); squaring/reduction is the loss
    module's job.
    """
    return c_pred - c_inflow
