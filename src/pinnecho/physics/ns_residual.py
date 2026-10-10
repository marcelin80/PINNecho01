"""Incompressible Navier-Stokes residuals (2D and 3D), via ``torch.autograd``.

Governing equations (dimensional, incompressible, Newtonian; blood
``rho ~ 1060 kg/m^3``, ``mu ~ 3.5e-3 Pa.s``):

    continuity:   div(u) = 0
    momentum:     rho ( du/dt + (u . grad) u ) = -grad p + mu lap(u) + f

``f`` is an optional body-forcing term (shape ``(N, spatial_dim)``):

* baseline backbone (Model A): ``f = 0`` -- generic incompressible NS;
* FSI-informed backbone (Model B): ``f`` is supplied by the IBAMR/IBFE
  fluid-structure-interaction simulation and encodes the net effect of
  myocardial active contraction / structural coupling on the fluid that a purely
  kinematic wall boundary condition cannot represent.

Coordinate convention
----------------------
Every spatio-temporal coordinate lives in a single leaf tensor ``X`` whose
columns are the spatial dimensions followed by time:

* 2D: ``X[:, :] = (x, y, t)``          -> ``X.shape == (N, 3)``
* 3D: ``X[:, :] = (x, y, z, t)``       -> ``X.shape == (N, 4)``

The number of spatial dimensions is inferred as ``X.shape[1] - 1``. The caller
must have set ``X.requires_grad_(True)`` and produced the field predictions
``u, v, [w], p`` from that same ``X`` so the autograd graph connects them.

All residuals are returned with ``create_graph=True`` derivatives so they remain
differentiable (needed for second derivatives *and* for backprop of the PDE
loss).

Note on ``torch.func``
----------------------
These residuals are written with ``torch.autograd.grad`` because it is the most
transparent and robust formulation for PINN training (the whole point-cloud is
differentiated in one reverse-mode pass, and the graph is reused for the loss
backward pass). A ``torch.func`` (``vmap`` + ``jacrev``/``jacfwd``) rewrite is a
possible performance optimisation for very large collocation batches and is
tracked as a TODO; it must produce identical residuals, which the Taylor-Green
unit test guards.
"""

from typing import List, Optional, Sequence, Tuple

import torch

from . import operators as ops


def _first_derivs(scalar: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """All first partials of ``scalar`` w.r.t. every column of ``X``.

    Returns shape ``(N, X.shape[1])`` = ``[d/dx, d/dy, (d/dz,) d/dt]``.
    """
    return ops.grad(scalar, X)


def _laplacian(scalar: torch.Tensor, X: torch.Tensor, spatial_dim: int) -> torch.Tensor:
    """Spatial Laplacian ``sum_i d2(scalar)/dx_i^2`` (shape ``(N, 1)``)."""
    g = ops.grad(scalar, X)
    lap = torch.zeros_like(scalar if scalar.dim() == 2 else scalar.unsqueeze(-1))
    for i in range(spatial_dim):
        gi = g[:, i:i + 1]
        lap = lap + ops.grad(gi, X)[:, i:i + 1]
    return lap


def continuity_residual_nd(
    velocity: Sequence[torch.Tensor], X: torch.Tensor
) -> torch.Tensor:
    """Mass-conservation residual ``div(u) = sum_i d(u_i)/dx_i`` (shape ``(N, 1)``).

    ``velocity`` is a sequence of ``spatial_dim`` component tensors, each
    ``(N, 1)``.
    """
    spatial_dim = len(velocity)
    div = torch.zeros_like(velocity[0])
    for i in range(spatial_dim):
        div = div + ops.grad(velocity[i], X)[:, i:i + 1]
    return div


def momentum_residual_nd(
    velocity: Sequence[torch.Tensor],
    p: torch.Tensor,
    X: torch.Tensor,
    rho: float,
    mu: float,
    forcing: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """Per-component momentum residuals (list of ``spatial_dim`` tensors ``(N, 1)``).

    Component ``i`` residual:

        rho ( du_i/dt + sum_j u_j du_i/dx_j ) + dp/dx_i - mu lap(u_i) - f_i

    ``forcing`` (if given) is ``(N, spatial_dim)`` and is *subtracted* from the
    residual, i.e. the equation the network is asked to satisfy is
    ``rho Du/Dt + grad p - mu lap u - f = 0``.
    """
    spatial_dim = len(velocity)
    time_col = X.shape[1] - 1

    grads = [ops.grad(velocity[i], X) for i in range(spatial_dim)]
    p_grad = ops.grad(p, X)

    residuals: List[torch.Tensor] = []
    for i in range(spatial_dim):
        gi = grads[i]
        u_i_t = gi[:, time_col:time_col + 1]
        advection = torch.zeros_like(velocity[i])
        for j in range(spatial_dim):
            advection = advection + velocity[j] * gi[:, j:j + 1]
        lap_i = _laplacian(velocity[i], X, spatial_dim)
        p_i = p_grad[:, i:i + 1]
        res_i = rho * (u_i_t + advection) + p_i - mu * lap_i
        if forcing is not None:
            res_i = res_i - forcing[:, i:i + 1]
        residuals.append(res_i)
    return residuals


def navier_stokes_residual_nd(
    velocity: Sequence[torch.Tensor],
    p: torch.Tensor,
    X: torch.Tensor,
    rho: float,
    mu: float,
    forcing: Optional[torch.Tensor] = None,
    enforce_continuity: bool = True,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Full residual bundle ``(continuity, [momentum_i])`` for 2D or 3D flow."""
    momentum = momentum_residual_nd(velocity, p, X, rho, mu, forcing=forcing)
    if enforce_continuity:
        cont = continuity_residual_nd(velocity, X)
    else:
        cont = torch.zeros_like(momentum[0])
    return cont, momentum


# ---------------------------------------------------------------------------
# 2D convenience wrappers (columns of X are (x, y, t)).
# These preserve the signatures used elsewhere in the codebase.
# ---------------------------------------------------------------------------
def continuity_residual(u: torch.Tensor, v: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """2D mass-conservation residual ``du/dx + dv/dy`` (shape ``(N, 1)``)."""
    return continuity_residual_nd((u, v), X)


def momentum_residual(
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    X: torch.Tensor,
    rho: float,
    mu: float,
    forcing: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """2D x/y momentum residuals ``(res_x, res_y)``, each shape ``(N, 1)``."""
    res = momentum_residual_nd((u, v), p, X, rho, mu, forcing=forcing)
    return res[0], res[1]


def navier_stokes_residual(
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    X: torch.Tensor,
    rho: float,
    mu: float,
    forcing: Optional[torch.Tensor] = None,
    enforce_continuity: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """2D residual bundle ``(continuity, momentum_x, momentum_y)``."""
    res_x, res_y = momentum_residual(u, v, p, X, rho, mu, forcing=forcing)
    if enforce_continuity:
        cont = continuity_residual(u, v, X)
    else:
        cont = torch.zeros_like(res_x)
    return cont, res_x, res_y
