"""Incompressible Navier-Stokes residuals for the 2D LV cavity.

The momentum balance we enforce is

    rho * ( u_t + (u . grad) u ) = -grad p + mu * lap u + f

where ``f`` is a body-forcing term. For the *baseline* backbone ``f = 0``
(generic incompressible NS). For the *FSI-informed* backbone ``f`` is supplied
by the fluid-structure-interaction simulation and represents the net effect of
myocardial active contraction / structural coupling on the fluid that a purely
kinematic wall boundary condition cannot capture.
"""

from typing import Optional, Tuple

import torch

from . import operators as ops


def continuity_residual(u: torch.Tensor, v: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Mass conservation residual ``du/dx + dv/dy`` (should be 0)."""
    return ops.divergence(u, v, X)


def momentum_residual(
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    X: torch.Tensor,
    rho: float,
    mu: float,
    forcing: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the x- and y-momentum residuals (each shape ``(N, 1)``).

    ``forcing`` (if given) is an ``(N, 2)`` tensor ``[f_x, f_y]`` that is
    *subtracted* from the residual, i.e. the momentum equation the network is
    asked to satisfy becomes ``rho Du/Dt + grad p - mu lap u - f = 0``.
    """
    g_u = ops.grad(u, X)
    g_v = ops.grad(v, X)

    u_t, u_x, u_y = g_u[:, 2:3], g_u[:, 0:1], g_u[:, 1:2]
    v_t, v_x, v_y = g_v[:, 2:3], g_v[:, 0:1], g_v[:, 1:2]

    lap_u = ops.d(g_u[:, 0:1], X, "x") + ops.d(g_u[:, 1:2], X, "y")
    lap_v = ops.d(g_v[:, 0:1], X, "x") + ops.d(g_v[:, 1:2], X, "y")

    p_x = ops.d(p, X, "x")
    p_y = ops.d(p, X, "y")

    res_x = rho * (u_t + u * u_x + v * u_y) + p_x - mu * lap_u
    res_y = rho * (v_t + u * v_x + v * v_y) + p_y - mu * lap_v

    if forcing is not None:
        res_x = res_x - forcing[:, 0:1]
        res_y = res_y - forcing[:, 1:2]

    return res_x, res_y


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
    """Full residual bundle ``(continuity, momentum_x, momentum_y)``."""
    res_x, res_y = momentum_residual(u, v, p, X, rho, mu, forcing=forcing)
    if enforce_continuity:
        cont = continuity_residual(u, v, X)
    else:
        cont = torch.zeros_like(res_x)
    return cont, res_x, res_y
