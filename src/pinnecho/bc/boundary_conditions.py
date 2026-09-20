"""Boundary conditions for the two backbones.

The network architecture is identical for Model A and Model B; the *only*
difference lives here (and in the momentum forcing term), keeping the ablation
clean.

Model A (kinematic baseline) -- IMPLEMENTED
    No-slip on the moving endocardium using the wall velocity prescribed
    directly from the segmented / tracked contour: ``u_wall = d(contour)/dt``.
    Applied as a soft Dirichlet penalty. Mitral/aortic valves are Dirichlet
    inlet/outlet conditions from the (synthetic) inflow profile. The scalar
    residence time is reinitialised to zero at the inflow.

Model B (FSI-informed) -- NOT IMPLEMENTED YET (deliberate)
    Per the staged plan, Model B is not started until Model A trains
    successfully end-to-end on the toy 2D case. Its two configurable options are
    scaffolded below as clearly-marked ``NotImplementedError`` stubs:
      (i)  Dirichlet BC using the FSI-computed structural wall velocity, and
      (ii) a soft traction-continuity penalty (fluid stress = structure stress
           at the interface).

All ``*_loss`` helpers return a scalar (mean-squared) loss. Residual helpers
return raw per-point residuals for the caller to weight/reduce.
"""

from typing import List, Optional, Sequence

import torch

from ..physics import operators as ops


def dirichlet_velocity_residual(
    velocity_pred: Sequence[torch.Tensor],
    velocity_target: torch.Tensor,
) -> torch.Tensor:
    """Per-point Dirichlet residual ``u_pred - u_target`` stacked ``(N, dim)``.

    ``velocity_pred`` is a sequence of ``dim`` component tensors ``(N, 1)``;
    ``velocity_target`` is ``(N, dim)``.
    """
    preds = torch.cat(list(velocity_pred), dim=1)
    return preds - velocity_target


def _relative_mse(residual: torch.Tensor, target: torch.Tensor,
                  eps: float = 1e-12) -> torch.Tensor:
    """Dimensionless MSE: ``mean(res^2) / (mean(target^2) + eps)``.

    Relative scaling keeps BC/data terms ``O(1)`` and comparable to the PDE
    residual, which otherwise dominates (a Stage-1 lesson: absolute wall/data
    losses in SI units collapse to the trivial ``u=0`` solution).
    """
    denom = target.pow(2).mean() + eps
    return residual.pow(2).mean() / denom


# ---------------------------------------------------------------------------
# Model A (kinematic baseline) -- implemented
# ---------------------------------------------------------------------------
def wall_kinematic_loss(
    velocity_pred: Sequence[torch.Tensor],
    wall_velocity: torch.Tensor,
    relative: bool = True,
) -> torch.Tensor:
    """No-slip penalty on the moving wall (Model A kinematic BC).

    ``wall_velocity`` (``(N, dim)``) is ``d(contour)/dt`` from the tracked
    endocardial contour.
    """
    res = dirichlet_velocity_residual(velocity_pred, wall_velocity)
    return _relative_mse(res, wall_velocity) if relative else res.pow(2).mean()


def valve_dirichlet_loss(
    velocity_pred: Sequence[torch.Tensor],
    valve_velocity: torch.Tensor,
    relative: bool = True,
) -> torch.Tensor:
    """Dirichlet inlet/outlet penalty at the mitral/aortic valves.

    ``valve_velocity`` (``(N, dim)``) is the prescribed time-varying valve
    profile (not learned).
    """
    res = dirichlet_velocity_residual(velocity_pred, valve_velocity)
    return _relative_mse(res, valve_velocity) if relative else res.pow(2).mean()


def scalar_inflow_loss(c_pred: torch.Tensor, c_inflow: float = 0.0) -> torch.Tensor:
    """Reinitialise residence time to ``c_inflow`` (=0) at the inflow boundary."""
    return (c_pred - c_inflow).pow(2).mean()


# ---------------------------------------------------------------------------
# Model B (FSI-informed) -- NOT IMPLEMENTED YET (staged; do not start until
# Model A trains end-to-end on the toy 2D case).
# ---------------------------------------------------------------------------
def fsi_wall_velocity_loss(
    velocity_pred: Sequence[torch.Tensor],
    fsi_wall_velocity: torch.Tensor,
    relative: bool = True,
) -> torch.Tensor:
    """Model B, option (i): Dirichlet BC using FSI structural wall velocity.

    Differs from :func:`wall_kinematic_loss` only in its *target*: the FSI
    solver's own interface velocity (from elastodynamics), not the pure
    kinematic ``d(contour)/dt``. Numerically identical form; kept separate so
    the backbone difference is explicit and auditable. In the Stage-1 synthetic
    setting the two targets coincide (exact no-slip), so this isolates the effect
    of the momentum forcing; with real IBFE data ``fsi_wall_velocity`` differs
    subtly from the kinematic contour velocity.
    """
    res = dirichlet_velocity_residual(velocity_pred, fsi_wall_velocity)
    return _relative_mse(res, fsi_wall_velocity) if relative else res.pow(2).mean()


def fluid_traction_2d(
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    coords: torch.Tensor,
    normals: torch.Tensor,
    mu: float,
) -> torch.Tensor:
    """Newtonian fluid Cauchy traction ``t = sigma . n`` at each point (N, 2).

    ``sigma = -p I + mu (grad u + grad u^T)`` (2D). ``u, v, p`` must have been
    produced from ``coords`` (a leaf tensor with ``requires_grad=True``) so the
    velocity gradients can be taken by autograd. ``normals`` is ``(N, 2)``.
    """
    u_x, u_y = ops.d(u, coords, "x"), ops.d(u, coords, "y")
    v_x, v_y = ops.d(v, coords, "x"), ops.d(v, coords, "y")
    sigma_xx = -p + 2.0 * mu * u_x
    sigma_yy = -p + 2.0 * mu * v_y
    sigma_xy = mu * (u_y + v_x)
    nx, ny = normals[:, 0:1], normals[:, 1:2]
    t_x = sigma_xx * nx + sigma_xy * ny
    t_y = sigma_xy * nx + sigma_yy * ny
    return torch.cat([t_x, t_y], dim=1)


def traction_continuity_loss(
    velocity_pred: Sequence[torch.Tensor],
    pressure_pred: torch.Tensor,
    coords: torch.Tensor,
    normals: torch.Tensor,
    structure_traction: torch.Tensor,
    mu: float,
    relative: bool = True,
) -> torch.Tensor:
    """Model B, option (ii): soft traction-continuity penalty at the interface.

    Enforces fluid Cauchy traction ``t_f = sigma_f . n`` equal to the structural
    traction from the FSI solve. Regularises the network by physical consistency
    with the structural model rather than position-matching alone.

    ``velocity_pred`` are components produced from ``coords`` (leaf tensor with
    ``requires_grad=True``); ``structure_traction`` is ``(N, 2)`` from the FSI
    interface. Currently 2D.
    """
    if len(velocity_pred) != 2:
        raise NotImplementedError("traction_continuity_loss is 2D only for now")
    t_fluid = fluid_traction_2d(
        velocity_pred[0], velocity_pred[1], pressure_pred, coords, normals, mu
    )
    res = t_fluid - structure_traction
    return _relative_mse(res, structure_traction) if relative else res.pow(2).mean()
