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


def fluid_traction(
    velocity: Sequence[torch.Tensor],
    p: torch.Tensor,
    coords: torch.Tensor,
    normals: torch.Tensor,
    mu: float,
) -> torch.Tensor:
    """Newtonian fluid Cauchy traction ``t = sigma . n`` (2D or 3D), shape ``(N, dim)``.

    ``sigma_ij = -p delta_ij + mu (du_i/dx_j + du_j/dx_i)`` so that
    ``t_i = -p n_i + sum_j mu (du_i/dx_j + du_j/dx_i) n_j``.

    ``velocity`` is a sequence of ``dim`` component tensors ``(N, 1)`` produced
    from ``coords`` (a leaf tensor with ``requires_grad=True``); ``normals`` is
    ``(N, dim)``. Dimension is inferred from ``len(velocity)``. Spatial gradient
    columns are sliced by index (0..dim-1), so this is agnostic to where the time
    column sits (unlike the ``"x"/"y"`` name map, which is 2D-specific).
    """
    dim = len(velocity)
    grads = [ops.grad(velocity[i], coords) for i in range(dim)]  # du_i/dx_j at [:, j]
    cols = []
    for i in range(dim):
        t_i = -p * normals[:, i:i + 1]
        for j in range(dim):
            dui_dxj = grads[i][:, j:j + 1]
            duj_dxi = grads[j][:, i:i + 1]
            t_i = t_i + mu * (dui_dxj + duj_dxi) * normals[:, j:j + 1]
        cols.append(t_i)
    return torch.cat(cols, dim=1)


def fluid_traction_2d(
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    coords: torch.Tensor,
    normals: torch.Tensor,
    mu: float,
) -> torch.Tensor:
    """2D convenience wrapper around :func:`fluid_traction` (``(N, 2)``)."""
    return fluid_traction([u, v], p, coords, normals, mu)


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

    ``velocity_pred`` are the ``dim`` components produced from ``coords`` (leaf
    tensor with ``requires_grad=True``); ``structure_traction`` is ``(N, dim)``
    from the FSI interface. Supports 2D and 3D.
    """
    dim = len(velocity_pred)
    if dim not in (2, 3):
        raise ValueError(f"traction_continuity_loss expects 2 or 3 components, got {dim}")
    t_fluid = fluid_traction(velocity_pred, pressure_pred, coords, normals, mu)
    res = t_fluid - structure_traction
    return _relative_mse(res, structure_traction) if relative else res.pow(2).mean()
