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
    the backbone difference is explicit and auditable.

    NOT IMPLEMENTED: requires ``velocity_wall`` from ``load_ibfe_output``.
    """
    raise NotImplementedError(
        "Model B FSI wall-velocity BC is a future stage. Do not enable until "
        "Model A trains end-to-end on the toy 2D case (see project plan)."
    )


def traction_continuity_loss(
    velocity_pred: Sequence[torch.Tensor],
    pressure_pred: torch.Tensor,
    coords: torch.Tensor,
    normals: torch.Tensor,
    structure_traction: torch.Tensor,
    mu: float,
) -> torch.Tensor:
    """Model B, option (ii): soft traction-continuity penalty at the interface.

    Enforces fluid Cauchy traction ``t_f = sigma_f . n`` equal to the structural
    traction from the FSI solve, where for a Newtonian fluid
    ``sigma_f = -p I + mu (grad u + grad u^T)``. Regularises the network by
    physical consistency with the structural model rather than position-matching
    alone.

    NOT IMPLEMENTED: requires ``normals_wall`` and ``traction_wall`` from
    ``load_ibfe_output`` and a viscous-stress assembly from autograd velocity
    gradients. Do not start until Model A is validated (see project plan).
    """
    raise NotImplementedError(
        "Model B traction-continuity penalty is a future stage. Do not enable "
        "until Model A trains end-to-end on the toy 2D case (see project plan)."
    )
