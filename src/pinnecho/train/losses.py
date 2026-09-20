"""Loss terms for the reconstruction PINN.

Residuals are non-dimensionalised before being squared so the data, PDE and
wall terms live on comparable scales and the loss weights in the config behave
predictably regardless of the physical units.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from ..data.dataset import Scales
from ..data.doppler import project_velocity
from ..physics.navier_stokes import navier_stokes_residual
from ..physics.boundary import wall_bc_residual


@dataclass
class ResidualScales:
    continuity: float
    momentum: float

    @classmethod
    def from_scales(cls, scales: Scales, density: float) -> "ResidualScales":
        u_over_l = scales.velocity / scales.length
        mom = density * scales.velocity ** 2 / scales.length
        return cls(continuity=1.0 / max(u_over_l, 1e-30),
                   momentum=1.0 / max(mom, 1e-30))


def data_loss(model, meas_X, meas_beam, meas_doppler) -> torch.Tensor:
    """Relative Doppler misfit.

    Normalised by the mean-square Doppler signal so the term is O(1) and
    dimensionless: ~1.0 for the trivial ``u = 0`` prediction and ~noise^2 for a
    perfect fit. Without this normalisation the raw (m/s)^2 misfit is numerically
    tiny and the optimiser prefers the trivial Navier-Stokes solution ``u = 0``
    (which satisfies the baseline ``f = 0`` momentum + continuity exactly),
    collapsing the reconstruction.
    """
    u, v, _ = model(meas_X)
    proj = project_velocity(u, v, meas_beam)
    denom = torch.mean(meas_doppler ** 2).clamp_min(1e-12)
    return torch.mean((proj - meas_doppler) ** 2) / denom


def pde_loss(model, col_X, density, viscosity, res_scales: ResidualScales,
             forcing: Optional[torch.Tensor] = None,
             enforce_continuity: bool = True):
    """Return ``(continuity_loss, momentum_loss)`` (both non-dimensional MSEs)."""
    X = col_X.clone().requires_grad_(True)
    u, v, p = model(X)
    cont, res_x, res_y = navier_stokes_residual(
        u, v, p, X, rho=density, mu=viscosity,
        forcing=forcing, enforce_continuity=enforce_continuity,
    )
    cont_loss = torch.mean((cont * res_scales.continuity) ** 2)
    mom_loss = torch.mean(
        (res_x * res_scales.momentum) ** 2 + (res_y * res_scales.momentum) ** 2
    )
    return cont_loss, mom_loss


def wall_loss(model, wall_X, wall_velocity, velocity_scale: float) -> torch.Tensor:
    """Relative no-slip misfit on the moving wall.

    Normalised by the mean-square wall speed (with a floor tied to
    ``velocity_scale``) so the term is dimensionless and does not vanish just
    because the endocardial velocity is small.
    """
    u, v, _ = model(wall_X)
    res = wall_bc_residual(u, v, wall_velocity)
    denom = torch.mean(wall_velocity ** 2).clamp_min((0.05 * velocity_scale) ** 2)
    return torch.mean(res ** 2) / denom
