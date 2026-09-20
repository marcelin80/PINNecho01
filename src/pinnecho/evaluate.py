"""Validation metrics vs held-out IBFE ground truth (spec-aligned).

Implements the quantitative comparison plan:

* velocity field RMSE (all components -- checks physics-driven recovery of the
  UNSEEN cross-beam component, even though only the beam component was in the
  data loss);
* relative pressure RMSE;
* vorticity / Q-criterion field correlation and RMSE;
* wall shear stress (WSS) RMSE;
* residence-time field RMSE;
* stagnation-volume agreement;
* a physics-free sanity baseline (spline/griddata interpolation of the sparse
  Doppler data) to demonstrate the PINN's value-add.

Field metrics take precomputed arrays so they are unit-testable in isolation.
Gradient quantities (vorticity, WSS) that require differentiating the network are
provided as autograd helpers taking a :class:`~pinnecho.models.mlp_pinn.PINNNet`.

The end-to-end ``evaluate`` / ``compare_backbones`` driver (loading IBFE GT,
running both trained models, tabulating side-by-side) is scaffolded with TODOs:
it depends on ``load_ibfe_output`` and on trained checkpoints from the (gated)
training stage.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .physics import operators as ops


# ---------------------------------------------------------------------------
# Elementary field metrics
# ---------------------------------------------------------------------------
def rmse(pred: torch.Tensor, true: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - true) ** 2)))


def relative_l2(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-12) -> float:
    """Relative L2 error ``||pred - true|| / ||true||``."""
    num = torch.linalg.norm((pred - true).reshape(-1))
    den = torch.linalg.norm(true.reshape(-1)) + eps
    return float(num / den)


def correlation(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Pearson correlation between two fields (flattened)."""
    a = pred.reshape(-1) - pred.mean()
    b = true.reshape(-1) - true.mean()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b) + 1e-12
    return float((a @ b) / denom)


def velocity_metrics(
    vel_pred: torch.Tensor, vel_true: torch.Tensor
) -> Dict[str, float]:
    """Per-component + speed RMSE and relative L2 for a velocity field.

    ``vel_pred``/``vel_true`` are ``(N, dim)``.
    """
    dim = vel_true.shape[1]
    names = ["u", "v", "w"][:dim]
    out: Dict[str, float] = {}
    for i, name in enumerate(names):
        out[f"rmse_{name}"] = rmse(vel_pred[:, i], vel_true[:, i])
        out[f"relL2_{name}"] = relative_l2(vel_pred[:, i], vel_true[:, i])
    speed_p = torch.linalg.norm(vel_pred, dim=1)
    speed_t = torch.linalg.norm(vel_true, dim=1)
    out["rmse_speed"] = rmse(speed_p, speed_t)
    out["relL2_speed"] = relative_l2(speed_p, speed_t)
    return out


def pressure_metrics(p_pred: torch.Tensor, p_true: torch.Tensor) -> Dict[str, float]:
    """Pressure is defined up to a constant -> compare after mean-centering."""
    pp = p_pred - p_pred.mean()
    pt = p_true - p_true.mean()
    return {"rmse_p": rmse(pp, pt), "relL2_p": relative_l2(pp, pt),
            "corr_p": correlation(pp, pt)}


def scalar_field_metrics(c_pred: torch.Tensor, c_true: torch.Tensor) -> Dict[str, float]:
    """Residence-time (or any scalar) field RMSE / relative L2."""
    return {"rmse_c": rmse(c_pred, c_true), "relL2_c": relative_l2(c_pred, c_true)}


# ---------------------------------------------------------------------------
# Gradient quantities via autograd (vorticity, Q-criterion, WSS)
# ---------------------------------------------------------------------------
def vorticity_2d(model, X: torch.Tensor) -> torch.Tensor:
    """Out-of-plane vorticity ``omega_z = dv/dx - du/dy`` from the model (N,1)."""
    X = X.clone().requires_grad_(True)
    f = model.fields(X)
    u, v = f["u"], f["v"]
    return ops.d(v, X, "x") - ops.d(u, X, "y")


def q_criterion_2d(model, X: torch.Tensor) -> torch.Tensor:
    """2D Q-criterion ``Q = 0.5(||Omega||^2 - ||S||^2)`` from the model (N,1).

    For 2D incompressible flow this reduces to
    ``Q = -(u_x^2 + v_y^2)/2 - u_y v_x`` (using ``u_x = -v_y``), computed here
    directly from the velocity-gradient tensor.
    """
    X = X.clone().requires_grad_(True)
    f = model.fields(X)
    u, v = f["u"], f["v"]
    u_x, u_y = ops.d(u, X, "x"), ops.d(u, X, "y")
    v_x, v_y = ops.d(v, X, "x"), ops.d(v, X, "y")
    # S_ij symmetric, Omega_ij antisymmetric parts of grad(u).
    s_norm2 = u_x**2 + v_y**2 + 0.5 * (u_y + v_x) ** 2
    omega_norm2 = 0.5 * (v_x - u_y) ** 2
    return 0.5 * (omega_norm2 - s_norm2)


def wall_shear_stress_2d(
    model, X_wall: torch.Tensor, normals: torch.Tensor, mu: float
) -> torch.Tensor:
    """Wall shear stress magnitude at wall points (N,1).

    WSS is the tangential component of the viscous traction
    ``t = mu (grad u + grad u^T) . n``. ``normals`` is ``(N, 2)`` outward unit
    normals; the tangential projection removes the normal component.
    """
    X = X_wall.clone().requires_grad_(True)
    f = model.fields(X)
    u, v = f["u"], f["v"]
    u_x, u_y = ops.d(u, X, "x"), ops.d(u, X, "y")
    v_x, v_y = ops.d(v, X, "x"), ops.d(v, X, "y")
    # 2*mu*S with S the symmetric strain-rate tensor.
    s11 = 2 * u_x
    s22 = 2 * v_y
    s12 = u_y + v_x
    nx, ny = normals[:, 0:1], normals[:, 1:2]
    tx = mu * (s11 * nx + s12 * ny)
    ty = mu * (s12 * nx + s22 * ny)
    # Remove normal component -> tangential traction magnitude (WSS).
    t_dot_n = tx * nx + ty * ny
    tan_x = tx - t_dot_n * nx
    tan_y = ty - t_dot_n * ny
    return torch.sqrt(tan_x**2 + tan_y**2 + 1e-30)


def gradient_field_metrics(pred: torch.Tensor, true: torch.Tensor,
                           name: str) -> Dict[str, float]:
    """RMSE + correlation for a gradient quantity (vorticity/Q/WSS)."""
    return {f"rmse_{name}": rmse(pred, true),
            f"relL2_{name}": relative_l2(pred, true),
            f"corr_{name}": correlation(pred, true)}


def stagnation_volume_agreement(
    speed_pred: torch.Tensor, speed_true: torch.Tensor, threshold: float
) -> Dict[str, float]:
    """Agreement of low-velocity (stagnation) regions.

    Returns the predicted/true stagnation fractions (speed < threshold) and their
    Jaccard (IoU) overlap. Useful as a thrombosis-risk proxy.
    """
    mask_p = speed_pred < threshold
    mask_t = speed_true < threshold
    inter = (mask_p & mask_t).float().sum()
    union = (mask_p | mask_t).float().sum() + 1e-12
    return {
        "stagnation_frac_pred": float(mask_p.float().mean()),
        "stagnation_frac_true": float(mask_t.float().mean()),
        "stagnation_iou": float(inter / union),
    }


# ---------------------------------------------------------------------------
# Physics-free sanity baseline
# ---------------------------------------------------------------------------
def spline_baseline_beam(
    coords_meas: np.ndarray,
    v_beam: np.ndarray,
    coords_query: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate the sparse Doppler beam component onto query points.

    Uses ``scipy.interpolate.griddata`` (piecewise-linear/cubic) as the
    physics-free reference. It can only reproduce the *observed* beam component;
    it cannot recover the cross-beam component or pressure -- which is precisely
    the value-add the PINN must demonstrate over this baseline.

    ``coords_*`` are ``(N, D)`` (spatial + optional time); returns ``(M,)``.
    """
    from scipy.interpolate import griddata  # local import: optional dependency

    interp = griddata(coords_meas, v_beam.reshape(-1), coords_query, method=method)
    # Fill extrapolation NaNs with nearest-neighbour.
    nan = np.isnan(interp)
    if nan.any():
        nn = griddata(coords_meas, v_beam.reshape(-1), coords_query[nan],
                      method="nearest")
        interp[nan] = nn
    return interp


# ---------------------------------------------------------------------------
# End-to-end driver (scaffold)
# ---------------------------------------------------------------------------
def compare_backbones(
    checkpoint_a: str,
    checkpoint_b: Optional[str],
    ibfe_path: str,
    threshold: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:  # pragma: no cover - depends on gated stages
    """Load GT + trained models and tabulate metrics for A vs B (+ baseline).

    TODO (depends on gated training stage + real IBFE loader):
      1. ``frames = load_ibfe_output(ibfe_path)`` (held-out GT).
      2. Load trained ``PINNNet`` checkpoints for Model A and (optionally) B.
      3. Evaluate every metric above on the GT collocation set and wall set.
      4. Compute the spline baseline via :func:`spline_baseline_beam`.
      5. Return ``{"model_a": {...}, "model_b": {...}, "baseline": {...}}``.
    """
    raise NotImplementedError(
        "compare_backbones depends on the (gated) training stage and the real "
        "IBFE loader. The metric primitives above are implemented and tested; "
        "wire this driver once trained checkpoints and IBFE GT are available."
    )
