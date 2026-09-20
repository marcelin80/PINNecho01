"""Accuracy metrics comparing a trained PINN against the synthetic ground truth.

The headline scientific question is whether the FSI-informed backbone recovers
velocity-*gradient* quantities (vorticity, wall shear stress) and pressure
better than the baseline. This module therefore reports, per field:

* relative L2 error of velocity components and speed,
* relative L2 error of pressure (mean-removed -- defined up to a constant),
* relative L2 error of vorticity (a first-derivative quantity),
* relative L2 error of wall shear stress (WSS) evaluated on the endocardium.
"""

from typing import Dict

import numpy as np
import torch

from ..config import Config
from ..data.synthetic_lv import SyntheticLVFSI
from ..physics import operators as ops


def relative_l2(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-12) -> float:
    num = torch.linalg.norm((pred - true).reshape(-1))
    den = torch.linalg.norm(true.reshape(-1)).clamp_min(eps)
    return float(num / den)


def velocity_gradient(model, X: torch.Tensor):
    """Return ``u, v, p`` and the velocity gradient columns at ``X`` (autograd)."""
    Xg = X.clone().requires_grad_(True)
    u, v, p = model(Xg)
    gu = ops.grad(u, Xg)
    gv = ops.grad(v, Xg)
    return u, v, p, gu, gv, Xg


def _eval_times(cfg: Config, n_frames: int) -> np.ndarray:
    return np.linspace(0.0, cfg.flow.period, n_frames)


@torch.no_grad()
def _predict_uvp(model, X):
    u, v, p = model(X)
    return u, v, p


def evaluate_fields(model, lv: SyntheticLVFSI, cfg: Config,
                    n_points: int = 6000, n_frames: int = 16,
                    seed: int = 123) -> Dict[str, float]:
    """Field-wise relative-L2 errors over interior points across the cycle."""
    rng = np.random.default_rng(seed)
    X = lv.sample_interior(n_points, _eval_times(cfg, n_frames), rng)
    X = X.to(next(model.parameters()).dtype)

    truth = lv.all_fields(X)

    # Predicted velocity + pressure (no grad needed for these).
    u, v, p = _predict_uvp(model, X)
    speed_pred = torch.sqrt(u ** 2 + v ** 2)
    speed_true = truth["speed"]

    # Predicted vorticity (needs autograd).
    Xg = X.clone().requires_grad_(True)
    up, vp, _ = model(Xg)
    vort_pred = ops.curl_z(up, vp, Xg).detach()

    # Pressure is defined up to a constant -> compare mean-removed fields.
    p_pred_c = p - p.mean()
    p_true_c = truth["p"] - truth["p"].mean()

    return {
        "rel_l2_u": relative_l2(u, truth["u"]),
        "rel_l2_v": relative_l2(v, truth["v"]),
        "rel_l2_speed": relative_l2(speed_pred, speed_true),
        "rel_l2_pressure": relative_l2(p_pred_c, p_true_c),
        "rel_l2_vorticity": relative_l2(vort_pred, truth["vorticity"]),
    }


def evaluate_wss(model, lv: SyntheticLVFSI, cfg: Config,
                 n_wall: int = 2000, n_frames: int = 16,
                 seed: int = 321) -> Dict[str, float]:
    """Relative-L2 error of the wall shear stress vector on the endocardium.

    WSS is the tangential part of the viscous traction
    ``t = mu (grad u + grad u^T) n`` on the wall.
    """
    rng = np.random.default_rng(seed)
    wall = lv.sample_wall(n_wall, _eval_times(cfg, n_frames), rng)
    X = wall["X"].to(next(model.parameters()).dtype)
    n = wall["normal"].to(X.dtype)
    mu = cfg.flow.viscosity

    def wss_vector(u, v, gu, gv):
        # Symmetric velocity-gradient (rate-of-strain) times normal.
        u_x, u_y = gu[:, 0:1], gu[:, 1:2]
        v_x, v_y = gv[:, 0:1], gv[:, 1:2]
        s11 = 2.0 * u_x
        s12 = u_y + v_x
        s22 = 2.0 * v_y
        tx = mu * (s11 * n[:, 0:1] + s12 * n[:, 1:2])
        ty = mu * (s12 * n[:, 0:1] + s22 * n[:, 1:2])
        # Remove normal component -> tangential traction (WSS).
        tn = tx * n[:, 0:1] + ty * n[:, 1:2]
        wss_x = tx - tn * n[:, 0:1]
        wss_y = ty - tn * n[:, 1:2]
        return torch.cat([wss_x, wss_y], dim=1)

    # Predicted.
    u, v, _, gu, gv, _ = velocity_gradient(model, X)
    wss_pred = wss_vector(u, v, gu, gv).detach()

    # True (from the analytic field).
    Xg = X.clone().requires_grad_(True)
    uv = lv.velocity(Xg)
    gu_t = ops.grad(uv[:, 0:1], Xg)
    gv_t = ops.grad(uv[:, 1:2], Xg)
    wss_true = wss_vector(uv[:, 0:1], uv[:, 1:2], gu_t, gv_t).detach()

    return {
        "rel_l2_wss": relative_l2(wss_pred, wss_true),
        "wss_true_rms": float(torch.sqrt((wss_true ** 2).sum(dim=1).mean())),
    }
