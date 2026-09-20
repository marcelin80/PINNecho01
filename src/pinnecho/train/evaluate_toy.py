"""Held-out evaluation of a trained toy Model A against the synthetic GT.

Samples fresh interior/wall points across the cardiac cycle (not the training
frames), evaluates the network, and compares to the analytic synthetic-LV ground
truth using the metric primitives in :mod:`pinnecho.evaluate`. Also reports the
physics-free spline baseline on the beam component for context.
"""

from typing import Dict

import numpy as np
import torch

from .. import evaluate as ev
from ..data.doppler import DopplerSampler, project_velocity
from ..physics import operators as ops


class _LVAdapter:
    """Expose the synthetic LV with a ``.fields()`` interface for eval helpers."""

    spatial_dim = 2
    predict_scalar = False

    def __init__(self, lv):
        self.lv = lv

    def fields(self, X):
        uv = self.lv._velocity_from_graph(X)
        return {"u": uv[:, 0:1], "v": uv[:, 1:2], "p": self.lv.pressure(X)}


@torch.no_grad()
def _speed(uv):
    return torch.linalg.norm(uv, dim=1)


def evaluate_toy(model, lv, config, n_interior: int = 4000, n_wall: int = 1000,
                 seed: int = 12345) -> Dict[str, float]:
    dtype = torch.get_default_dtype()
    rng = np.random.default_rng(seed)
    t_values = np.linspace(0.0, config.flow.period, 24)

    X = lv.sample_interior(n_interior, t_values, rng).to(dtype)
    truth = lv.all_fields(X)
    f = model.fields(X.clone())
    vel_pred = torch.cat([f["u"].detach(), f["v"].detach()], dim=1)
    vel_true = torch.cat([truth["u"], truth["v"]], dim=1)

    out: Dict[str, float] = {}
    out.update({f"vel_{k}": v for k, v in ev.velocity_metrics(vel_pred, vel_true).items()
                if k in ("relL2_u", "relL2_v", "relL2_speed")})
    pm = ev.pressure_metrics(f["p"].detach(), truth["p"])
    out["pressure_relL2"] = pm["relL2_p"]
    out["pressure_corr"] = pm["corr_p"]

    # Vorticity (1st-derivative quantity).
    vort_pred = ev.vorticity_2d(model, X.clone()).detach()
    vort_true = lv.vorticity(X.clone().requires_grad_(True)).detach()
    gm = ev.gradient_field_metrics(vort_pred, vort_true, "vort")
    out["vorticity_relL2"] = gm["relL2_vort"]
    out["vorticity_corr"] = gm["corr_vort"]

    # Wall shear stress on fresh wall points.
    wall = lv.sample_wall(n_wall, t_values, rng)
    Xw, nrm = wall["X"].to(dtype), wall["normal"].to(dtype)
    wss_pred = ev.wall_shear_stress_2d(model, Xw.clone(), nrm, config.flow.viscosity).detach()
    wss_true = ev.wall_shear_stress_2d(_LVAdapter(lv), Xw.clone(), nrm,
                                       config.flow.viscosity).detach()
    wm = ev.gradient_field_metrics(wss_pred, wss_true, "wss")
    out["wss_relL2"] = wm["relL2_wss"]
    out["wss_corr"] = wm["corr_wss"]

    # Physics-free spline baseline on the beam component (context for value-add).
    try:
        sampler = DopplerSampler(config.doppler)
        meas = sampler.sample(lv, np.random.default_rng(config.doppler.seed))
        beam_q = sampler.beam_directions(X)
        v_beam_true = project_velocity(truth["u"], truth["v"], beam_q).squeeze(1).numpy()
        base = ev.spline_baseline_beam(
            meas["X"][:, :3].numpy(), meas["doppler"].squeeze(1).numpy(),
            X[:, :3].numpy(), method="linear",
        )
        num = np.linalg.norm(base - v_beam_true)
        den = np.linalg.norm(v_beam_true) + 1e-12
        out["baseline_beam_relL2"] = float(num / den)
    except Exception:
        pass

    return out
