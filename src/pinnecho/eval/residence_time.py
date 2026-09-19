"""Blood residence time (RT) via Lagrangian backward integration.

Residence time -- how long a fluid parcel has dwelled inside the LV cavity --
is a clinically relevant marker of stasis / thrombus risk and is a demanding
test of a reconstructed velocity field because it integrates the field over
time. We estimate it by tracing parcels *backwards* in time from seed points
and accumulating the time spent inside the (moving) cavity until the parcel
exits or a cap is reached.

The same estimator is applied to the ground-truth field and to the field
predicted by a trained PINN so their RT maps can be compared directly.
"""

from typing import Callable, Dict

import numpy as np
import torch

from ..config import Config
from ..data.synthetic_lv import SyntheticLVFSI


def _inside_mask(lv: SyntheticLVFSI, xy: np.ndarray, t: float) -> np.ndarray:
    t_tensor = torch.tensor([[t]], dtype=lv.dtype)
    rx, ry = lv.semi_axes(t_tensor)
    rx = float(rx); ry = float(ry)
    xi = (xy[:, 0] - lv.geom.center_x) / rx
    eta = (xy[:, 1] - lv.geom.center_y) / ry
    return (xi ** 2 + eta ** 2) <= 1.0


def residence_time_map(vel_fn: Callable[[np.ndarray], np.ndarray],
                       lv: SyntheticLVFSI, seeds_xy: np.ndarray, t0: float,
                       max_time: float, dt: float) -> np.ndarray:
    """Backward-integrate parcels from ``seeds_xy`` at time ``t0``.

    ``vel_fn`` maps an ``(M, 3)`` array ``[x, y, t]`` to an ``(M, 2)`` velocity.
    Returns the accumulated in-cavity residence time per seed (shape ``(M,)``).
    """
    pos = np.asarray(seeds_xy, dtype=float).copy()
    m = pos.shape[0]
    rt = np.zeros(m)
    active = _inside_mask(lv, pos, t0)
    t = float(t0)
    n_steps = int(np.ceil(max_time / dt))

    for _ in range(n_steps):
        if not active.any():
            break
        query = np.concatenate([pos, np.full((m, 1), t)], axis=1)
        uv = vel_fn(query)
        # Backward Euler-explicit step (upwind-free, cavity is smooth).
        pos_new = pos - dt * uv
        t_new = t - dt
        rt[active] += dt
        active = active & _inside_mask(lv, pos_new, t_new)
        pos = pos_new
        t = t_new

    return np.minimum(rt, max_time)


def _model_velocity_fn(model, dtype):
    def fn(query: np.ndarray) -> np.ndarray:
        X = torch.as_tensor(query, dtype=dtype)
        with torch.no_grad():
            u, v, _ = model(X)
        return torch.cat([u, v], dim=1).cpu().numpy()
    return fn


def _true_velocity_fn(lv: SyntheticLVFSI):
    def fn(query: np.ndarray) -> np.ndarray:
        X = torch.as_tensor(query, dtype=lv.dtype).clone().requires_grad_(True)
        uv = lv.velocity(X)
        return uv.detach().cpu().numpy()
    return fn


def evaluate_residence_time(model, lv: SyntheticLVFSI, cfg: Config,
                            n_seeds: int = 400, t0_frac: float = 0.5,
                            n_cycles: float = 1.5, seed: int = 7) -> Dict[str, float]:
    """Compare predicted vs true residence-time maps at a fixed frame."""
    rng = np.random.default_rng(seed)
    t0 = t0_frac * cfg.flow.period
    seeds = lv.sample_interior(n_seeds, np.array([t0]), rng)[:, 0:2].cpu().numpy()

    max_time = n_cycles * cfg.flow.period
    dt = cfg.flow.period / 200.0
    dtype = next(model.parameters()).dtype

    rt_true = residence_time_map(_true_velocity_fn(lv), lv, seeds, t0, max_time, dt)
    rt_pred = residence_time_map(_model_velocity_fn(model, dtype), lv, seeds, t0, max_time, dt)

    rt_true_t = torch.as_tensor(rt_true)
    rt_pred_t = torch.as_tensor(rt_pred)
    num = torch.linalg.norm(rt_pred_t - rt_true_t)
    den = torch.linalg.norm(rt_true_t).clamp_min(1e-12)

    return {
        "rel_l2_residence_time": float(num / den),
        "mean_rt_true": float(rt_true.mean()),
        "mean_rt_pred": float(rt_pred.mean()),
    }
