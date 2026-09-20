"""Turn a real (or synthetic) :class:`IBFEFrames` bundle into a training run.

This is the drop-in that makes an IBAMR/IBFE export flow into the existing spec
trainer without touching the loss or optimisation loop:

* :func:`make_ibfe_batch_builder` -- a ``build_batches(step)`` closure producing
  the dict :class:`~pinnecho.train.composite_loss.CompositeLoss` consumes
  (Doppler ``data``, ``collocation`` + FSI ``forcing``, ``wall`` velocity,
  optional ``traction`` continuity, ``valve`` Dirichlet, ``scalar_inflow`` for
  residence time).
* :func:`build_model_for_ibfe` -- the shared :class:`PINNNet` + a backbone-
  appropriate :class:`CompositeLoss` (2D or 3D, sized from the data).
* :func:`evaluate_ibfe` -- held-out velocity / pressure / vorticity error vs the
  fluid ground truth.
* :func:`train_ibfe` -- a convenience wiring the above to the Adam(+L-BFGS) loop.

Only the beam-projected velocity component is used as data (Doppler); the full
``velocity_fluid`` is treated as held-out ground truth for evaluation.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..models.mlp_pinn import PINNNet
from ..train.composite_loss import AnnealSchedule, CompositeLoss, LossWeights
from ..train.train import TrainConfig, train_model
from .. import evaluate as ev


def input_bounds_from_frames(frames, pad: float = 0.08) -> Tuple[List[float], List[float]]:
    """Per-column ``(lows, highs)`` (spatial + time) from all point sets."""
    dim = frames.spatial_dim
    coords = [frames.coords_fluid, frames.coords_wall,
              frames.coords_mitral, frames.coords_aortic]
    coords = [c for c in coords if c.shape[0] > 0]
    allc = torch.cat(coords, dim=0)
    lo = allc.min(dim=0).values
    hi = allc.max(dim=0).values
    span = (hi - lo).clamp_min(1e-9)
    lo = lo - pad * span
    hi = hi + pad * span
    lo[dim] = min(0.0, float(lo[dim]))  # time starts at 0
    return lo.tolist(), hi.tolist()


def scales_from_frames(frames) -> Dict[str, float]:
    """Reference velocity / pressure / length scales from the data."""
    dim = frames.spatial_dim
    speed = frames.velocity_fluid.norm(dim=1)
    velocity = float(torch.quantile(speed, 0.99)) if speed.numel() else 1.0
    velocity = max(velocity, 1e-6)
    p = frames.pressure_fluid
    p_range = float(p.max() - p.min()) if p.numel() else 0.0
    pressure = max(p_range, frames.rho * velocity ** 2, 1.0)
    coords = frames.coords_fluid[:, :dim]
    length = float((coords.max(dim=0).values - coords.min(dim=0).values).mean()) * 0.5
    length = max(length, 1e-4)
    return {"velocity": velocity, "pressure": pressure, "length": length}


def default_windows_from_frames(frames, n_windows: int = 2) -> List[Tuple[float, ...]]:
    """Virtual transducer positions outside the cavity (apical + lateral)."""
    dim = frames.spatial_dim
    coords = frames.coords_fluid[:, :dim]
    center = coords.mean(dim=0)
    ext = (coords.max(dim=0).values - coords.min(dim=0).values)
    c = center.tolist()
    e = ext.tolist()
    if dim == 2:
        apical = (c[0], c[1] - 2.5 * e[1])
        lateral = (c[0] + 2.5 * e[0], c[1])
        return [apical, lateral][:max(1, n_windows)]
    apical = (c[0], c[1], c[2] - 2.5 * e[2])
    lat_x = (c[0] + 2.5 * e[0], c[1], c[2])
    lat_y = (c[0], c[1] + 2.5 * e[1], c[2])
    return [apical, lat_x, lat_y][:max(1, n_windows)]


def _beam_dirs(X: torch.Tensor, transducer, dim: int) -> torch.Tensor:
    tr = torch.tensor(transducer, dtype=X.dtype, device=X.device).reshape(1, dim)
    d = X[:, :dim] - tr
    return d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)


def build_doppler_from_frames(frames, transducers: Sequence,
                              noise_level: float = 0.05, seed: int = 0):
    """Project ``velocity_fluid`` onto each window's beam -> single-component data.

    Returns ``(data_X, beam_dir, v_beam)`` stacked over all windows.
    """
    dim = frames.spatial_dim
    g = torch.Generator().manual_seed(seed)
    X = frames.coords_fluid
    vel = frames.velocity_fluid
    Xs, beams, vs = [], [], []
    for w in transducers:
        b = _beam_dirs(X, w, dim)
        vb = (vel * b).sum(dim=1, keepdim=True)
        if noise_level > 0:
            vb = vb + noise_level * vb.abs().mean() * torch.randn(
                vb.shape, generator=g, dtype=vb.dtype)
        Xs.append(X.clone()); beams.append(b); vs.append(vb)
    return torch.cat(Xs, 0), torch.cat(beams, 0), torch.cat(vs, 0)


def make_ibfe_batch_builder(frames, transducers: Optional[Sequence] = None,
                            noise_level: float = 0.05, seed: int = 0,
                            n_data: int = 3072, n_col: int = 3072,
                            n_wall: int = 768, n_valve: int = 256,
                            use_traction: bool = False,
                            predict_scalar: bool = False,
                            shuffle_forcing: bool = False):
    """Return a ``build_batches(step)`` closure sourced from ``frames``.

    ``shuffle_forcing`` enables the **forcing-shuffle diagnostic** (see
    ``pinnecho.train.ablation.run_forcing_shuffle`` and ``docs/ABLATIONS.md``
    Ablation 6) on *real* IBFE forcing: the fluid forcing rows are randomly
    permuted (same marginal distribution, trajectory correspondence destroyed).
    Run A/B with this on and off; if the exact-vs-shuffled gap seen on synthetic
    data collapses for real (independently-estimated) FSI forcing, the physics
    benefit is genuine rather than trajectory injection.
    """
    from .ibfe_io import frames_to_dtype
    frames = frames_to_dtype(frames, torch.get_default_dtype())
    dim = frames.spatial_dim
    if transducers is None:
        transducers = default_windows_from_frames(frames, n_windows=2)
    data_X, beam_dir, v_beam = build_doppler_from_frames(
        frames, transducers, noise_level=noise_level, seed=seed)
    g = torch.Generator().manual_seed(seed)

    forcing_fluid = frames.forcing_fluid
    if shuffle_forcing and forcing_fluid.shape[0] > 1:
        gp = torch.Generator().manual_seed(seed + 991)
        perm = torch.randperm(forcing_fluid.shape[0], generator=gp)
        forcing_fluid = forcing_fluid[perm].clone()

    has_valve = frames.coords_mitral.shape[0] > 0

    def _idx(n_total, n):
        if n_total == 0:
            return torch.zeros(0, dtype=torch.long)
        if n >= n_total:
            return torch.arange(n_total)
        return torch.randint(0, n_total, (n,), generator=g)

    def build_batches(step: int) -> Dict[str, dict]:
        di = _idx(data_X.shape[0], n_data)
        ci = _idx(frames.coords_fluid.shape[0], n_col)
        wi = _idx(frames.coords_wall.shape[0], n_wall)
        batches = {
            "data": {"X": data_X[di].clone(), "beam_dir": beam_dir[di],
                     "v_beam": v_beam[di]},
            "collocation": {"X": frames.coords_fluid[ci].clone(),
                            "forcing": forcing_fluid[ci]},
            "wall": {"X": frames.coords_wall[wi].clone(),
                     "u_wall": frames.velocity_wall[wi]},
        }
        if use_traction:
            batches["traction"] = {
                "X": frames.coords_wall[wi].clone(),
                "normals": frames.normals_wall[wi],
                "structure_traction": frames.traction_wall[wi]}
        if has_valve:
            vi = _idx(frames.coords_mitral.shape[0], n_valve)
            batches["valve"] = {"X": frames.coords_mitral[vi].clone(),
                                "u_valve": frames.velocity_mitral[vi]}
            if predict_scalar:
                batches["scalar_inflow"] = {"X": frames.coords_mitral[vi].clone()}
        return batches

    return build_batches


def build_model_for_ibfe(frames, backbone: str = "fsi_informed",
                         predict_scalar: bool = False, use_traction: bool = False,
                         init_seed: int = 0, model_overrides: Optional[dict] = None):
    """Build the shared :class:`PINNNet` + a backbone-appropriate loss for ``frames``."""
    from .ibfe_io import frames_to_dtype
    frames = frames_to_dtype(frames, torch.get_default_dtype())
    dim = frames.spatial_dim
    lows, highs = input_bounds_from_frames(frames)
    s = scales_from_frames(frames)
    m = dict(spatial_dim=dim, width=96, depth=5, activation="tanh",
             fourier_features=0, predict_scalar=predict_scalar)
    if model_overrides:
        m.update(model_overrides)
    torch.manual_seed(init_seed)
    model = PINNNet(input_lows=lows, input_highs=highs,
                    velocity_scale=s["velocity"], pressure_scale=s["pressure"], **m)

    cont_scale = s["length"] / max(s["velocity"], 1e-30)
    mom_scale = s["length"] / max(frames.rho * s["velocity"] ** 2, 1e-30)
    is_fsi = backbone == "fsi_informed"
    weights = LossWeights(data=10.0, pde=1.0, scalar=1.0 if predict_scalar else 0.0,
                          bc=10.0, ic=0.0, periodic=0.0,
                          traction=5.0 if use_traction else 0.0)
    loss_fn = CompositeLoss(
        rho=frames.rho, mu=frames.mu, weights=weights,
        anneal=AnnealSchedule(enabled=True, pde_warmup_frac=0.3),
        forcing="fsi" if is_fsi else None,
        wall_mode="fsi" if is_fsi else "kinematic",
        use_traction=use_traction,
        continuity_scale=cont_scale, momentum_scale=mom_scale,
        traction_scale=mom_scale, scalar_scale=s["length"] / max(s["velocity"], 1e-30),
    )
    return model, loss_fn


@torch.no_grad()
def _vel_pred(model, X, dim):
    f = model.fields(X)
    return torch.cat([f[k] for k in ("u", "v", "w")[:dim]], dim=1)


def evaluate_ibfe(model, frames) -> Dict[str, float]:
    """Held-out velocity / pressure / vorticity error vs the fluid ground truth."""
    from .ibfe_io import frames_to_dtype
    frames = frames_to_dtype(frames, next(model.parameters()).dtype)
    dim = frames.spatial_dim
    X = frames.coords_fluid
    vel_true = frames.velocity_fluid
    vel_pred = _vel_pred(model, X.clone(), dim)
    out: Dict[str, float] = {}
    comps = ("u", "v", "w")[:dim]
    for i, name in enumerate(comps):
        num = torch.linalg.norm(vel_pred[:, i] - vel_true[:, i])
        den = torch.linalg.norm(vel_true[:, i]) + 1e-12
        out[f"vel_relL2_{name}"] = float(num / den)
    out["vel_relL2_speed"] = float(
        torch.linalg.norm(vel_pred - vel_true) / (torch.linalg.norm(vel_true) + 1e-12))

    with torch.no_grad():
        p_pred = model.fields(X.clone())["p"]
    p_pred = p_pred - p_pred.mean()
    p_true = frames.pressure_fluid - frames.pressure_fluid.mean()
    out["pressure_relL2"] = float(
        torch.linalg.norm(p_pred - p_true) / (torch.linalg.norm(p_true) + 1e-12))

    if dim == 2:
        # Ground-truth vorticity is not stored in IBFEFrames; report the model's
        # vorticity magnitude as a smoke check that gradients are non-degenerate.
        vort_pred = ev.vorticity_2d(model, X.clone()).detach()
        out["vorticity_absmax"] = float(vort_pred.abs().max())
    return out


def train_ibfe(frames, backbone: str = "fsi_informed", steps: int = 1500,
               lbfgs_iters: int = 150, lr: float = 2e-3, seed: int = 0,
               predict_scalar: bool = False, use_traction: bool = False,
               transducers: Optional[Sequence] = None, noise_level: float = 0.05,
               model_overrides: Optional[dict] = None, verbose: bool = True,
               shuffle_forcing: bool = False):
    """Train a backbone on ``frames`` end-to-end and return ``(model, metrics)``.

    Set ``shuffle_forcing=True`` to run the Ablation-6 forcing-shuffle diagnostic
    on real IBFE forcing (see ``make_ibfe_batch_builder``).
    """
    model, loss_fn = build_model_for_ibfe(
        frames, backbone=backbone, predict_scalar=predict_scalar,
        use_traction=use_traction, init_seed=seed, model_overrides=model_overrides)
    loss_fn.anneal.total_steps = steps
    build_batches = make_ibfe_batch_builder(
        frames, transducers=transducers, noise_level=noise_level, seed=seed,
        use_traction=use_traction, predict_scalar=predict_scalar,
        shuffle_forcing=shuffle_forcing)
    cfg = TrainConfig(steps=steps, lr=lr, lbfgs_iters=lbfgs_iters,
                      log_every=max(1, steps // 6), seed=seed)
    logger = None
    if verbose:
        def logger(step, row):
            print(f"[ibfe {step:5d}] " + " ".join(
                f"{k}={row[k]:.3e}" for k in ("total", "data", "pde", "bc") if k in row))
    train_model(model, loss_fn, build_batches, cfg, logger=logger)
    return model, evaluate_ibfe(model, frames)
