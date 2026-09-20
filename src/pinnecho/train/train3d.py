"""End-to-end 3D toy validation for the spec PINN.

Mirrors the 2D toy driver but on the volume-preserving ellipsoid ground truth
(:class:`~pinnecho.data.synthetic_lv_3d.SyntheticLV3D`). Reuses the
dimension-general :class:`~pinnecho.train.composite_loss.CompositeLoss` and the
generic Adam(+L-BFGS) :func:`~pinnecho.train.train.train_model` loop; the only
3D-specific pieces are the ground truth, the multi-window Doppler projection and
the input bounds.

Doppler observability: a single beam constrains one velocity component, so in 3D
we use three well-separated acoustic windows (apical + two lateral) to make the
full 3-component velocity recoverable, exactly as the 2D case uses dual windows.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..models.mlp_pinn import PINNNet
from .composite_loss import AnnealSchedule, CompositeLoss, LossWeights
from .train import TrainConfig, train_model
from ..data.synthetic_lv_3d import SyntheticLV3D
from .. import evaluate as ev


def input_bounds_3d(geometry, flow, pad: float = 1.15):
    strain = geometry.strain_amplitude
    inv = 1.0 / max(1e-6, np.sqrt(1.0 - strain))
    rx_max = geometry.r0_x * inv
    ry_max = geometry.r0_y * inv
    rz_max = geometry.r0_z * (1.0 + strain)
    lows = [geometry.center_x - pad * rx_max, geometry.center_y - pad * ry_max,
            geometry.center_z - pad * rz_max, 0.0]
    highs = [geometry.center_x + pad * rx_max, geometry.center_y + pad * ry_max,
             geometry.center_z + pad * rz_max, float(flow.period)]
    return lows, highs


def default_windows_3d(geometry) -> List[Tuple[float, float, float]]:
    cx, cy, cz = geometry.center_x, geometry.center_y, geometry.center_z
    return [
        (cx, cy, cz - 2.5 * geometry.r0_z),   # apical long-axis
        (cx + 2.5 * geometry.r0_x, cy, cz),   # lateral (x)
        (cx, cy + 2.5 * geometry.r0_y, cz),   # lateral (y)
    ]


def _beam_dirs(X: torch.Tensor, transducer) -> torch.Tensor:
    tr = torch.tensor(transducer, dtype=X.dtype, device=X.device).reshape(1, 3)
    d = X[:, :3] - tr
    return d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)


@dataclass
class Toy3DDataset:
    col_X: torch.Tensor
    col_forcing: torch.Tensor
    wall_X: torch.Tensor
    wall_velocity: torch.Tensor
    data_X: torch.Tensor
    data_beam: torch.Tensor
    data_v: torch.Tensor
    velocity: float
    pressure: float
    length: float
    density: float
    viscosity: float


def build_toy_dataset_3d(config, n_interior: int = 6000, n_wall: int = 1200,
                         n_data_per_frame: int = 250, n_frames: int = 12,
                         seed: int = 0) -> Tuple[Toy3DDataset, SyntheticLV3D]:
    dtype = torch.get_default_dtype()
    lv = SyntheticLV3D(config.geometry, config.flow, dtype=dtype)
    rng = np.random.default_rng(seed)
    t_values = np.linspace(0.0, config.flow.period, n_frames)

    # Interior collocation + FSI forcing (ground-truth momentum residual).
    col_X = lv.sample_interior(n_interior, t_values, rng)
    col_forcing = lv.forcing(col_X.clone().requires_grad_(True)).detach()

    # Wall points + accurate wall velocity.
    wall = lv.sample_wall(n_wall, t_values, rng)

    # Multi-window Doppler measurements (single-component per sample).
    windows = default_windows_3d(config.geometry)
    Xs, beams, vs = [], [], []
    noise = float(getattr(config.doppler, "noise_level", 0.05))
    for w in windows:
        Xw = lv.sample_interior(n_data_per_frame * n_frames, t_values, rng)
        truth = lv.all_fields(Xw)
        vel = torch.cat([truth["u"], truth["v"], truth["w"]], dim=1)
        b = _beam_dirs(Xw, w)
        vb = (vel * b).sum(dim=1, keepdim=True)
        if noise > 0:
            vb = vb + noise * vb.abs().mean() * torch.randn_like(vb)
        Xs.append(Xw); beams.append(b); vs.append(vb)
    data_X = torch.cat(Xs, dim=0)
    data_beam = torch.cat(beams, dim=0)
    data_v = torch.cat(vs, dim=0)

    stats = lv.field_statistics(seed=seed)
    velocity_scale = max(stats["speed_max"], lv.l_mean / config.flow.period)
    pressure_scale = max(config.flow.pressure_amplitude,
                         config.flow.density * velocity_scale ** 2)

    ds = Toy3DDataset(
        col_X=col_X, col_forcing=col_forcing,
        wall_X=wall["X"], wall_velocity=wall["wall_velocity"],
        data_X=data_X, data_beam=data_beam, data_v=data_v,
        velocity=velocity_scale, pressure=pressure_scale, length=lv.l_mean,
        density=config.flow.density, viscosity=config.flow.viscosity,
    )
    return ds, lv


def build_toy_model_3d(config, dataset: Toy3DDataset, backbone: str = "baseline",
                       init_seed: int = 0, model_overrides: Optional[dict] = None):
    lows, highs = input_bounds_3d(config.geometry, config.flow)
    m = dict(spatial_dim=3, width=96, depth=5, activation="tanh",
             fourier_features=0, predict_scalar=False)
    if model_overrides:
        m.update(model_overrides)
    torch.manual_seed(init_seed)
    model = PINNNet(input_lows=lows, input_highs=highs,
                    velocity_scale=dataset.velocity, pressure_scale=dataset.pressure,
                    **m)

    cont_scale = dataset.length / max(dataset.velocity, 1e-30)
    mom_scale = dataset.length / max(dataset.density * dataset.velocity ** 2, 1e-30)
    is_fsi = backbone == "fsi_informed"
    weights = LossWeights(data=10.0, pde=1.0, scalar=0.0, bc=10.0, ic=0.0,
                          periodic=0.0, traction=0.0)
    loss_fn = CompositeLoss(
        rho=dataset.density, mu=dataset.viscosity, weights=weights,
        anneal=AnnealSchedule(enabled=True, pde_warmup_frac=0.3),
        forcing="fsi" if is_fsi else None,
        wall_mode="fsi" if is_fsi else "kinematic",
        continuity_scale=cont_scale, momentum_scale=mom_scale,
    )
    return model, loss_fn


def make_batch_builder_3d(dataset: Toy3DDataset, seed: int = 0,
                          n_data: int = 3072, n_col: int = 3072, n_wall: int = 768):
    g = torch.Generator().manual_seed(seed)

    def _idx(n_total, n):
        if n >= n_total:
            return torch.arange(n_total)
        return torch.randint(0, n_total, (n,), generator=g)

    def build_batches(step: int) -> Dict[str, dict]:
        di = _idx(dataset.data_X.shape[0], n_data)
        ci = _idx(dataset.col_X.shape[0], n_col)
        wi = _idx(dataset.wall_X.shape[0], n_wall)
        return {
            "data": {"X": dataset.data_X[di].clone(),
                     "beam_dir": dataset.data_beam[di],
                     "v_beam": dataset.data_v[di]},
            "collocation": {"X": dataset.col_X[ci].clone(),
                            "forcing": dataset.col_forcing[ci]},
            "wall": {"X": dataset.wall_X[wi].clone(),
                     "u_wall": dataset.wall_velocity[wi]},
        }

    return build_batches


@torch.no_grad()
def _vel_pred(model, X):
    f = model.fields(X)
    return torch.cat([f["u"], f["v"], f["w"]], dim=1)


def evaluate_3d(model, lv: SyntheticLV3D, config, n_interior: int = 6000,
                seed: int = 999) -> Dict[str, float]:
    dtype = torch.get_default_dtype()
    rng = np.random.default_rng(seed)
    t_values = np.linspace(0.0, config.flow.period, 16)
    X = lv.sample_interior(n_interior, t_values, rng).to(dtype)
    truth = lv.all_fields(X)

    vel_pred = _vel_pred(model, X.clone())
    vel_true = torch.cat([truth["u"], truth["v"], truth["w"]], dim=1)
    out: Dict[str, float] = {}
    for i, name in enumerate("uvw"):
        num = torch.linalg.norm(vel_pred[:, i] - vel_true[:, i])
        den = torch.linalg.norm(vel_true[:, i]) + 1e-12
        out[f"vel_relL2_{name}"] = float(num / den)
    num = torch.linalg.norm(vel_pred - vel_true)
    den = torch.linalg.norm(vel_true) + 1e-12
    out["vel_relL2_speed"] = float(num / den)

    with torch.no_grad():
        p_pred = model.fields(X.clone())["p"]
    p_pred = p_pred - p_pred.mean()
    p_true = truth["p"] - truth["p"].mean()
    out["pressure_relL2"] = float(torch.linalg.norm(p_pred - p_true) /
                                  (torch.linalg.norm(p_true) + 1e-12))

    # Vorticity magnitude (first-derivative quantity).
    Xg = X.clone().requires_grad_(True)
    f = model.fields(Xg)
    from ..physics import operators as ops
    gu, gv, gw = ops.grad(f["u"], Xg), ops.grad(f["v"], Xg), ops.grad(f["w"], Xg)
    wx = gw[:, 1:2] - gv[:, 2:3]
    wy = gu[:, 2:3] - gw[:, 0:1]
    wz = gv[:, 0:1] - gu[:, 1:2]
    vort_pred = torch.cat([wx, wy, wz], dim=1).detach()
    vort_true = truth["vorticity"]
    out["vorticity_relL2"] = float(torch.linalg.norm(vort_pred - vort_true) /
                                   (torch.linalg.norm(vort_true) + 1e-12))
    return out


def train_toy_3d(config, backbone: str = "fsi_informed", steps: int = 1500,
                 lbfgs_iters: int = 150, lr: float = 2e-3, seed: int = 0,
                 model_overrides: Optional[dict] = None, verbose: bool = True):
    ds, lv = build_toy_dataset_3d(config, seed=seed)
    model, loss_fn = build_toy_model_3d(config, ds, backbone=backbone,
                                        init_seed=seed, model_overrides=model_overrides)
    loss_fn.anneal.total_steps = steps
    build_batches = make_batch_builder_3d(ds, seed=seed)
    cfg = TrainConfig(steps=steps, lr=lr, lbfgs_iters=lbfgs_iters,
                      log_every=max(1, steps // 6), seed=seed)
    logger = None
    if verbose:
        def logger(step, row):
            print(f"[3d {step:5d}] " + " ".join(
                f"{k}={row[k]:.3e}" for k in ("total", "data", "pde", "bc") if k in row))
    train_model(model, loss_fn, build_batches, cfg, logger=logger)
    metrics = evaluate_3d(model, lv, config)
    return model, metrics, lv


def main() -> None:  # pragma: no cover - CLI wiring
    import argparse
    from ..config import load_config, Config

    ap = argparse.ArgumentParser(description="3D toy PINN validation.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--backbone", default="fsi_informed",
                    choices=["baseline", "fsi_informed"])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lbfgs-iters", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64 if args.dtype == "float64" else torch.float32)
    config = load_config(args.config) if args.config else Config()
    _model, metrics, _lv = train_toy_3d(
        config, backbone=args.backbone, steps=args.steps,
        lbfgs_iters=args.lbfgs_iters, seed=args.seed)
    print("\n3D held-out reconstruction metrics (relative L2):")
    for k, v in metrics.items():
        print(f"  {k:22s} {v:.4f}")
