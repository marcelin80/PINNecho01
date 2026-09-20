"""Training entry point (spec-aligned) for the kinematic baseline (Model A).

This is a scaffold with real signatures and a functional Adam(+optional L-BFGS)
loop over the :class:`~pinnecho.train.composite_loss.CompositeLoss`. The data
assembly for the toy 2D case (and later the real IBFE problem) is factored behind
a ``build_batches`` callable so the loop itself is backbone- and data-agnostic.

Staging note
------------
Per the project plan, actually *running* Model A end-to-end on the toy case is the
next stage and is gated on explicit confirmation. Model B (FSI wall velocity /
traction continuity) must not be started until Model A validates. The Model B
switch here is a ``forcing="fsi"`` flag wired through to the loss, but the
corresponding BC helpers still raise ``NotImplementedError`` by design.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

from ..models.mlp_pinn import PINNNet
from .composite_loss import AnnealSchedule, CompositeLoss, LossWeights

BatchBuilder = Callable[[int], Dict[str, dict]]


@dataclass
class TrainConfig:
    steps: int = 10000
    lr: float = 2e-3
    lbfgs_iters: int = 0
    log_every: int = 500
    seed: int = 0
    grad_clip: Optional[float] = None


def build_model(model_cfg: dict) -> PINNNet:
    """Instantiate the shared network from a config dict (identical for A and B)."""
    return PINNNet(
        spatial_dim=model_cfg.get("spatial_dim", 2),
        width=model_cfg.get("width", 256),
        depth=model_cfg.get("depth", 8),
        activation=model_cfg.get("activation", "tanh"),
        fourier_features=model_cfg.get("fourier_features", 0),
        fourier_scales=tuple(model_cfg.get("fourier_scales", (1.0, 10.0, 100.0))),
        predict_scalar=model_cfg.get("predict_scalar", True),
        input_lows=model_cfg.get("input_lows"),
        input_highs=model_cfg.get("input_highs"),
    )


def train_model(
    model: PINNNet,
    loss_fn: CompositeLoss,
    build_batches: BatchBuilder,
    cfg: TrainConfig,
    logger: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Dict[str, list]:
    """Run Adam then optional L-BFGS polishing. Returns per-term loss history.

    ``build_batches(step)`` returns the ``batches`` dict expected by
    :class:`CompositeLoss` (see its docstring). It may resample collocation
    points each call.
    """
    torch.manual_seed(cfg.seed)
    history: Dict[str, list] = {}

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)

    for step in range(cfg.steps):
        batches = build_batches(step)
        opt.zero_grad(set_to_none=True)
        terms = loss_fn(model, batches, step=step)
        terms["total"].backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            row = {k: float(v.detach().item()) for k, v in terms.items()
                   if v.dim() == 0}
            for k, val in row.items():
                history.setdefault(k, []).append(val)
            if logger:
                logger(step, row)

    if cfg.lbfgs_iters > 0:
        _lbfgs_polish(model, loss_fn, build_batches, cfg)

    return history


def _lbfgs_polish(model, loss_fn, build_batches, cfg: TrainConfig) -> None:
    """L-BFGS refinement on a fixed batch (quasi-Newton polishing)."""
    batches = build_batches(cfg.steps)  # one fixed batch for a deterministic closure
    opt = torch.optim.LBFGS(
        model.parameters(), max_iter=cfg.lbfgs_iters,
        line_search_fn="strong_wolfe", tolerance_grad=1e-9,
    )

    def closure():
        opt.zero_grad(set_to_none=True)
        terms = loss_fn(model, batches, step=cfg.steps)
        terms["total"].backward()
        return terms["total"]

    opt.step(closure)


def input_bounds(geometry, flow, pad: float = 1.15):
    """Domain bounds ``(lows, highs)`` for input normalisation of the toy cavity.

    The ellipse semi-axes vary as ``rx = r0_x / s``, ``ry = r0_y * s`` with
    ``s in [1 - strain, 1 + strain]``, so the extreme extents are
    ``r0_x / (1 - strain)`` and ``r0_y * (1 + strain)``.
    """
    strain = geometry.strain_amplitude
    rx_max = geometry.r0_x / max(1e-6, 1.0 - strain)
    ry_max = geometry.r0_y * (1.0 + strain)
    lows = [geometry.center_x - pad * rx_max, geometry.center_y - pad * ry_max, 0.0]
    highs = [geometry.center_x + pad * rx_max, geometry.center_y + pad * ry_max,
             float(flow.period)]
    return lows, highs


def build_toy_model_a(config, model_overrides: Optional[dict] = None):
    """Assemble ``(model, loss_fn, dataset, lv)`` for the toy 2D Model A run.

    Uses the synthetic LV FSI generator as a stand-in for real IBFE output. The
    data term sees only the single-component Doppler signal; the physics is the
    baseline incompressible NS (forcing = 0). Residual/output scales come from
    the dataset's reference scales so everything is O(1) during optimisation.
    """
    from ..data.dataset import build_dataset
    from ..data.synthetic_lv import SyntheticLVFSI

    lv = SyntheticLVFSI(config.geometry, config.flow, dtype=torch.get_default_dtype())
    dataset = build_dataset(config, lv=lv, dtype=torch.get_default_dtype())
    s = dataset.scales

    lows, highs = input_bounds(config.geometry, config.flow)
    m = dict(spatial_dim=2, width=96, depth=5, activation="tanh",
             fourier_features=0, predict_scalar=False)
    if model_overrides:
        m.update(model_overrides)
    model = PINNNet(
        input_lows=lows, input_highs=highs,
        velocity_scale=s.velocity, pressure_scale=s.pressure,
        **m,
    )

    cont_scale = s.length / max(s.velocity, 1e-30)
    mom_scale = s.length / max(dataset.density * s.velocity ** 2, 1e-30)
    loss_fn = CompositeLoss(
        rho=dataset.density, mu=dataset.viscosity,
        weights=LossWeights(data=10.0, pde=1.0, scalar=0.0, bc=10.0,
                            ic=0.0, periodic=0.0),
        anneal=AnnealSchedule(enabled=True, pde_warmup_frac=0.3),
        forcing=None,  # Model A baseline
        continuity_scale=cont_scale, momentum_scale=mom_scale,
    )
    return model, loss_fn, dataset, lv


def make_toy_batch_builder(dataset, seed: int = 0,
                           n_data: int = 2048, n_col: int = 2048,
                           n_wall: int = 512):
    """Return a ``build_batches(step)`` closure that subsamples the fixed sets."""
    g = torch.Generator().manual_seed(seed)

    def _idx(n_total, n):
        if n >= n_total:
            return torch.arange(n_total)
        return torch.randperm(n_total, generator=g)[:n]

    def build_batches(step: int):
        di = _idx(dataset.meas_X.shape[0], n_data)
        ci = _idx(dataset.col_X.shape[0], n_col)
        wi = _idx(dataset.wall_X.shape[0], n_wall)
        return {
            "data": {
                "X": dataset.meas_X[di].clone(),
                "beam_dir": dataset.meas_beam[di],
                "v_beam": dataset.meas_doppler[di],
            },
            "collocation": {
                "X": dataset.col_X[ci].clone(),
                "forcing": dataset.col_forcing[ci],
            },
            "wall": {
                "X": dataset.wall_X[wi].clone(),
                "u_wall": dataset.wall_velocity[wi],
            },
        }

    return build_batches


def main() -> None:  # pragma: no cover - CLI wiring
    """CLI: train Model A end-to-end on the toy 2D case with the spec modules."""
    import argparse
    from ..config import load_config, Config

    parser = argparse.ArgumentParser(description="Train Model A (toy 2D case).")
    parser.add_argument("--config", default=None, help="YAML config (geometry/flow).")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lbfgs-iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    config = load_config(args.config) if args.config else Config()

    model, loss_fn, dataset, lv = build_toy_model_a(config)
    loss_fn.anneal.total_steps = args.steps
    build_batches = make_toy_batch_builder(dataset, seed=args.seed)
    cfg = TrainConfig(steps=args.steps, lr=args.lr, lbfgs_iters=args.lbfgs_iters,
                      log_every=max(1, args.steps // 10), seed=args.seed)

    def logger(step, row):
        print(f"[{step:5d}] " + " ".join(f"{k}={row[k]:.4e}" for k in
              ("total", "data", "pde", "bc") if k in row))

    print("Training Model A (baseline, kinematic no-slip) on the toy 2D case...")
    train_model(model, loss_fn, build_batches, cfg, logger=logger)

    from ..train.evaluate_toy import evaluate_toy  # local to avoid cycles
    metrics = evaluate_toy(model, lv, config)
    print("\nHeld-out reconstruction metrics (relative L2 unless noted):")
    for k, v in metrics.items():
        print(f"  {k:28s} {v:.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()
