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


def build_toy_dataset(config, wall_tracking_bias: float = -0.08,
                      wall_tracking_noise: float = 0.10, seed: int = 0):
    """Build the shared synthetic-LV dataset + generator (identical for A and B).

    Precomputes, at the wall points:
    * the true fluid-interface traction (structure == fluid for the exact
      synthetic solution) -- the traction-continuity target;
    * ``wall_velocity_fsi``  -- the accurate FSI structural wall velocity (== the
      true endocardial velocity of the manufactured field), used by Model B;
    * ``wall_velocity_kin``  -- a contour-tracking estimate of the wall velocity
      corrupted by a systematic bias + noise (segmentation / finite-difference of
      tracked contours typically *under*-estimates peak wall speed), used by
      Model A. This makes the kinematic-vs-FSI wall BC distinction real rather
      than coincident, as in the true FSI setting.
    """
    from ..data.dataset import build_dataset
    from ..data.synthetic_lv import SyntheticLVFSI
    from ..bc.boundary_conditions import fluid_traction_2d

    dtype = torch.get_default_dtype()
    lv = SyntheticLVFSI(config.geometry, config.flow, dtype=dtype)
    dataset = build_dataset(config, lv=lv, dtype=dtype)

    Xg = dataset.wall_X.clone().requires_grad_(True)
    uv = lv._velocity_from_graph(Xg)
    p = lv.pressure(Xg)
    t_true = fluid_traction_2d(uv[:, 0:1], uv[:, 1:2], p, Xg,
                              dataset.wall_normal, config.flow.viscosity).detach()
    dataset.extras["wall_traction"] = t_true

    # Diverge the two wall velocities. FSI (true) vs kinematic (tracking error).
    v_fsi = dataset.wall_velocity
    rms = float(v_fsi.pow(2).mean().sqrt().clamp_min(1e-9))
    g = torch.Generator().manual_seed(seed + 7)
    noise = torch.randn(v_fsi.shape, generator=g, dtype=dtype) * (wall_tracking_noise * rms)
    v_kin = v_fsi * (1.0 + wall_tracking_bias) + noise
    dataset.extras["wall_velocity_fsi"] = v_fsi
    dataset.extras["wall_velocity_kin"] = v_kin
    return dataset, lv


def build_toy_model(config, dataset, backbone: str = "baseline",
                    model_overrides: Optional[dict] = None, init_seed: int = 0,
                    use_traction: bool = False):
    """Assemble ``(model, loss_fn)`` for a backbone on the shared toy dataset.

    ``backbone`` is ``"baseline"`` (Model A, forcing = 0, kinematic wall) or
    ``"fsi_informed"`` (Model B: FSI momentum forcing + FSI wall velocity, and
    optionally the traction-continuity penalty). The network architecture is
    identical for both; ``init_seed`` seeds the weight init so the comparison
    starts both backbones from the *same* parameters.
    """
    s = dataset.scales
    lows, highs = input_bounds(config.geometry, config.flow)
    m = dict(spatial_dim=2, width=96, depth=5, activation="tanh",
             fourier_features=0, predict_scalar=False)
    if model_overrides:
        m.update(model_overrides)

    torch.manual_seed(init_seed)  # identical initialisation across backbones
    model = PINNNet(
        input_lows=lows, input_highs=highs,
        velocity_scale=s.velocity, pressure_scale=s.pressure,
        **m,
    )

    cont_scale = s.length / max(s.velocity, 1e-30)
    mom_scale = s.length / max(dataset.density * s.velocity ** 2, 1e-30)
    is_fsi = backbone == "fsi_informed"
    weights = LossWeights(data=10.0, pde=1.0, scalar=0.0, bc=10.0,
                          ic=0.0, periodic=0.0,
                          traction=5.0 if use_traction else 0.0)
    loss_fn = CompositeLoss(
        rho=dataset.density, mu=dataset.viscosity,
        weights=weights,
        anneal=AnnealSchedule(enabled=True, pde_warmup_frac=0.3),
        forcing="fsi" if is_fsi else None,
        wall_mode="fsi" if is_fsi else "kinematic",
        use_traction=use_traction,
        continuity_scale=cont_scale, momentum_scale=mom_scale,
        traction_scale=mom_scale,
    )
    return model, loss_fn


def build_toy_model_a(config, model_overrides: Optional[dict] = None):
    """Convenience: shared dataset + Model A ``(model, loss_fn, dataset, lv)``."""
    dataset, lv = build_toy_dataset(config)
    model, loss_fn = build_toy_model(config, dataset, backbone="baseline",
                                     model_overrides=model_overrides)
    return model, loss_fn, dataset, lv


def make_toy_batch_builder(dataset, seed: int = 0,
                           n_data: int = 2048, n_col: int = 2048,
                           n_wall: int = 512, with_traction: bool = False,
                           wall_target: str = "fsi",
                           traction_key: str = "wall_traction",
                           forcing_key: Optional[str] = None):
    """Return a ``build_batches(step)`` closure that subsamples the fixed sets.

    ``wall_target`` selects the wall-velocity BC target: ``"kinematic"`` (the
    contour-tracking estimate, for Model A) or ``"fsi"`` (the accurate FSI wall
    velocity, for Model B). Falls back to ``dataset.wall_velocity`` if the
    diverged velocities are not present.

    ``traction_key`` selects which ``dataset.extras`` entry supplies the
    traction-continuity target. Defaults to the exact ``"wall_traction"``; the
    ablation driver stores perturbed copies under other keys to run the
    traction-uncertainty stress test without touching the exact target.

    ``forcing_key`` optionally selects a ``dataset.extras`` entry to use as the
    collocation FSI forcing target instead of the exact ``dataset.col_forcing``.
    Used by the forcing-uncertainty stress test to feed perturbed forcing.
    """
    g = torch.Generator().manual_seed(seed)
    if wall_target == "kinematic":
        u_wall_all = dataset.extras.get("wall_velocity_kin", dataset.wall_velocity)
    else:
        u_wall_all = dataset.extras.get("wall_velocity_fsi", dataset.wall_velocity)
    forcing_all = dataset.col_forcing
    if forcing_key is not None and forcing_key in dataset.extras:
        forcing_all = dataset.extras[forcing_key]

    def _idx(n_total, n):
        if n >= n_total:
            return torch.arange(n_total)
        return torch.randperm(n_total, generator=g)[:n]

    def build_batches(step: int):
        di = _idx(dataset.meas_X.shape[0], n_data)
        ci = _idx(dataset.col_X.shape[0], n_col)
        wi = _idx(dataset.wall_X.shape[0], n_wall)
        batches = {
            "data": {
                "X": dataset.meas_X[di].clone(),
                "beam_dir": dataset.meas_beam[di],
                "v_beam": dataset.meas_doppler[di],
            },
            "collocation": {
                "X": dataset.col_X[ci].clone(),
                "forcing": forcing_all[ci],
            },
            "wall": {
                "X": dataset.wall_X[wi].clone(),
                "u_wall": u_wall_all[wi],
            },
        }
        if with_traction and traction_key in dataset.extras:
            batches["traction"] = {
                "X": dataset.wall_X[wi].clone(),
                "normals": dataset.wall_normal[wi],
                "structure_traction": dataset.extras[traction_key][wi],
            }
        return batches

    return build_batches


def compare_backbones_toy(config, steps: int = 3000, lbfgs_iters: int = 300,
                          lr: float = 2e-3, seed: int = 0,
                          use_traction: bool = False, verbose: bool = True,
                          return_models: bool = False):
    """Train Model A and Model B on the SAME data/init and return side-by-side.

    The only differences between the two runs are the FSI momentum forcing, the
    FSI wall-velocity BC, and (optionally) the traction-continuity penalty. The
    network, initialisation, dataset, optimiser schedule and batch order are
    identical, so the comparison isolates the backbone effect.

    Returns ``results`` (dict per backbone). If ``return_models`` is True, returns
    ``(results, models, dataset, lv)`` where ``models`` maps backbone -> PINNNet.
    """
    from ..train.evaluate_toy import evaluate_toy

    dataset, lv = build_toy_dataset(config)
    results, models = {}, {}
    for backbone in ("baseline", "fsi_informed"):
        model, loss_fn = build_toy_model(
            config, dataset, backbone=backbone, init_seed=seed,
            use_traction=(use_traction and backbone == "fsi_informed"),
        )
        loss_fn.anneal.total_steps = steps
        build_batches = make_toy_batch_builder(
            dataset, seed=seed,
            with_traction=(use_traction and backbone == "fsi_informed"),
            wall_target="fsi" if backbone == "fsi_informed" else "kinematic",
        )
        cfg = TrainConfig(steps=steps, lr=lr, lbfgs_iters=lbfgs_iters,
                          log_every=max(1, steps // 6), seed=seed)
        logger = None
        if verbose:
            print(f"\n=== Training backbone: {backbone} ===")

            def logger(step, row, _b=backbone):
                print(f"[{_b[:4]} {step:5d}] " + " ".join(
                    f"{k}={row[k]:.3e}" for k in ("total", "data", "pde", "bc")
                    if k in row))
        train_model(model, loss_fn, build_batches, cfg, logger=logger)
        results[backbone] = evaluate_toy(model, lv, config)
        models[backbone] = model
    if return_models:
        return results, models, dataset, lv
    return results


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
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64 if args.dtype == "float64" else torch.float32)
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


def compare_main() -> None:  # pragma: no cover - CLI wiring
    """CLI: A-vs-B comparison on the toy case (identical architecture/data/init)."""
    import argparse
    import json
    from pathlib import Path
    from ..config import load_config, Config

    parser = argparse.ArgumentParser(description="Compare Model A vs B (toy case).")
    parser.add_argument("--config", default=None)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lbfgs-iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--traction", action="store_true",
                        help="add the traction-continuity penalty to Model B")
    parser.add_argument("--out", default=None, help="write comparison.json here")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64 if args.dtype == "float64" else torch.float32)
    config = load_config(args.config) if args.config else Config()
    results = compare_backbones_toy(
        config, steps=args.steps, lbfgs_iters=args.lbfgs_iters, lr=args.lr,
        seed=args.seed, use_traction=args.traction,
    )

    keys = sorted({k for r in results.values() for k in r})
    print("\n=== Model A (baseline) vs Model B (fsi_informed) ===")
    print(f"{'metric':28s} {'baseline':>12s} {'fsi_informed':>14s} {'delta':>10s}")
    for k in keys:
        a = results['baseline'].get(k, float('nan'))
        b = results['fsi_informed'].get(k, float('nan'))
        print(f"{k:28s} {a:12.4f} {b:14.4f} {b - a:10.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nWrote {args.out}")


def visualize_main() -> None:  # pragma: no cover - CLI wiring
    """CLI: train A and B on the toy case, then render figures + a cycle GIF.

    Produces, under ``--out``:

    * ``fields_t*.png``  -- truth / A / B panels (speed, vorticity, pressure)
    * ``metric_bars.png``-- grouped bar chart of held-out relative-L2 errors
    * ``cycle_speed.gif`` / ``cycle_vorticity.gif`` -- cardiac-cycle animations
    * ``model_{baseline,fsi_informed}.pt`` -- checkpoints
    * ``comparison.json`` -- the metric dictionary
    """
    import argparse
    import json
    from pathlib import Path
    from ..config import load_config, Config
    from ..eval import visualize as viz

    parser = argparse.ArgumentParser(description="Visualise A vs B (toy case).")
    parser.add_argument("--config", default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lbfgs-iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--traction", action="store_true")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--out", default="docs/results")
    parser.add_argument("--grid", type=int, default=90)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--no-anim", action="store_true")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64 if args.dtype == "float64" else torch.float32)
    config = load_config(args.config) if args.config else Config()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results, models, _dataset, lv = compare_backbones_toy(
        config, steps=args.steps, lbfgs_iters=args.lbfgs_iters, lr=args.lr,
        seed=args.seed, use_traction=args.traction, return_models=True,
    )

    for name, model in models.items():
        model.save_checkpoint(out / f"model_{name}.pt",
                              meta={"backbone": name, "metrics": results.get(name, {})})
    with open(out / "comparison.json", "w") as fh:
        json.dump(results, fh, indent=2)

    # Representative frames: early filling and peak systole-ish.
    for frac in (0.3, 0.6):
        t = frac * float(config.flow.period)
        p = viz.plot_ab_panels(lv, models, config, t,
                               out / f"fields_t{frac:.2f}.png", n=args.grid)
        print(f"wrote {p}")
    print(f"wrote {viz.plot_metric_bars(results, out / 'metric_bars.png')}")

    if not args.no_anim:
        for field in ("speed", "vorticity"):
            p = viz.animate_cycle(lv, models, config, out / f"cycle_{field}.gif",
                                  field=field, frames=args.frames,
                                  n=min(args.grid, 80))
            print(f"wrote {p}" if p else f"(animation writer unavailable for {field})")

    print("\n=== metrics ===")
    keys = sorted({k for r in results.values() for k in r})
    for k in keys:
        a = results['baseline'].get(k, float('nan'))
        b = results['fsi_informed'].get(k, float('nan'))
        print(f"{k:28s} A={a:8.4f}  B={b:8.4f}  delta={b - a:+8.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()
