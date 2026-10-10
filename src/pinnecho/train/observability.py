"""Observability / robustness sweep for the Doppler PINN.

The cross-beam velocity component is only weakly constrained by a single
acoustic window; extra windows, lower noise and denser sampling all improve
recoverability. This module trains a backbone under a grid of acquisition
conditions and reports held-out reconstruction error so the observability
bottleneck can be quantified.

Kept CPU-friendly: defaults use a small network and short schedule. Runtime
scales linearly with the number of (condition x seed) trainings.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .train import (
    build_toy_dataset, build_toy_model, make_toy_batch_builder,
    train_model, TrainConfig,
)
from .evaluate_toy import evaluate_toy


# Roughly complementary extra acoustic windows (parasternal-like), used when a
# condition requests >1 window. The primary window is the config default. These
# match ``pinnecho.train.ablation`` so window counts are consistent across studies.
SECOND_WINDOW = (0.07, -0.02)
THIRD_WINDOW = (-0.07, -0.02)

# Report these held-out metrics for every condition.
REPORT_KEYS = ("vel_relL2_speed", "vel_relL2_u", "vel_relL2_v",
               "vorticity_relL2", "wss_relL2", "pressure_relL2",
               "pressure_corr", "vorticity_corr", "wss_corr")


@dataclass
class SweepCondition:
    """One acquisition setting. ``None`` fields keep the base-config value."""
    name: str
    n_windows: Optional[int] = None
    noise_level: Optional[float] = None
    n_points_per_frame: Optional[int] = None

    def apply(self, config):
        cfg = copy.deepcopy(config)
        d = cfg.doppler
        if self.n_windows is not None:
            extra = [SECOND_WINDOW, THIRD_WINDOW]
            if self.n_windows <= 1:
                d.transducers = ()
            else:
                d.transducers = tuple([d.transducer] + extra[: self.n_windows - 1])
        if self.noise_level is not None:
            d.noise_level = float(self.noise_level)
        if self.n_points_per_frame is not None:
            d.n_points_per_frame = int(self.n_points_per_frame)
        return cfg


def default_conditions(base_windows: int = 1, base_noise: float = 0.05,
                       base_points: int = 300) -> List[SweepCondition]:
    """Three 1-D sweeps (windows / noise / sparsity) around a base setting."""
    conds = [
        SweepCondition("windows=1", n_windows=1),
        SweepCondition("windows=2", n_windows=2),
        SweepCondition("windows=3", n_windows=3),
        SweepCondition("noise=0.02", noise_level=0.02),
        SweepCondition("noise=0.10", noise_level=0.10),
        SweepCondition("points=150", n_points_per_frame=150),
        SweepCondition("points=600", n_points_per_frame=600),
    ]
    # Ensure every condition inherits the requested base for the non-swept knobs.
    for c in conds:
        if c.n_windows is None:
            c.n_windows = base_windows
        if c.noise_level is None:
            c.noise_level = base_noise
        if c.n_points_per_frame is None:
            c.n_points_per_frame = base_points
    return conds


def _snr_db(noise_level: float) -> float:
    """Approximate SNR in dB for a relative-std Gaussian noise model."""
    return float(20.0 * np.log10(1.0 / max(noise_level, 1e-9)))


def run_one(config, backbone: str, steps: int, lbfgs_iters: int, lr: float,
            seed: int, use_traction: bool) -> Dict[str, float]:
    dataset, lv = build_toy_dataset(config, seed=seed)
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
                      log_every=max(1, steps), seed=seed)
    train_model(model, loss_fn, build_batches, cfg, logger=None)
    return evaluate_toy(model, lv, config)


def run_sweep(config, conditions: Optional[Sequence[SweepCondition]] = None,
              backbone: str = "fsi_informed", steps: int = 1200,
              lbfgs_iters: int = 150, lr: float = 2e-3,
              seeds: Sequence[int] = (0,), use_traction: bool = True,
              verbose: bool = True) -> List[Dict]:
    """Train ``backbone`` under each condition/seed and collect metrics.

    Returns a list of records: ``{"name", "snr_db", "n_windows", ..., "<metric>_mean",
    "<metric>_std"}`` aggregated over seeds.
    """
    conditions = list(conditions) if conditions is not None else default_conditions()
    records: List[Dict] = []
    for cond in conditions:
        cfg = cond.apply(config)
        per_seed: List[Dict[str, float]] = []
        for s in seeds:
            m = run_one(cfg, backbone, steps, lbfgs_iters, lr, s, use_traction)
            per_seed.append(m)
            if verbose:
                print(f"[{cond.name} seed={s}] " + " ".join(
                    f"{k}={m.get(k, float('nan')):.3f}" for k in REPORT_KEYS))
        d = cfg.doppler
        n_windows = len(d.transducers) if getattr(d, "transducers", ()) else 1
        noise_level = float(d.noise_level)
        rec = {"name": cond.name, "n_windows": n_windows,
               "noise_level": noise_level, "snr_db": _snr_db(noise_level),
               "n_points_per_frame": int(d.n_points_per_frame)}
        for k in REPORT_KEYS:
            vals = np.array([d.get(k, np.nan) for d in per_seed], dtype=float)
            rec[f"{k}_mean"] = float(np.nanmean(vals))
            rec[f"{k}_std"] = float(np.nanstd(vals))
        records.append(rec)
    return records


def plot_sweep(records: List[Dict], out_path, metric: str = "vel_relL2_speed"):
    """Bar chart of ``metric`` (mean +/- std) across conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    names = [r["name"] for r in records]
    means = [r[f"{metric}_mean"] for r in records]
    stds = [r[f"{metric}_std"] for r in records]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(1.4 * len(names) + 2, 5))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4c72b0")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel(f"{metric} (relative L2, lower is better)")
    ax.set_title("Observability sweep")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def sweep_main() -> None:  # pragma: no cover - CLI wiring
    import argparse
    import json
    from pathlib import Path
    from ..config import load_config, Config

    ap = argparse.ArgumentParser(description="Doppler observability sweep.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--backbone", default="fsi_informed",
                    choices=["baseline", "fsi_informed"])
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lbfgs-iters", type=int, default=150)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--no-traction", action="store_true")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--out", default="docs/results/observability")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64 if args.dtype == "float64" else torch.float32)
    config = load_config(args.config) if args.config else Config()
    records = run_sweep(config, backbone=args.backbone, steps=args.steps,
                        lbfgs_iters=args.lbfgs_iters, seeds=tuple(args.seeds),
                        use_traction=not args.no_traction)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sweep.json", "w") as fh:
        json.dump(records, fh, indent=2)
    for metric in ("vel_relL2_speed", "vel_relL2_v", "wss_relL2"):
        p = plot_sweep(records, out / f"sweep_{metric}.png", metric=metric)
        print(f"wrote {p}")

    print(f"\n{'condition':14s} {'SNR(dB)':>8s} {'win':>4s} {'pts':>5s} "
          f"{'speed':>8s} {'v':>8s} {'wss':>8s}")
    for r in records:
        print(f"{r['name']:14s} {r['snr_db']:8.1f} {r['n_windows']:4d} "
              f"{r['n_points_per_frame']:5d} {r['vel_relL2_speed_mean']:8.3f} "
              f"{r['vel_relL2_v_mean']:8.3f} {r['wss_relL2_mean']:8.3f}")
