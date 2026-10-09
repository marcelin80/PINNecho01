"""Plane-coverage observability sweep for 3D Doppler reconstruction.

The whole point of the multi-plane acquisition layer
(:mod:`pinnecho.data.acquisition`) is that a *single* echocardiographic view
cannot observe the full 3D velocity field: it measures one component (along the
beam) of only the fluid points lying in its thin imaging slab. Flow that is
out-of-plane, or out of every acquired slab, is simply unmeasured. This module
quantifies that bottleneck by training the reconstruction under a few realistic
acquisition protocols and reporting held-out error vs. the (3D) fluid ground
truth:

* a single apical plane (``a4c``),
* an apical biplane (``a4c + a2c``, angularly complementary slabs),
* apical + parasternal (``a4c + plax``, a different probe apex),
* the full 4-view set (``a4c + a2c + plax + psax``), and
* an *idealized* reference: whole-volume point windows (every fluid point seen
  from N beam angles) -- the setup the earlier toy observability sweep used.

The contrast between the plane protocols and the idealized point windows is the
headline: real echo coverage is slab-limited, so even four standard views do not
reach the recoverability of three volumetric windows. Kept CPU-friendly (small
net, short schedule); runtime scales with ``conditions x seeds``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..data.acquisition import standard_views_from_frames
from ..data.ibfe_dataset import default_windows_from_frames, train_ibfe
from ..data.load_ibfe_output import synthetic_ibfe_frames_3d

# Held-out reconstruction metrics reported for every protocol.
REPORT_KEYS = ("vel_relL2_speed", "vel_relL2_u", "vel_relL2_v", "vel_relL2_w",
               "pressure_relL2")


@dataclass
class CoverageCondition:
    """One acquisition protocol.

    ``views`` names standard imaging planes (see
    :func:`~pinnecho.data.acquisition.standard_views_from_frames`); when it is
    ``None`` the protocol uses ``n_point_windows`` idealized whole-volume point
    windows instead (every fluid point visible from each window).
    """

    name: str
    views: Optional[Tuple[str, ...]] = None
    n_point_windows: Optional[int] = None
    thickness_frac: float = 0.12

    def is_planar(self) -> bool:
        return self.views is not None


def default_coverage_conditions() -> List[CoverageCondition]:
    """Standard 3D protocols from one apical plane up to the full 4-view set,
    plus an idealized 3-point-window reference."""
    return [
        CoverageCondition("a4c (single plane)", views=("a4c",)),
        CoverageCondition("a4c+a2c (apical biplane)", views=("a4c", "a2c")),
        CoverageCondition("a4c+plax (apical+parasternal)", views=("a4c", "plax")),
        CoverageCondition("4-view", views=("a4c", "a2c", "plax", "psax")),
        CoverageCondition("point-windows x3 (idealized)", n_point_windows=3),
    ]


def gap_closing_conditions() -> List[CoverageCondition]:
    """Protocols that isolate the acquisition lever for the unobserved lateral-``y``
    component: the standard apical+parasternal pair (recovers ``u``, not ``v``),
    the same pair plus the non-standard ``lat_y`` research window (whose beam
    carries ``y``), and the idealized whole-volume reference. If adding ``lat_y``
    pulls ``v`` down toward the idealized reference, the gap is a pure coverage
    (beam-direction) gap, not a method limitation."""
    return [
        CoverageCondition("a4c+plax (standard pair)", views=("a4c", "plax")),
        CoverageCondition("a4c+plax+lat_y (+research window)",
                          views=("a4c", "plax", "lat_y")),
        CoverageCondition("point-windows x3 (idealized)", n_point_windows=3),
    ]


PROTOCOL_SETS = {
    "default": default_coverage_conditions,
    "gap": gap_closing_conditions,
}


def _count_data_points(frames, cond: CoverageCondition) -> int:
    """Number of single-component Doppler samples a protocol yields."""
    if cond.is_planar():
        planes = standard_views_from_frames(
            frames, views=cond.views, thickness_frac=cond.thickness_frac)
        return int(sum(int(p.select_mask(frames.coords_fluid).sum()) for p in planes))
    nw = int(cond.n_point_windows or 1)
    return nw * int(frames.coords_fluid.shape[0])


def run_coverage_one(frames, cond: CoverageCondition, backbone: str, steps: int,
                     lbfgs_iters: int, lr: float, seed: int, use_traction: bool,
                     noise_level: float) -> Dict[str, float]:
    """Train one protocol on ``frames`` and return held-out metrics."""
    if cond.is_planar():
        planes = standard_views_from_frames(
            frames, views=cond.views, thickness_frac=cond.thickness_frac)
        _, metrics = train_ibfe(
            frames, backbone=backbone, steps=steps, lbfgs_iters=lbfgs_iters,
            lr=lr, seed=seed, use_traction=use_traction, noise_level=noise_level,
            planes=planes, verbose=False)
    else:
        transducers = default_windows_from_frames(
            frames, n_windows=int(cond.n_point_windows or 1))
        _, metrics = train_ibfe(
            frames, backbone=backbone, steps=steps, lbfgs_iters=lbfgs_iters,
            lr=lr, seed=seed, use_traction=use_traction, noise_level=noise_level,
            transducers=transducers, verbose=False)
    return metrics


def run_coverage_sweep(config=None, frames=None,
                       conditions: Optional[Sequence[CoverageCondition]] = None,
                       backbone: str = "fsi_informed", steps: int = 1200,
                       lbfgs_iters: int = 150, lr: float = 2e-3,
                       seeds: Sequence[int] = (0,), use_traction: bool = True,
                       noise_level: float = 0.05, n_fluid: int = 4000,
                       n_wall: int = 900, n_frames: int = 8,
                       data_seed: int = 0, verbose: bool = True) -> List[Dict]:
    """Train ``backbone`` under each acquisition protocol/seed; aggregate metrics.

    The 3D ground-truth ``frames`` are shared across all conditions (built once
    from ``config`` with ``synthetic_ibfe_frames_3d`` unless passed in), so only
    the acquisition geometry and training seed vary -- a fair coverage A/B.
    Returns records ``{"name", "n_views", "n_data", "<metric>_mean/std"}``.
    """
    if frames is None:
        if config is None:
            from ..config import Config
            config = Config()
        frames = synthetic_ibfe_frames_3d(
            config, n_fluid=n_fluid, n_wall=n_wall, n_frames=n_frames,
            with_valve=True, seed=data_seed)
    conditions = list(conditions) if conditions is not None else default_coverage_conditions()

    records: List[Dict] = []
    for cond in conditions:
        per_seed: List[Dict[str, float]] = []
        for s in seeds:
            m = run_coverage_one(frames, cond, backbone, steps, lbfgs_iters, lr,
                                 s, use_traction, noise_level)
            per_seed.append(m)
            if verbose:
                print(f"[{cond.name} seed={s}] " + " ".join(
                    f"{k}={m.get(k, float('nan')):.3f}" for k in REPORT_KEYS))
        n_views = len(cond.views) if cond.is_planar() else int(cond.n_point_windows or 1)
        rec = {"name": cond.name, "planar": cond.is_planar(), "n_views": n_views,
               "n_data": _count_data_points(frames, cond)}
        for k in REPORT_KEYS:
            vals = np.array([d.get(k, np.nan) for d in per_seed], dtype=float)
            rec[f"{k}_mean"] = float(np.nanmean(vals))
            rec[f"{k}_std"] = float(np.nanstd(vals))
        records.append(rec)
    return records


def plot_coverage(records: List[Dict], out_path, metric: str = "vel_relL2_speed"):
    """Bar chart of ``metric`` (mean +/- std) across acquisition protocols."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    names = [r["name"] for r in records]
    means = [r[f"{metric}_mean"] for r in records]
    stds = [r[f"{metric}_std"] for r in records]
    colors = ["#4c72b0" if r.get("planar", True) else "#c44e52" for r in records]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2, 5))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel(f"{metric} (relative L2, lower is better)")
    ax.set_title("Plane-coverage observability (red = idealized point windows)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def coverage_main() -> None:  # pragma: no cover - CLI wiring
    import argparse
    import json
    from pathlib import Path
    from ..config import load_config, Config

    ap = argparse.ArgumentParser(
        description="3D Doppler plane-coverage observability sweep.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--protocols", default="default", choices=sorted(PROTOCOL_SETS),
                    help="'default' (single plane -> 4-view -> idealized) or "
                         "'gap' (standard pair vs. +lat_y research window).")
    ap.add_argument("--backbone", default="baseline",
                    choices=["baseline", "fsi_informed"])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lbfgs-iters", type=int, default=150)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--no-traction", action="store_true")
    ap.add_argument("--noise-level", type=float, default=0.05)
    ap.add_argument("--n-fluid", type=int, default=4000)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--out", default="docs/results/coverage")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64 if args.dtype == "float64" else torch.float32)
    config = load_config(args.config) if args.config else Config()
    records = run_coverage_sweep(
        config, conditions=PROTOCOL_SETS[args.protocols](),
        backbone=args.backbone, steps=args.steps,
        lbfgs_iters=args.lbfgs_iters, seeds=tuple(args.seeds),
        use_traction=not args.no_traction, noise_level=args.noise_level,
        n_fluid=args.n_fluid)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "coverage.json", "w") as fh:
        json.dump(records, fh, indent=2)
    for metric in ("vel_relL2_speed", "vel_relL2_v", "pressure_relL2"):
        p = plot_coverage(records, out / f"coverage_{metric}.png", metric=metric)
        print(f"wrote {p}")

    print(f"\n{'protocol':30s} {'views':>5s} {'n_data':>8s} "
          f"{'speed':>8s} {'u':>8s} {'v':>8s} {'w':>8s} {'p':>8s}")
    for r in records:
        print(f"{r['name']:30s} {r['n_views']:5d} {r['n_data']:8d} "
              f"{r['vel_relL2_speed_mean']:8.3f} {r['vel_relL2_u_mean']:8.3f} "
              f"{r['vel_relL2_v_mean']:8.3f} {r['vel_relL2_w_mean']:8.3f} "
              f"{r['pressure_relL2_mean']:8.3f}")


if __name__ == "__main__":  # pragma: no cover - CLI wiring
    coverage_main()
