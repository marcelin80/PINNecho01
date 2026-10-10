"""Public 4D-flow / phantom volume sweeps (no oracle forcing).

Public volumes have ``forcing_fluid = 0``. The *informative* question on that
data is **observability**: hold the 3-component velocity as GT, synthesise
single-component Doppler from 1 / 2 / 3 beam directions, and ask whether
cross-beam recovery improves. That is the first non-manufactured check of the
window-count result in ``docs/OBSERVABILITY.md``.

``baseline`` vs ``fsi_param`` (``--which ab``) is kept as a negative-control
plumbing path. Ablation 8 already showed those two are indistinguishable
without true ``f``; repeating it here is not a new scientific claim.

Do not train ``fsi_informed``: the oracle is absent by construction.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

from ..data.ibfe_dataset import default_windows_from_frames, train_ibfe
from ..data.public_volume import (
    has_oracle_forcing, synthetic_public_volume_frames,
)


REPORT_KEYS = ("vel_relL2_speed", "vel_relL2_u", "vel_relL2_v", "vel_relL2_w",
               "pressure_relL2")


def _require_public(frames) -> bool:
    if has_oracle_forcing(frames):
        raise ValueError(
            "this sweep is for public volumes (f = 0). An IBFE bundle with "
            "nonzero forcing belongs on pinnecho-band-oracle, not here.")
    return float(frames.pressure_fluid.abs().max()) > 0.0


def _strip_pressure_if_absent(records, has_p: bool) -> None:
    if has_p:
        return
    for rec in records:
        rec["pressure_relL2_mean"] = float("nan")
        rec["pressure_relL2_std"] = float("nan")


def _train_one(frames, *, backbone, steps, lbfgs_iters, lr, seed,
               noise_level, transducers, n_harmonics, verbose, label):
    if verbose:
        print(f"==> start {label}  steps={steps} lbfgs={lbfgs_iters}", flush=True)
    _, metrics = train_ibfe(
        frames, backbone=backbone, steps=steps, lbfgs_iters=lbfgs_iters,
        lr=lr, seed=seed, use_traction=False, noise_level=noise_level,
        transducers=transducers, n_harmonics=n_harmonics,
        verbose=verbose)
    if verbose:
        print(f"==> done  {label}  " + " ".join(
            f"{k}={metrics.get(k, float('nan')):.3f}" for k in REPORT_KEYS),
              flush=True)
    return metrics


def run_public_volume_observability(
    frames=None, *,
    windows: Sequence[int] = (1, 2, 3),
    steps: int = 400, lbfgs_iters: int = 50, lr: float = 2e-3,
    seeds: Sequence[int] = (0,),
    noise_level: float = 0.05, verbose: bool = True,
) -> Dict:
    """Window-count observability on one public volume (baseline only).

    Each window is still a single-component beam. Adding windows adds
    *directions*, which is the lever documented on manufactured data.
    """
    if frames is None:
        from ..config import Config
        frames = synthetic_public_volume_frames(Config(), n_frames=4,
                                                n_fluid=1500, n_wall=400)
    has_p = _require_public(frames)
    records: List[Dict] = []
    per: Dict[str, List[Dict[str, float]]] = {}
    for n_win in windows:
        name = f"windows={int(n_win)}"
        transducers = default_windows_from_frames(frames, n_windows=int(n_win))
        rows = []
        for s in seeds:
            m = _train_one(
                frames, backbone="baseline", steps=steps,
                lbfgs_iters=lbfgs_iters, lr=lr, seed=s,
                noise_level=noise_level, transducers=transducers,
                n_harmonics=2, verbose=verbose,
                label=f"{name} seed={s}")
            rows.append(m)
        per[name] = rows
        rec: Dict = {"name": name, "n_windows": int(n_win), "backbone": "baseline"}
        for k in REPORT_KEYS:
            vals = np.array([d[k] for d in rows if k in d], dtype=float)
            rec[f"{k}_mean"] = float(vals.mean()) if vals.size else float("nan")
            rec[f"{k}_std"] = float(vals.std()) if vals.size else float("nan")
        records.append(rec)

    from .ablation import _paired_stats
    contrasts: Dict = {}
    names = [r["name"] for r in records]
    if "windows=1" in per and "windows=3" in per:
        a = [d.get("vel_relL2_u", np.nan) for d in per["windows=1"]]
        b = [d.get("vel_relL2_u", np.nan) for d in per["windows=3"]]
        contrasts["windows3_vs_1_crossbeam_u"] = _paired_stats(
            a, b, higher_better=True)
    _strip_pressure_if_absent(records, has_p)
    return {"source": "public_volume", "which": "observability",
            "has_oracle_forcing": False, "has_pressure_gt": bool(has_p),
            "windows": [int(w) for w in windows],
            "records": records, "contrasts": contrasts,
            "conditions": names}


def run_public_volume_ab(
    frames=None, *,
    steps: int = 400, lbfgs_iters: int = 50, lr: float = 2e-3,
    seeds: Sequence[int] = (0,),
    n_windows: int = 2, noise_level: float = 0.05,
    n_harmonics: int = 2, verbose: bool = True,
) -> Dict:
    """Negative control: baseline vs ``fsi_param`` (not the primary question)."""
    if frames is None:
        from ..config import Config
        frames = synthetic_public_volume_frames(Config(), n_frames=4,
                                                n_fluid=1500, n_wall=400)
    has_p = _require_public(frames)

    transducers = default_windows_from_frames(frames, n_windows=n_windows)
    records: List[Dict] = []
    per: Dict[str, List[Dict[str, float]]] = {}
    for name, backbone in (("A (baseline)", "baseline"),
                           ("B-deliverable (fsi_param)", "fsi_param")):
        rows = []
        for s in seeds:
            m = _train_one(
                frames, backbone=backbone, steps=steps,
                lbfgs_iters=lbfgs_iters, lr=lr, seed=s,
                noise_level=noise_level, transducers=transducers,
                n_harmonics=n_harmonics, verbose=verbose,
                label=f"{name} seed={s}")
            rows.append(m)
        per[name] = rows
        rec: Dict = {"name": name, "backbone": backbone}
        for k in REPORT_KEYS:
            vals = np.array([d[k] for d in rows if k in d], dtype=float)
            rec[f"{k}_mean"] = float(vals.mean()) if vals.size else float("nan")
            rec[f"{k}_std"] = float(vals.std()) if vals.size else float("nan")
        records.append(rec)

    from .ablation import _paired_stats
    a = [d.get("vel_relL2_speed", np.nan) for d in per["A (baseline)"]]
    b = [d.get("vel_relL2_speed", np.nan) for d in per["B-deliverable (fsi_param)"]]
    contrasts = {
        "deliverable_vs_baseline": {
            "vel_relL2_speed": _paired_stats(a, b, higher_better=True),
        }
    }
    if has_p:
        pa = [d.get("pressure_relL2", np.nan) for d in per["A (baseline)"]]
        pb = [d.get("pressure_relL2", np.nan) for d in per["B-deliverable (fsi_param)"]]
        contrasts["deliverable_vs_baseline"]["pressure_relL2"] = _paired_stats(
            pa, pb, higher_better=True)
    _strip_pressure_if_absent(records, has_p)
    return {"source": "public_volume", "which": "ab",
            "has_oracle_forcing": False, "has_pressure_gt": bool(has_p),
            "n_windows": int(n_windows), "records": records,
            "contrasts": contrasts}


def public_volume_ab_main() -> None:  # pragma: no cover - CLI wiring
    import argparse
    import json
    from pathlib import Path
    from ..config import Config
    from ..data.public_volume import synthetic_public_volume_frames
    from ..data.load_ibfe_output import load_ibfe_output

    ap = argparse.ArgumentParser(
        description="Public 4D-flow / phantom volume sweeps (f = 0). "
                    "Default: window-count observability (the informative "
                    "question). --which ab is the A vs fsi_param negative control.")
    ap.add_argument("--which", default="observability",
                    choices=("observability", "ab"))
    ap.add_argument("--volume", default=None,
                    help="public-volume or IBFE NPZ. Default: synthetic stand-in.")
    ap.add_argument("--n-fluid", type=int, default=2000)
    ap.add_argument("--n-wall", type=int, default=500)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lbfgs-iters", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 2, 3],
                    help="window counts for --which observability.")
    ap.add_argument("--n-windows", type=int, default=2,
                    help="fixed window count for --which ab.")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-step training logs.")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--out", default="docs/results/public_volume")
    args = ap.parse_args()

    torch.set_default_dtype(
        torch.float64 if args.dtype == "float64" else torch.float32)
    if args.volume:
        frames = load_ibfe_output(
            args.volume, n_fluid=args.n_fluid, n_wall=args.n_wall,
            dtype=torch.get_default_dtype())
    else:
        frames = synthetic_public_volume_frames(
            Config(), n_fluid=args.n_fluid, n_wall=args.n_wall)

    verbose = not args.quiet
    if args.which == "observability":
        result = run_public_volume_observability(
            frames, windows=tuple(args.windows), steps=args.steps,
            lbfgs_iters=args.lbfgs_iters, seeds=tuple(args.seeds),
            verbose=verbose)
        out_name = "public_volume_observability.json"
    else:
        result = run_public_volume_ab(
            frames, steps=args.steps, lbfgs_iters=args.lbfgs_iters,
            seeds=tuple(args.seeds), n_windows=args.n_windows,
            verbose=verbose)
        out_name = "public_volume_ab.json"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / out_name, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {out / out_name}", flush=True)
    print(f"\n{'condition':28s} {'speed':>8s} {'u':>8s} {'v':>8s} {'w':>8s} {'p':>8s}")
    for r in result["records"]:
        print(f"{r['name']:28s} {r['vel_relL2_speed_mean']:8.3f} "
              f"{r['vel_relL2_u_mean']:8.3f} {r['vel_relL2_v_mean']:8.3f} "
              f"{r['vel_relL2_w_mean']:8.3f} {r['pressure_relL2_mean']:8.3f}")


if __name__ == "__main__":  # pragma: no cover
    public_volume_ab_main()
