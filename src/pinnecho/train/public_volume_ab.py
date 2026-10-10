"""A vs deliverable-B on a public 4D-flow / phantom volume (no oracle).

Public volumes have ``forcing_fluid = 0``. Training ``fsi_informed`` on them is
meaningless (the oracle is absent). This sweep is the 3-way of Ablation 8 with
the oracle arm removed: ``baseline`` vs ``fsi_param`` on one shared
non-manufactured field.
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


def run_public_volume_ab(
    frames=None, *,
    steps: int = 400, lbfgs_iters: int = 50, lr: float = 2e-3,
    seeds: Sequence[int] = (0,),
    n_windows: int = 2, noise_level: float = 0.05,
    n_harmonics: int = 2, verbose: bool = True,
) -> Dict:
    """Train baseline and ``fsi_param`` on one public-volume ``frames``."""
    if frames is None:
        from ..config import Config
        frames = synthetic_public_volume_frames(Config(), n_frames=4,
                                                n_fluid=1500, n_wall=400)
    if has_oracle_forcing(frames):
        raise ValueError(
            "this sweep is for public volumes (f = 0). An IBFE bundle with "
            "nonzero forcing belongs on pinnecho-band-oracle --conditions "
            "deliverable, not here.")
    has_p = float(frames.pressure_fluid.abs().max()) > 0.0

    transducers = default_windows_from_frames(frames, n_windows=n_windows)
    records: List[Dict] = []
    per: Dict[str, List[Dict[str, float]]] = {}
    for name, backbone in (("A (baseline)", "baseline"),
                           ("B-deliverable (fsi_param)", "fsi_param")):
        rows = []
        for s in seeds:
            _, m = train_ibfe(
                frames, backbone=backbone, steps=steps, lbfgs_iters=lbfgs_iters,
                lr=lr, seed=s, use_traction=False, noise_level=noise_level,
                transducers=transducers, n_harmonics=n_harmonics,
                verbose=False)
            rows.append(m)
            if verbose:
                print(f"[{name} seed={s}] " + " ".join(
                    f"{k}={m.get(k, float('nan')):.3f}" for k in REPORT_KEYS))
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
    else:
        # 4D-flow MRI has no pressure GT; relL2 against a zero field is not a metric.
        for rec in records:
            rec["pressure_relL2_mean"] = float("nan")
            rec["pressure_relL2_std"] = float("nan")
    return {"source": "public_volume", "has_oracle_forcing": False,
            "has_pressure_gt": bool(has_p),
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
        description="A vs deliverable-B on a public 4D-flow / phantom volume "
                    "(f = 0; no oracle).")
    ap.add_argument("--volume", default=None,
                    help="public-volume or IBFE NPZ. Default: synthetic stand-in.")
    ap.add_argument("--n-fluid", type=int, default=2000)
    ap.add_argument("--n-wall", type=int, default=500)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lbfgs-iters", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n-windows", type=int, default=2)
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

    result = run_public_volume_ab(
        frames, steps=args.steps, lbfgs_iters=args.lbfgs_iters,
        seeds=tuple(args.seeds), n_windows=args.n_windows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "public_volume_ab.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {out / 'public_volume_ab.json'}")
    print(f"\n{'condition':28s} {'speed':>8s} {'u':>8s} {'v':>8s} {'w':>8s} {'p':>8s}")
    for r in result["records"]:
        print(f"{r['name']:28s} {r['vel_relL2_speed_mean']:8.3f} "
              f"{r['vel_relL2_u_mean']:8.3f} {r['vel_relL2_v_mean']:8.3f} "
              f"{r['vel_relL2_w_mean']:8.3f} {r['pressure_relL2_mean']:8.3f}")


if __name__ == "__main__":  # pragma: no cover
    public_volume_ab_main()
