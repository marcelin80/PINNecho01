#!/usr/bin/env python3
"""Eulerian stasis report on a public 4D-flow / phantom volume (no thrombus model).

Layer 1 of the CHF-thrombus motivation: flag cavity voxels slower than ~10 cm/s.
This is a kinematic risk *marker*. It is not a coagulation solver.

GT-only (no training) — run this on the Corrado NPZ first::

    PYTHONPATH=src python3 scripts/public_volume_stasis.py \\
        --volume data/corrado_lv_phantom/day1_vipr_30min.npz \\
        --out docs/results/public_volume/stasis_gt.json

``--train`` adds a short baseline reconstruction and Dice vs the GT stasis set.
Do not treat a low Dice after 400 steps as a thrombus finding.
"""

import _bootstrap  # noqa: F401
import json
from pathlib import Path

import numpy as np
import torch

from pinnecho.data import load_ibfe_output
from pinnecho.data.ibfe_dataset import train_ibfe
from pinnecho.eval.stasis import volume_from_public_npz, volume_stasis_report


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Public-volume Eulerian stasis (not a thrombus solver).")
    ap.add_argument("--volume", required=True)
    ap.add_argument("--speed-thresh", type=float, default=0.10,
                    help="stasis cut in m/s (default 0.10 = 10 cm/s).")
    ap.add_argument("--persist-frac", type=float, default=0.50)
    ap.add_argument("--train", action="store_true",
                    help="also train baseline and score stasis Dice.")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lbfgs-iters", type=int, default=50)
    ap.add_argument("--n-windows", type=int, default=2)
    ap.add_argument("--n-fluid", type=int, default=2000)
    ap.add_argument("--n-wall", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="docs/results/public_volume/stasis_gt.json")
    args = ap.parse_args()

    arrays = volume_from_public_npz(args.volume)
    gt = volume_stasis_report(
        arrays["velocity"], arrays["mask"], arrays["spacing"],
        speed_thresh=args.speed_thresh, persist_frac=args.persist_frac,
        rho=arrays["rho"])
    result = {"source": args.volume, "which": "stasis_gt",
              "has_oracle_forcing": False, "gt": gt}
    print("GT Eulerian stasis (kinematic marker, not a thrombus)")
    for k in ("speed_thresh", "stasis_frac", "persist_stasis_frac",
              "persist_stasis_ml", "cavity_ml", "speed_mean", "speed_p99",
              "ke_mean"):
        print(f"  {k:22s} {gt[k]}")

    if args.train:
        torch.set_default_dtype(torch.float32)
        from pinnecho.data.ibfe_dataset import default_windows_from_frames
        frames = load_ibfe_output(
            args.volume, n_fluid=args.n_fluid, n_wall=args.n_wall)
        transducers = default_windows_from_frames(frames, n_windows=args.n_windows)
        _, metrics = train_ibfe(
            frames, backbone="baseline", steps=args.steps,
            lbfgs_iters=args.lbfgs_iters, seed=args.seed,
            use_traction=False, transducers=transducers, verbose=True)
        result["which"] = "stasis_gt_and_recon"
        result["recon"] = {k: float(v) for k, v in metrics.items()
                           if isinstance(v, (int, float, np.floating))}
        print("reconstruction vs GT stasis")
        for k in ("vel_relL2_speed", "stasis_frac_true", "stasis_frac_pred",
                  "stasis_dice", "rt_proxy_relL2"):
            if k in metrics:
                print(f"  {k:22s} {metrics[k]:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
