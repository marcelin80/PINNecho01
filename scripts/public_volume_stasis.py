#!/usr/bin/env python3
"""Eulerian stasis on a public 4D-flow / phantom volume (no thrombus model).

GT-only (already run on Corrado)::

    PYTHONPATH=src python3 scripts/public_volume_stasis.py \\
        --volume data/corrado_lv_phantom/day1_vipr_30min.npz \\
        --out docs/results/public_volume/stasis_gt.json

Reconstruction vs that GT (volume persist Dice + jet mL), one seed::

    PYTHONPATH=src python3 scripts/public_volume_stasis.py --train \\
        --volume data/corrado_lv_phantom/day1_vipr_30min.npz \\
        --steps 2000 --n-windows 3 --seed 0 \\
        --out docs/results/public_volume/stasis_recon.json

MRI vs registered PIV (same grid, overlap mask, fixed 10 cm/s; no train)::

    PYTHONPATH=src python3 scripts/public_volume_stasis.py \\
        --volume data/corrado_lv_phantom/day1_vipr_30min.npz \\
        --other data/corrado_lv_phantom/day1_piv_avg.npz \\
        --speed-thresh 0.10 \\
        --out docs/results/public_volume/stasis_mri_vs_piv.json
"""

import _bootstrap  # noqa: F401
import json
from pathlib import Path

import numpy as np
import torch

from pinnecho.data import load_ibfe_output
from pinnecho.data.ibfe_dataset import default_windows_from_frames, train_ibfe
from pinnecho.eval.stasis import (
    compare_registered_volumes, compare_volume_stasis, predict_volume_velocity,
    volume_from_public_npz, volume_stasis_report,
)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Public-volume Eulerian stasis (not a thrombus solver).")
    ap.add_argument("--volume", required=True)
    ap.add_argument("--other", default=None,
                    help="second registered volume (e.g. PIV). GT vs GT, no train.")
    ap.add_argument("--speed-thresh", type=float, default=0.10,
                    help="stasis cut in m/s (default 0.10 = 10 cm/s).")
    ap.add_argument("--absolute", action="store_true",
                    help="do not shrink speed_thresh when p99 is below the cut.")
    ap.add_argument("--persist-frac", type=float, default=0.50)
    ap.add_argument("--train", action="store_true",
                    help="train baseline and score volume persist/jet vs GT.")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lbfgs-iters", type=int, default=50)
    ap.add_argument("--n-windows", type=int, default=3)
    ap.add_argument("--n-fluid", type=int, default=2000)
    ap.add_argument("--n-wall", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.other and args.train:
        raise SystemExit("--other is GT vs GT; do not pass --train")

    arrays = volume_from_public_npz(args.volume)
    absolute = bool(args.absolute or args.other)
    gt = volume_stasis_report(
        arrays["velocity"], arrays["mask"], arrays["spacing"],
        speed_thresh=args.speed_thresh, persist_frac=args.persist_frac,
        rho=arrays["rho"], absolute=absolute)
    result = {"source": args.volume, "which": "stasis_gt",
              "has_oracle_forcing": False, "gt": gt}
    print("GT Eulerian stasis (kinematic marker, not a thrombus)")
    for k in ("speed_thresh", "stasis_frac", "persist_stasis_frac",
              "persist_stasis_ml", "jet_ml", "cavity_ml", "speed_mean",
              "speed_p99", "ke_mean"):
        print(f"  {k:22s} {gt[k]}")

    if args.other:
        other = volume_from_public_npz(args.other)
        if not np.allclose(arrays["spacing"], other["spacing"]):
            raise SystemExit(
                f"spacing {arrays['spacing']} vs {other['spacing']}")
        cmp = compare_registered_volumes(
            arrays["velocity"], arrays["mask"],
            other["velocity"], other["mask"],
            arrays["spacing"],
            origin_a=arrays["origin"], origin_b=other["origin"],
            speed_thresh=args.speed_thresh, persist_frac=args.persist_frac,
            rho=arrays["rho"], name_a="mri", name_b="piv")
        result["which"] = "stasis_registered"
        result["other"] = args.other
        result["registered"] = cmp
        print("registered MRI vs other (overlap mask, fixed cut)")
        for k in ("cavity_ml_mri", "cavity_ml_piv", "overlap_ml",
                  "overlap_frac_of_mri", "origin_delta_vox",
                  "mri_jet_ml", "piv_jet_ml",
                  "overlap_jet_ml_true", "overlap_jet_ml_pred",
                  "overlap_jet_dice", "overlap_persist_stasis_dice",
                  "overlap_speed_mean_true", "overlap_speed_mean_pred",
                  "overlap_speed_relL2",
                  "mri_speed_p99", "piv_speed_p99"):
            if k in cmp:
                print(f"  {k:32s} {cmp[k]}")

    if args.train:
        torch.set_default_dtype(torch.float32)
        frames = load_ibfe_output(
            args.volume, n_fluid=args.n_fluid, n_wall=args.n_wall)
        transducers = default_windows_from_frames(
            frames, n_windows=args.n_windows)
        model, point = train_ibfe(
            frames, backbone="baseline", steps=args.steps,
            lbfgs_iters=args.lbfgs_iters, seed=args.seed,
            use_traction=False, transducers=transducers, verbose=True)
        vel_pred = predict_volume_velocity(
            model, arrays["origin"], arrays["spacing"], arrays["times"],
            arrays["velocity"].shape[1:4])
        vol = compare_volume_stasis(
            arrays["velocity"], vel_pred, arrays["mask"], arrays["spacing"],
            speed_thresh=args.speed_thresh, persist_frac=args.persist_frac,
            rho=arrays["rho"])
        result["which"] = "stasis_gt_and_recon"
        result["n_windows"] = int(args.n_windows)
        result["steps"] = int(args.steps)
        result["seed"] = int(args.seed)
        result["point"] = {k: float(v) for k, v in point.items()
                           if isinstance(v, (int, float, np.floating))}
        result["volume"] = vol
        print("volume reconstruction vs GT stasis (same cut)")
        for k in ("persist_stasis_ml_true", "persist_stasis_ml_pred",
                  "jet_ml_true", "jet_ml_pred", "persist_stasis_dice",
                  "jet_dice",
                  "stasis_frac_true", "stasis_frac_pred",
                  "speed_mean_true", "speed_mean_pred"):
            print(f"  {k:26s} {vol[k]}")
        print(f"  vel_relL2_speed           {point.get('vel_relL2_speed', float('nan'))}")

    out = Path(args.out or (
        "docs/results/public_volume/stasis_recon.json" if args.train
        else "docs/results/public_volume/stasis_registered.json" if args.other
        else "docs/results/public_volume/stasis_gt.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
