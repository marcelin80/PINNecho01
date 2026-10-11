#!/usr/bin/env python3
"""Eulerian stasis on a public 4D-flow / phantom volume (no thrombus model).

GT-only (already run on Corrado)::

    PYTHONPATH=src python3 scripts/public_volume_stasis.py \\
        --volume data/corrado_lv_phantom/day1_vipr_30min.npz \\
        --out docs/results/public_volume/stasis_gt.json

Blur-GT Dice upper bound (no train; voxel σ ≈ PINN smoothness)::

    PYTHONPATH=src python3 scripts/public_volume_stasis.py \\
        --volume data/corrado_lv_phantom/day1_vipr_30min.npz \\
        --blur-sigma 1.0 --cuts 0.05 0.10 0.15 \\
        --out docs/results/public_volume/stasis_blur_gt.json

Reconstruction vs that GT. Commands: see
``docs/results/public_volume/JET_PRESERVE.md``. Do not overwrite
``stasis_recon.json``. Do not rerun A.
"""

import _bootstrap  # noqa: F401
import json
from pathlib import Path

import numpy as np
import torch

from pinnecho.data import load_ibfe_output
from pinnecho.data.ibfe_dataset import default_windows_from_frames, train_ibfe
from pinnecho.eval.stasis import (
    blur_velocity, compare_registered_volumes, compare_volume_stasis,
    peak_speed, persistent_stasis_mask, predict_volume_velocity,
    volume_from_public_npz, volume_stasis_report,
)


def _resolve_fast_set(args):
    """B: per-frame |v_beam| quantile, independent of 5/10/15 cm/s.

    ``--fast-frac 0.5`` with no other flags → quantile 0.80 (top 20 % per
    frame). Absolute ``--fast-thresh`` is only used when the quantile is
    omitted on purpose (legacy / A is ``fast_frac=None``).
    """
    if args.fast_frac is None:
        return None, None, "none"
    if args.fast_quantile is not None:
        return float(args.fast_quantile), None, "per_frame_vbeam_quantile"
    if args.fast_thresh is not None:
        return None, float(args.fast_thresh), "abs_v_beam"
    return 0.80, None, "per_frame_vbeam_quantile"


def _print_cut_row(cut, vol, prefix=""):
    print(f"{prefix}cut={cut:.2f} jet {vol['jet_ml_true']:.2f}→"
          f"{vol['jet_ml_pred']:.2f} err={vol['signed_jet_err']:+.1%} "
          f"dice={vol['jet_dice']:.3f} P={vol['jet_precision']:.3f} "
          f"R={vol['jet_recall']:.3f} slow={vol['slow_speed_relL2']:.3f} "
          f"peak={vol['peak_ratio_median']:.3f}")


def _score_cuts(vel_true, vel_pred, mask, spacing, cuts, persist_frac, rho):
    by_cut = {}
    for cut in cuts:
        vol = compare_volume_stasis(
            vel_true, vel_pred, mask, spacing, speed_thresh=cut,
            persist_frac=persist_frac, rho=rho, absolute=True)
        by_cut[f"{cut:.2f}"] = vol
        _print_cut_row(cut, vol, prefix="  ")
    return by_cut


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
    ap.add_argument("--fast-frac", type=float, default=None,
                    help="Doppler-batch fraction drawn from the fast set. "
                         "The set is per-frame |v_beam| quantile (not GT |u|).")
    ap.add_argument("--fast-quantile", type=float, default=None,
                    help="per-frame |v_beam| quantile defining the fast set "
                         "(default 0.80 when --fast-frac is set). Independent "
                         "of the 5/10/15 cm/s eval cuts.")
    ap.add_argument("--fast-thresh", type=float, default=None,
                    help="absolute |v_beam| cut in m/s. Do not use for B — "
                         "it ties the set to 10 cm/s. Quantile is the default.")
    ap.add_argument("--n-data", type=int, default=3072,
                    help="Doppler batch size. Fixed across window counts "
                         "unless --scale-n-data-with-windows.")
    ap.add_argument("--scale-n-data-with-windows", action="store_true",
                    help="set n_data = --n-data * n_windows (per-window budget).")
    ap.add_argument("--cuts", type=float, nargs="+", default=[0.05, 0.10, 0.15],
                    help="absolute stasis cuts (m/s) scored on the same vel_pred.")
    ap.add_argument("--blur-sigma", type=float, nargs="+", default=None,
                    help="no-train: blur GT by these voxel σ and score vs sharp GT.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.other and args.train:
        raise SystemExit("--other is GT vs GT; do not pass --train")

    arrays = volume_from_public_npz(args.volume)
    absolute = bool(args.absolute or args.other)
    cuts = [float(c) for c in args.cuts]
    gt = volume_stasis_report(
        arrays["velocity"], arrays["mask"], arrays["spacing"],
        speed_thresh=args.speed_thresh, persist_frac=args.persist_frac,
        rho=arrays["rho"], absolute=absolute)
    result = {"source": args.volume, "which": "stasis_gt",
              "has_oracle_forcing": False, "gt": gt,
              "jet_definition": (
                  "cavity & ~persist; persist = |u|<cut for >= persist_frac "
                  "of frames (not ever-above-cut)")}
    print("GT Eulerian stasis (kinematic marker, not a thrombus)")
    print("  jet = above cut for >50% of the cycle (not ever-above)")
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

    if args.blur_sigma:
        print("blur-GT control (no train; Dice upper bound vs sharp GT)")
        blur_rows = {}
        for sig in args.blur_sigma:
            vel_b = blur_velocity(arrays["velocity"], float(sig))
            print(f"  sigma={sig:g} vox")
            blur_rows[f"{sig:g}"] = _score_cuts(
                arrays["velocity"], vel_b, arrays["mask"], arrays["spacing"],
                cuts, args.persist_frac, arrays["rho"])
        result["which"] = "stasis_blur_gt"
        result["blur_sigma"] = [float(s) for s in args.blur_sigma]
        result["blur_gt"] = blur_rows

    if args.train:
        torch.set_default_dtype(torch.float32)
        if args.seeds:
            seeds = tuple(int(s) for s in args.seeds)
        elif args.seed is not None:
            seeds = (int(args.seed),)
        else:
            seeds = (0,)
        n_data = int(args.n_data)
        if args.scale_n_data_with_windows:
            n_data = n_data * int(args.n_windows)
        fast_q, fast_th, fast_src = _resolve_fast_set(args)
        out_early = Path(args.out or "docs/results/public_volume/stasis_recon.json")
        records = []
        for seed in seeds:
            frames = load_ibfe_output(
                args.volume, n_fluid=args.n_fluid, n_wall=args.n_wall,
                seed=seed)
            transducers = default_windows_from_frames(
                frames, n_windows=args.n_windows)
            print(f"[train] seed={seed} windows={args.n_windows} "
                  f"n_data={n_data} (fixed unless --scale-n-data-with-windows) "
                  f"data_fast_frac={args.fast_frac} source={fast_src} "
                  f"quantile={fast_q} thresh={fast_th}")
            model, point = train_ibfe(
                frames, backbone="baseline", steps=args.steps,
                lbfgs_iters=args.lbfgs_iters, seed=seed,
                use_traction=False, transducers=transducers, verbose=True,
                n_data=n_data, data_fast_frac=args.fast_frac,
                data_fast_thresh=0.10 if fast_th is None else fast_th,
                data_fast_quantile=fast_q)
            vel_pred = predict_volume_velocity(
                model, arrays["origin"], arrays["spacing"], arrays["times"],
                arrays["velocity"].shape[1:4])
            print(f"volume vs GT (seed={seed}; cuts {cuts})")
            by_cut = _score_cuts(
                arrays["velocity"], vel_pred, arrays["mask"],
                arrays["spacing"], cuts, args.persist_frac, arrays["rho"])
            print(f"  vel_relL2_speed {point.get('vel_relL2_speed', float('nan'))}")
            persist_010 = persistent_stasis_mask(
                arrays["velocity"], arrays["mask"],
                speed_thresh=0.10, persist_frac=args.persist_frac)
            cavity = np.asarray(arrays["mask"], dtype=bool)
            if cavity.ndim == 4:
                cavity = cavity.any(axis=0)
            jet_010 = cavity & np.logical_not(persist_010)
            peak_path = out_early.with_name(f"{out_early.stem}_seed{seed}_peaks.npz")
            peak_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                peak_path,
                peak_true=peak_speed(arrays["velocity"]).astype(np.float32),
                peak_pred=peak_speed(vel_pred).astype(np.float32),
                jet_true_010=jet_010.astype(np.uint8),
            )
            print(f"  wrote {peak_path}")
            rec = {
                "seed": seed,
                "n_windows": int(args.n_windows),
                "n_data": n_data,
                "data_fast_frac": args.fast_frac,
                "data_fast_source": fast_src,
                "data_fast_quantile": fast_q,
                "data_fast_thresh": fast_th,
                "peaks_npz": str(peak_path),
                "point": {k: float(v) for k, v in point.items()
                          if isinstance(v, (int, float, np.floating))},
                "volume_by_cut": by_cut,
            }
            records.append(rec)
        result["which"] = "stasis_gt_and_recon"
        result["n_windows"] = int(args.n_windows)
        result["steps"] = int(args.steps)
        result["seeds"] = list(seeds)
        result["cuts"] = cuts
        result["n_data"] = n_data
        result["n_data_scales_with_windows"] = bool(args.scale_n_data_with_windows)
        result["scale_n_data_with_windows"] = bool(args.scale_n_data_with_windows)
        result["data_fast_frac"] = args.fast_frac
        result["data_fast_source"] = fast_src
        result["data_fast_quantile"] = fast_q
        result["data_fast_thresh"] = fast_th
        result["beam_perpendicular_limit"] = (
            "|v_beam| oversample cannot see a jet moving perpendicular to the beam")
        result["records"] = records
        if len(records) == 1:
            result["point"] = records[0]["point"]
            result["volume"] = records[0]["volume_by_cut"].get("0.10") or next(
                iter(records[0]["volume_by_cut"].values()))

    out = Path(args.out or (
        "docs/results/public_volume/stasis_recon.json" if args.train
        else "docs/results/public_volume/stasis_registered.json" if args.other
        else "docs/results/public_volume/stasis_blur_gt.json" if args.blur_sigma
        else "docs/results/public_volume/stasis_gt.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
