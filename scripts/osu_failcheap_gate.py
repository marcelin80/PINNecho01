#!/usr/bin/env python3
"""Fail-cheap gate on a packed OSU public volume. No train.

Pre-registered (do not move after seeing a painted LV):
  lv_vs_tissue   LV time-RMS median ≥ 3× static-tissue ROI
  time_consistency  harmonics 1–4 hold ≥ 0.55 of AC power
  aao_sv         |SV − 93.6| / 93.6 ≤ 0.40
  valve_balance  |SV_mitral − SV_aorta| / max ≤ 0.40
  jet_chamber    10 cm/s remainder mostly in the LV, not aorta/LVOT

The retired LV speed_p99 ≥ 10 cm/s cut cannot fail on noise.

    # after convert (placeholder pack + QA)
    PYTHONPATH=src python3 scripts/osu_failcheap_gate.py \\
        --volume data/osu_4dflow/rest_wholeheart.npz \\
        --qa-dir data/osu_4dflow/qa

    # after versioned LV + tissue + valve masks exist
    PYTHONPATH=src python3 scripts/osu_failcheap_gate.py \\
        --volume data/osu_4dflow/rest_wholeheart.npz \\
        --lv-mask data/osu_4dflow/rest_lv_mask_v1.nii.gz \\
        --tissue-mask data/osu_4dflow/rest_tissue_mask_v1.nii.gz \\
        --aao-mask data/osu_4dflow/rest_aao_mask_v1.nii.gz \\
        --mv-mask data/osu_4dflow/rest_mv_mask_v1.nii.gz \\
        --out docs/results/public_volume/osu_gate.json
    # optional: --aorta-region for a filled Aao/LVOT contamination mask
"""

import _bootstrap  # noqa: F401

from pinnecho.data.osu_gate import (
    evaluate_osu_gate, format_osu_gate, load_mask_any, load_volume_arrays,
    write_gate_json, write_osu_qa,
)
from pinnecho.data.public_volume import save_public_volume_npz


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="OSU dataset #11 fail-cheap gate (not a thrombus).")
    ap.add_argument("--volume", required=True, help="packed public-volume NPZ.")
    ap.add_argument("--lv-mask", default=None,
                    help="LV cavity .npy/.npz/.nii.gz on the same grid.")
    ap.add_argument("--tissue-mask", default=None,
                    help="static-tissue ROI (chest-wall muscle, not lung).")
    ap.add_argument("--aao-mask", default=None,
                    help="thin Aao analysis plane (stroke volume only).")
    ap.add_argument("--mv-mask", default=None,
                    help="thin mitral analysis plane (valve-balance SV).")
    ap.add_argument("--aorta-region", default=None,
                    help="optional Aao/LVOT *region* for jet location (not the plane).")
    ap.add_argument("--aao-normal", type=float, nargs=3, default=None)
    ap.add_argument("--mv-normal", type=float, nargs=3, default=None)
    ap.add_argument("--eval-is-lv", action="store_true",
                    help="packed mask is already an LV cavity (weaker gate).")
    ap.add_argument("--write-lv-npz", default=None,
                    help="if PASS and --lv-mask, write an LV-only public NPZ.")
    ap.add_argument("--qa-dir", default=None)
    ap.add_argument("--out", default=None, help="gate JSON path.")
    args = ap.parse_args()

    vol = load_volume_arrays(args.volume)
    grid = vol["velocity"].shape[1:4]
    lv = load_mask_any(args.lv_mask, grid) if args.lv_mask else None
    tissue = load_mask_any(args.tissue_mask, grid) if args.tissue_mask else None
    aao = load_mask_any(args.aao_mask, grid) if args.aao_mask else None
    mv = load_mask_any(args.mv_mask, grid) if args.mv_mask else None
    ao_region = (load_mask_any(args.aorta_region, grid)
                 if args.aorta_region else None)
    mask_is_lv = bool(args.eval_is_lv or lv is not None)

    gate = evaluate_osu_gate(
        vol["velocity"], vol["mask"], vol["spacing"], vol["times"],
        lv_mask=lv, tissue_mask=tissue, aorta_mask=ao_region,
        aao_plane_mask=aao, aao_normal=args.aao_normal,
        mv_plane_mask=mv, mv_normal=args.mv_normal, mask_is_lv=mask_is_lv)

    print(f"volume {args.volume}")
    print(f"  velocity   {tuple(vol['velocity'].shape)}  m/s")
    print(f"  spacing    {vol['spacing']} m")
    if gate.get("cavity_ml_lv") is not None:
        print(f"  lv_ml      {gate['cavity_ml_lv']:.1f}  "
              "(adult LVEDV is typically ≥ 100 mL)")
    if gate.get("cavity_ml_tissue") is not None:
        print(f"  tissue_ml  {gate['cavity_ml_tissue']:.1f}")
    print(format_osu_gate(gate))
    print("  qa NIfTI intensities are 0-1000 display units, not cm/s")
    if args.out:
        print(f"  gate_json  {write_gate_json(args.out, gate)}")
    if args.qa_dir:
        print(f"  qa_dir     {write_osu_qa(args.qa_dir, velocity=vol['velocity'], spacing_m=vol['spacing'], origin_m=vol['origin'], mask=vol['mask'])}")
    if args.write_lv_npz:
        if gate["verdict"] != "pass":
            print("  skip --write-lv-npz: gate is not PASS")
        elif lv is None:
            print("  skip --write-lv-npz: need --lv-mask")
        else:
            period = float(vol["times"][-1] - vol["times"][0] + (
                vol["times"][1] - vol["times"][0] if vol["times"].size > 1 else 1.0))
            path = save_public_volume_npz(
                args.write_lv_npz, velocity=vol["velocity"], mask=lv,
                spacing=vol["spacing"], origin=vol["origin"], times=vol["times"],
                cycle_period=period, rho=1060.0, mu=0.0035)
            print(f"  lv_npz     {path}")
    print("Cite: Arshad et al., Magn Reson Med 2024; CC BY 4.0 Zenodo 12515230.")
    if gate["verdict"] == "fail":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
