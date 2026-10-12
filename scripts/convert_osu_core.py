#!/usr/bin/env python3
"""CORe recon ``outputs.theta*`` (phase) → public-volume NPZ. No train.

Dataset #11 VENC is 200 cm/s (Arshad et al., MRM 2024, Fig. 4).
First pack is **without** ``--mask`` (magnitude placeholder). Draw LV + Aao
from the QA NIfTI, then ``scripts/osu_failcheap_gate.py``.

    PYTHONPATH=src python3 scripts/convert_osu_core.py \\
        --recon path/to/rest11_*_core.mat \\
        --spacing-mm 2.3 2.3 2.7 \\
        --venc-cm-s 200 \\
        --out data/osu_4dflow/rest_wholeheart.npz \\
        --qa-dir data/osu_4dflow/qa \\
        --gate-json docs/results/public_volume/osu_gate_placeholder.json
"""

import _bootstrap  # noqa: F401

import numpy as np

from pinnecho.data.osu_core import (
    DATASET11_VENC_MPS, load_core_outputs, magnitude_mask, pack_osu_public_volume,
)
from pinnecho.data.osu_gate import (
    evaluate_osu_gate, format_osu_gate, load_mask_any, write_gate_json,
    write_osu_qa,
)
from pinnecho.data.public_volume import save_public_volume_npz


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="OSU CORe phase → public-volume NPZ (f = 0). Not a thrombus.")
    ap.add_argument("--recon", required=True,
                    help="CORe save .mat with outputs.thetaX/Y/Z and xHat.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--venc-cm-s", type=float, default=200.0,
                    help="VENC in cm/s. Dataset #11 = 200 (paper Fig. 4).")
    ap.add_argument("--spacing-mm", type=float, nargs=3, required=True,
                    help="voxel size in mm (Study IV range 2.3–3.1).")
    ap.add_argument("--origin-mm", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--period", type=float, default=1.0,
                    help="RR interval in seconds (20 bins).")
    ap.add_argument("--mask", default=None,
                    help="LV cavity .npy/.npz/.nii.gz. Required to claim LV.")
    ap.add_argument("--tissue-mask", default=None,
                    help="static-tissue ROI (chest-wall muscle, not lung).")
    ap.add_argument("--aao-mask", default=None,
                    help="thin Aao analysis plane (stroke volume only).")
    ap.add_argument("--mv-mask", default=None,
                    help="thin mitral analysis plane (valve-balance SV).")
    ap.add_argument("--aorta-region", default=None,
                    help="optional Aao/LVOT region for jet location.")
    ap.add_argument("--aao-normal", type=float, nargs=3, default=None,
                    help="through-plane unit vector; default = mean v in plane.")
    ap.add_argument("--mv-normal", type=float, nargs=3, default=None,
                    help="mitral through-plane unit vector; default = mean v.")
    ap.add_argument("--mag-percentile", type=float, default=70.0,
                    help="placeholder mask on |xHat| if --mask is omitted.")
    ap.add_argument("--qa-dir", default=None,
                    help="write mid-slice PNGs + NIfTI for drawing LV/Aao.")
    ap.add_argument("--gate-json", default=None,
                    help="write the fail-cheap verdict JSON.")
    args = ap.parse_args()

    thetas = load_core_outputs(args.recon)
    venc = float(args.venc_cm_s) / 100.0
    spacing = np.asarray(args.spacing_mm, dtype=np.float64) / 1000.0
    origin = np.asarray(args.origin_mm, dtype=np.float64) / 1000.0

    mag_mask = magnitude_mask(thetas["xHat"], args.mag_percentile)
    lv_mask = None
    mask_is_lv = False
    if args.mask:
        lv_mask = load_mask_any(args.mask, mag_mask.shape)
        mask_is_lv = True
    pack_mask = lv_mask if lv_mask is not None else mag_mask
    vol = pack_osu_public_volume(
        thetas, venc_mps=venc, spacing_m=spacing, origin_m=origin,
        period_s=args.period, mask=pack_mask, mask_is_lv=mask_is_lv)
    tissue_mask = None
    if args.tissue_mask:
        tissue_mask = load_mask_any(args.tissue_mask, mag_mask.shape)
    aao_mask = None
    if args.aao_mask:
        aao_mask = load_mask_any(args.aao_mask, mag_mask.shape)
    mv_mask = None
    if args.mv_mask:
        mv_mask = load_mask_any(args.mv_mask, mag_mask.shape)
    ao_region = None
    if args.aorta_region:
        ao_region = load_mask_any(args.aorta_region, mag_mask.shape)

    path = save_public_volume_npz(
        args.out, velocity=vol["velocity"], mask=vol["mask"],
        spacing=vol["spacing"], origin=vol["origin"], times=vol["times"],
        cycle_period=vol["cycle_period"], rho=1060.0, mu=0.0035)

    # Jet location uses the magnitude mask so aorta contamination is visible.
    gate = evaluate_osu_gate(
        vol["velocity"], mag_mask, vol["spacing"], vol["times"],
        lv_mask=lv_mask, tissue_mask=tissue_mask, aorta_mask=ao_region,
        aao_plane_mask=aao_mask, aao_normal=args.aao_normal,
        mv_plane_mask=mv_mask, mv_normal=args.mv_normal,
        mask_is_lv=mask_is_lv)

    print(f"wrote {path}")
    print(f"  venc       {venc:.2f} m/s  (dataset #11 paper default "
          f"{DATASET11_VENC_MPS:.2f})")
    print(f"  velocity   {tuple(vol['velocity'].shape)}  m/s")
    print(f"  spacing    {vol['spacing']} m")
    print(f"  mask_is_lv {mask_is_lv}  "
          "(False = magnitude placeholder, fail-cheap stays PENDING)")
    print(format_osu_gate(gate))
    if args.gate_json:
        gpath = write_gate_json(args.gate_json, gate)
        print(f"  gate_json  {gpath}")
    if args.qa_dir:
        qdir = write_osu_qa(
            args.qa_dir, velocity=vol["velocity"], spacing_m=vol["spacing"],
            origin_m=vol["origin"], mag=thetas["xHat"], mask=mag_mask)
        print(f"  qa_dir     {qdir}")
    print("Cite: Arshad et al., Magn Reson Med 2024; CC BY 4.0 Zenodo 12515230.")
    if gate["verdict"] == "fail":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
