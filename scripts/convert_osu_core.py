#!/usr/bin/env python3
"""CORe recon ``outputs.theta*`` (phase) → public-volume NPZ. No train.

Dataset #11 VENC is 200 cm/s (Arshad et al., MRM 2024, Fig. 4).
``--mask`` must be an LV cavity to pass the OSU gate. A magnitude
percentile is only a placeholder and is **not** an LV segmentation.

    PYTHONPATH=src python3 scripts/convert_osu_core.py \\
        --recon path/to/rest_core.mat \\
        --spacing-mm 2.3 2.3 2.7 \\
        --venc-cm-s 200 \\
        --out data/osu_4dflow/rest_lv.npz
"""

import _bootstrap  # noqa: F401

import numpy as np

from pinnecho.data.osu_core import (
    DATASET11_VENC_MPS, load_core_outputs, pack_osu_public_volume,
)
from pinnecho.data.public_volume import save_public_volume_npz
from pinnecho.eval.stasis import volume_stasis_report


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
                    help="LV cavity .npy/.npz boolean. Required to claim LV.")
    ap.add_argument("--mag-percentile", type=float, default=70.0,
                    help="placeholder mask on |xHat| if --mask is omitted.")
    args = ap.parse_args()

    thetas = load_core_outputs(args.recon)
    mask = None
    mask_is_lv = False
    if args.mask:
        raw = np.load(args.mask)
        mask = raw["mask"] if isinstance(raw, np.lib.npyio.NpzFile) else raw
        mask = np.asarray(mask, dtype=bool)
        mask_is_lv = True
    venc = float(args.venc_cm_s) / 100.0
    spacing = np.asarray(args.spacing_mm, dtype=np.float64) / 1000.0
    origin = np.asarray(args.origin_mm, dtype=np.float64) / 1000.0
    vol = pack_osu_public_volume(
        thetas, venc_mps=venc, spacing_m=spacing, origin_m=origin,
        period_s=args.period, mask=mask, mag_percentile=args.mag_percentile)
    path = save_public_volume_npz(
        args.out, velocity=vol["velocity"], mask=vol["mask"],
        spacing=vol["spacing"], origin=vol["origin"], times=vol["times"],
        cycle_period=vol["cycle_period"], rho=1060.0, mu=0.0035)
    gt = volume_stasis_report(
        vol["velocity"], vol["mask"], vol["spacing"],
        speed_thresh=0.10, absolute=True)
    print(f"wrote {path}")
    print(f"  venc       {venc:.2f} m/s  (dataset #11 paper default "
          f"{DATASET11_VENC_MPS:.2f})")
    print(f"  velocity   {tuple(vol['velocity'].shape)}  m/s")
    print(f"  spacing    {vol['spacing']} m")
    print(f"  mask_is_lv {mask_is_lv}  "
          "(False = magnitude placeholder, fail the LV gate)")
    print(f"  speed_p99  {gt['speed_p99']:.4f} m/s")
    print(f"  cavity_ml  {gt['cavity_ml']:.2f}")
    print(f"  jet_ml@10  {gt['jet_ml']:.2f}")
    if gt["speed_p99"] < 0.10:
        print("FAIL: speed_p99 < 10 cm/s — clinical cut has no jet")
    if not mask_is_lv:
        print("FAIL-pending: pass --mask with an LV cavity. "
              "If the 10 cm/s remainder is aorta/LVOT only, reject.")
    print("Cite: Arshad et al., Magn Reson Med 2024; CC BY 4.0 Zenodo 12515230.")


if __name__ == "__main__":
    main()
