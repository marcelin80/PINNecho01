#!/usr/bin/env python3
"""Convert a Corrado LV-phantom NIfTI scan (OSF 2naw4) to a public-volume NPZ.

Recommended first scan (longest MRI, LV cavity mask on disk)::

    PYTHONPATH=src python3 scripts/convert_corrado_lv.py \\
        --scan "data/corrado_lv_phantom/MRI_Data/Day 1/PC VIPR/30_Minutes" \\
        --out data/corrado_lv_phantom/day1_vipr_30min.npz

Then::

    PYTHONPATH=src python3 scripts/public_volume_ab.py --which observability \\
        --volume data/corrado_lv_phantom/day1_vipr_30min.npz \\
        --windows 1 2 3 --steps 400 --seeds 0 \\
        --out docs/results/public_volume
"""

import _bootstrap  # noqa: F401

import numpy as np
import torch

from pinnecho.data import (
    has_oracle_forcing, load_ibfe_output, validate_ibfe_frames,
)
from pinnecho.data.corrado_lv import convert_corrado_scan


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Corrado LV phantom NIfTI → public-volume NPZ (f = 0).")
    ap.add_argument("--scan", required=True,
                    help="directory with VELX/VELY/VELZ.nii.gz")
    ap.add_argument("--mask", default=None,
                    help="cavity mask. Default: discover next to --scan.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--velocity-unit", default="mm/s",
                    choices=("mm/s", "cm/s", "m/s"))
    ap.add_argument("--period", type=float, default=None,
                    help="cycle period in seconds. Default: T * dt from header.")
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth frame (1 = all).")
    ap.add_argument("--n-fluid", type=int, default=2000)
    ap.add_argument("--n-wall", type=int, default=500)
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float32)
    path, volume = convert_corrado_scan(
        args.scan, args.out, mask_path=args.mask,
        velocity_unit=args.velocity_unit, period=args.period,
        stride=args.stride)
    vel = volume["velocity"]
    mask = volume["mask"]
    interior = mask if mask.ndim == 4 else np.broadcast_to(mask, vel.shape[:4])
    speeds = np.linalg.norm(vel[interior], axis=-1) if interior.any() else np.array([0.0])
    print(f"wrote {path}")
    print(f"  source     {volume['source']}")
    print(f"  mask       {volume['mask_path']}")
    print(f"  velocity   {tuple(vel.shape)}  m/s")
    print(f"  spacing    {volume['spacing']}  m")
    print(f"  origin     {volume['origin']}  m")
    print(f"  times      {volume['times']}")
    print(f"  period     {volume['cycle_period']} s")
    print(f"  speed_max  {float(speeds.max()):.4f} m/s")
    print(f"  mask_frac  {float(interior.mean()):.3f}")
    if args.no_validate:
        return
    frames = load_ibfe_output(
        str(path), n_fluid=args.n_fluid, n_wall=args.n_wall)
    print(validate_ibfe_frames(frames).summary())
    print(f"has_oracle_forcing={has_oracle_forcing(frames)}")


if __name__ == "__main__":
    main()
