#!/usr/bin/env python3
"""Coarse cavity from an ITK-SNAP cursor box. Not an endocardial contour.

2.3 mm CORe magnitude cannot support a publication LV wall. Note the
cursor (x,y,z) at opposite corners of the chamber (or a 1-slice Aao
plane) and threshold inside that box.

    PYTHONPATH=src python3 scripts/osu_box_mask.py \\
        --image data/osu_4dflow/qa/mag_mean.npy \\
        --volume data/osu_4dflow/rest_wholeheart.npz \\
        --box 20:45 35:70 25:50 \\
        --percentile 55 \\
        --out data/osu_4dflow/rest_lv_mask.nii.gz
"""

import _bootstrap  # noqa: F401

from pathlib import Path

import numpy as np

from pinnecho.data.osu_gate import (
    _try_write_nifti, box_threshold_mask, load_volume_arrays, parse_index_span,
)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Box + percentile cavity (coarse, not a wall trace).")
    ap.add_argument("--image", required=True, help="3D .npy (prefer mag_mean.npy).")
    ap.add_argument("--volume", required=True, help="public-volume NPZ (grid/spacing).")
    ap.add_argument("--box", nargs=3, required=True, metavar="LO:HI",
                    help="SNAP cursor ranges x y z, inclusive.")
    ap.add_argument("--percentile", type=float, default=55.0,
                    help="keep voxels >= this percentile inside the box.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vol = load_volume_arrays(args.volume)
    grid = tuple(int(s) for s in vol["velocity"].shape[1:4])
    image = np.load(args.image)
    if image.ndim != 3:
        raise SystemExit(f"--image must be 3D, got {image.shape}")
    if tuple(image.shape) != grid:
        raise SystemExit(f"image {image.shape} vs volume grid {grid}")
    box = (
        parse_index_span(args.box[0], grid[0]),
        parse_index_span(args.box[1], grid[1]),
        parse_index_span(args.box[2], grid[2]),
    )
    mask = box_threshold_mask(image, box, percentile=args.percentile)
    voxel_ml = float(np.prod(vol["spacing"])) * 1e6
    ml = float(mask.sum()) * voxel_ml
    out = Path(args.out)
    written = _try_write_nifti(out, mask.astype(np.uint8), vol["spacing"], vol["origin"])
    npy_name = out.name.replace(".nii.gz", ".npy").replace(".nii", ".npy")
    np.save(out.with_name(npy_name), mask.astype(np.uint8))
    print(f"wrote {out}  ({written})")
    print(f"  box         {args.box}  percentile={args.percentile:g}")
    print(f"  voxels      {int(mask.sum())}  ~{ml:.1f} mL")
    print("  this is a coarse cavity, not an LV wall. inspect in ITK-SNAP.")


if __name__ == "__main__":
    main()
