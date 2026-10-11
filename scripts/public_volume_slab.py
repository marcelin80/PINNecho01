#!/usr/bin/env python3
"""Echo-plane slab coverage of the public-volume jet. No training.

    PYTHONPATH=src python3 scripts/public_volume_slab.py \\
        --volume data/corrado_lv_phantom/day1_vipr_30min.npz \\
        --out docs/results/public_volume/slab_coverage.json

AABB-canonical A4C/A2C/PLAX/PSAX (same as plane_coverage), 10 cm/s jet.
Not a cardiologist-placed probe. Not a thrombus. Do not pass --train.
"""

import _bootstrap  # noqa: F401
import json
from pathlib import Path

from pinnecho.eval.slab_coverage import DEFAULT_THICKNESS_FRAC, slab_coverage_report
from pinnecho.eval.stasis import volume_from_public_npz


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Jet vs echo-plane slabs (no train, not a thrombus).")
    ap.add_argument("--volume", required=True)
    ap.add_argument("--speed-thresh", type=float, default=0.10)
    ap.add_argument("--persist-frac", type=float, default=0.50)
    ap.add_argument("--thickness-frac", type=float, default=DEFAULT_THICKNESS_FRAC)
    ap.add_argument("--out", default="docs/results/public_volume/slab_coverage.json")
    args = ap.parse_args()

    arrays = volume_from_public_npz(args.volume)
    rows = slab_coverage_report(
        arrays["velocity"], arrays["mask"], arrays["spacing"], arrays["origin"],
        speed_thresh=args.speed_thresh, persist_frac=args.persist_frac,
        thickness_frac=args.thickness_frac)
    print("echo-plane slab coverage of the 10 cm/s jet (no train)")
    print(f"{'protocol':20s} {'jet_ml':>8s} {'in_slab':>8s} {'frac':>7s} "
          f"{'along':>7s} {'cavity_slab':>12s}")
    for r in rows:
        print(f"{r['name']:20s} {r['jet_ml']:8.2f} {r['jet_ml_in_slab']:8.2f} "
              f"{r['jet_frac_in_slab']:7.3f} {r['jet_along_beam_frac']:7.3f} "
              f"{r['cavity_ml_in_slab']:12.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": args.volume,
        "which": "slab_coverage",
        "speed_thresh": float(args.speed_thresh),
        "thickness_frac": float(args.thickness_frac),
        "has_oracle_forcing": False,
        "protocols": rows,
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
