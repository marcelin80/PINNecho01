#!/usr/bin/env python3
"""Emit a concrete example IBFE export to match your real IBAMR/IBFE output to.

Generates a synthetic (NS-exact) :class:`IBFEFrames` and writes it BOTH as:

* ``<out>/bundle.npz``        -- the single-file NPZ contract, and
* ``<out>/manifest.yaml`` + ``<out>/frames/*.csv`` -- the per-frame CSV layout,

then validates the round-trip. Point your real exporter at these files as a
template: reproduce the same column names / array keys / units and PINNecho's
``load_ibfe_output(path)`` will consume it. Use ``--dim 3`` for the 3D ellipsoid
example and ``--valve`` to include a mitral-inflow patch (for residence time).
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from pinnecho.config import Config, load_config
from pinnecho.data.load_ibfe_output import (
    synthetic_ibfe_frames, synthetic_ibfe_frames_3d, load_ibfe_output,
)
from pinnecho.data.ibfe_io import save_ibfe_npz, save_ibfe_manifest_csv
from pinnecho.data.ibfe_validate import validate_ibfe_frames


def main() -> None:
    ap = argparse.ArgumentParser(description="Write an example IBFE export.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dim", type=int, default=2, choices=[2, 3])
    ap.add_argument("--valve", action="store_true", help="include a mitral inflow patch")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--out", default="data/ibfe_example")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    cfg = load_config(args.config) if args.config else Config()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.dim == 3:
        frames = synthetic_ibfe_frames_3d(cfg, n_fluid=3000, n_wall=800,
                                          n_frames=args.n_frames, with_valve=args.valve)
    else:
        frames = synthetic_ibfe_frames(cfg, n_fluid=2500, n_wall=700,
                                       n_frames=args.n_frames, with_valve=args.valve)

    npz_path = save_ibfe_npz(frames, out / "bundle.npz")
    manifest_path = save_ibfe_manifest_csv(frames, out)
    print(f"wrote {npz_path}")
    print(f"wrote {manifest_path} (+ {out}/frames/*.csv)")

    # Round-trip both and validate.
    for label, path in (("npz", npz_path), ("manifest", manifest_path)):
        loaded = load_ibfe_output(str(path), spatial_dim=args.dim)
        report = validate_ibfe_frames(loaded)
        print(f"\n[{label}] {report.summary()}")


if __name__ == "__main__":
    main()
