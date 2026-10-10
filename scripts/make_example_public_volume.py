#!/usr/bin/env python3
"""Write a public-volume NPZ (4D-flow / phantom contract) from the synthetic LV.

This is a *format template*, not a public dataset. Convert a real 4D-flow or
phantom release into the same keys (velocity, mask, spacing, origin, times)
and ``load_ibfe_output`` will consume it with ``f = 0``.
"""

import _bootstrap  # noqa: F401
import torch

from pinnecho.config import Config
from pinnecho.data import (
    load_ibfe_output, save_public_volume_npz, rasterize_synthetic_lv_3d,
    validate_ibfe_frames, has_oracle_forcing,
)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Write an example public-volume NPZ.")
    ap.add_argument("--out", default="data/public_volume_example/volume.npz")
    ap.add_argument("--nx", type=int, default=16)
    ap.add_argument("--ny", type=int, default=16)
    ap.add_argument("--nz", type=int, default=20)
    ap.add_argument("--n-frames", type=int, default=4)
    ap.add_argument("--with-pressure", action="store_true")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float32)
    volume, _ = rasterize_synthetic_lv_3d(
        Config(), nx=args.nx, ny=args.ny, nz=args.nz, n_frames=args.n_frames,
        with_pressure=args.with_pressure)
    path = save_public_volume_npz(
        args.out, velocity=volume["velocity"], mask=volume["mask"],
        spacing=volume["spacing"], origin=volume["origin"], times=volume["times"],
        pressure=volume.get("pressure"),
        cycle_period=volume["cycle_period"], rho=volume["rho"], mu=volume["mu"])
    print(f"wrote {path}")
    frames = load_ibfe_output(str(path), n_fluid=1500, n_wall=400)
    print(validate_ibfe_frames(frames).summary())
    print(f"has_oracle_forcing={has_oracle_forcing(frames)}")


if __name__ == "__main__":
    main()
