#!/usr/bin/env python3
"""Decision 9 approximate V0: isotropically scale a reference mesh.

This is the PINNecho-side next step after locking (a). It does **not**
solve IBFE. Scale the image mesh to V0, write the exposed assumptions,
then on the laptop solver fill with atrial pressure at 5 mmHg and
10 mmHg toward the image volume.

    python3 scripts/decision9_isotropic_v0.py print-scale \\
        --v0-ml 58 --image-volume-ml 105

    python3 scripts/decision9_isotropic_v0.py scale \\
        --vertices /absolute/path/to/real_endo_vertices.csv \\
        --faces /absolute/path/to/real_endo_faces.csv \\
        --v0-ml 58 --image-volume-ml 105 \\
        --out data/decision9_v0_approx

    python3 scripts/decision9_isotropic_v0.py tag \\
        --export /absolute/path/to/new_beat.npz \\
        --v0-ml 58 --image-volume-ml 105 --p-img-mmhg 5 \\
        --out /absolute/path/to/new_beat_d9.npz

Do not pass a simulated IBFEFrames bundle to ``scale``. Those points
are already on the loaded trajectory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from pinnecho.data.decision9_scale import (
    Decision9ScaleError,
    apply_isotropic_v0,
    load_faces,
    load_vertices,
    tag_ibfe_export_approx,
    write_scaled_mesh,
)
from pinnecho.data.ibfe_fidelity import (
    IMAGE_VOLUME_ML_PAT12,
    KLOTZ_V0_ML,
    P_IMG_MMHG_ASSUMED,
    isotropic_linear_scale,
)


def _add_shared(p: argparse.ArgumentParser) -> None:
    p.add_argument("--v0-ml", type=float, default=KLOTZ_V0_ML,
                   help="assumed unstressed volume (mL); default Klotz 58")
    p.add_argument("--image-volume-ml", type=float, default=IMAGE_VOLUME_ML_PAT12,
                   help="assumed image-state cavity volume (mL); default 105")
    p.add_argument(
        "--p-img-mmhg", type=float, nargs="+",
        default=list(P_IMG_MMHG_ASSUMED),
        help="assumed mid–late-diastolic image pressure(s), mmHg",
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_print = sub.add_parser("print-scale", help="print the exposed linear scale")
    _add_shared(p_print)

    p_scale = sub.add_parser("scale", help="scale a reference mesh to V0")
    _add_shared(p_scale)
    p_scale.add_argument("--vertices", required=True,
                         help="CSV / NPY / NPZ vertices (N, 2|3), SI metres")
    p_scale.add_argument("--faces", default=None,
                         help="optional (F, 3) triangles to measure volume")
    p_scale.add_argument("--vertices-key", default="vertices")
    p_scale.add_argument("--faces-key", default="faces")
    p_scale.add_argument("--center", type=float, nargs="+", default=None,
                         help="scale centre; default is the vertex mean")
    p_scale.add_argument("--millimetres", action="store_true",
                         help="input/output lengths are millimetres, not metres")
    p_scale.add_argument("--out", required=True)

    p_tag = sub.add_parser("tag", help="tag a new IBFE export as Decision 9 approx")
    _add_shared(p_tag)
    p_tag.add_argument("--export", required=True, help=".npz IBFE bundle")
    p_tag.add_argument("--out", required=True)

    args = ap.parse_args(argv)

    if args.cmd == "print-scale":
        scale = isotropic_linear_scale(args.v0_ml, args.image_volume_ml)
        payload = {
            "v0_ml": args.v0_ml,
            "image_volume_ml": args.image_volume_ml,
            "v0_scale": scale,
            "p_img_mmhg": list(args.p_img_mmhg),
            "approximation": True,
            "decision_9": "approx",
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    try:
        if args.cmd == "scale":
            return _scale(args)
        if args.cmd == "tag":
            return _tag(args)
    except Decision9ScaleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1


def _require_file(path: str, what: str) -> None:
    p = Path(path)
    if not p.is_file():
        raise Decision9ScaleError(
            f"{what} not found: {p}. Run from the PINNecho repo root with "
            "python3 (not python). print-scale needs no mesh; scale needs a "
            "real reference-vertex file, not a path/to/... placeholder."
        )


def _scale(args) -> int:
    _require_file(args.vertices, "vertices")
    if args.faces:
        _require_file(args.faces, "faces")
    points = load_vertices(args.vertices, key=args.vertices_key)
    faces = load_faces(args.faces, key=args.faces_key) if args.faces else None
    center = None if args.center is None else args.center
    scaled, report = apply_isotropic_v0(
        points,
        v0_ml=args.v0_ml,
        image_volume_ml=args.image_volume_ml,
        p_img_mmhg=args.p_img_mmhg,
        center=center,
        faces=faces,
        millimetres=args.millimetres,
    )
    out = write_scaled_mesh(args.out, scaled, report, faces=faces)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    print(f"wrote {out}", file=sys.stderr)
    return 0


def _tag(args) -> int:
    from pinnecho.data.ibfe_io import save_ibfe_npz
    from pinnecho.data.load_ibfe_output import load_ibfe_output

    _require_file(args.export, "export")
    frames = load_ibfe_output(args.export)
    tagged = tag_ibfe_export_approx(
        frames,
        v0_ml=args.v0_ml,
        image_volume_ml=args.image_volume_ml,
        p_img_mmhg=args.p_img_mmhg,
    )
    path = save_ibfe_npz(tagged, Path(args.out))
    print(json.dumps(tagged.fidelity.to_dict(), indent=2, sort_keys=True))
    print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
