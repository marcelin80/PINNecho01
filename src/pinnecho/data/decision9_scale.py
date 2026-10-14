"""Decision 9 approximate unstressed reference: isotropic scale to V0.

The image mesh is already loaded. This module scales vertex coordinates
about a centre so the cavity volume becomes the assumed unstressed V0.
It does **not** run IBFE, inflate with atrial pressure, or invert a
loaded shape (backward displacement). Those stay on the solver.

This is an approximation: wall thickness scales with the mesh and the
strain state is not the in-vivo loaded state. The linear ratio
``(V0 / V_img)^{1/3}`` is an exposed assumed parameter because V0 sets
the fill volume.

Do not scale a simulated ``IBFEFrames`` export. Those points are already
on the loaded trajectory. Scale the *reference* mesh, then solve.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from .ibfe_fidelity import (
    APPROX_NOTE,
    IMAGE_VOLUME_ML_PAT12,
    KLOTZ_V0_ML,
    P_IMG_MMHG_ASSUMED,
    WALL_THICKNESS_IMAGE_MM,
    decision_9_approx_fidelity,
    dumps_fidelity,
    isotropic_linear_scale,
)


class Decision9ScaleError(ValueError):
    """User-facing refusal or contract error for the isotropic V0 step."""


IBFE_EXPORT_KEYS = frozenset({
    "coords_fluid", "velocity_fluid", "pressure_fluid", "forcing_fluid",
})


@dataclass(frozen=True)
class IsotropicV0Report:
    """What was assumed and what the scale did. Write this next to the mesh."""

    scale: float
    v0_ml: float
    image_volume_ml: float
    p_img_mmhg: Tuple[float, ...]
    center: Tuple[float, ...]
    n_points: int
    spatial_dim: int
    millimetres: bool
    measured_image_volume_ml: Optional[float]
    measured_v0_ml: Optional[float]
    wall_thickness_image_mm: float
    wall_thickness_approx_mm: float
    approximation: bool = True
    notes: str = APPROX_NOTE
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["p_img_mmhg"] = list(self.p_img_mmhg)
        out["center"] = list(self.center)
        return out


def isotropic_scale_points(
    points: np.ndarray,
    scale: float,
    center: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Scale spatial points about ``center`` (default: vertex mean)."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise Decision9ScaleError(
            f"expected points shape (N, 2) or (N, 3), got {tuple(pts.shape)}"
        )
    if pts.shape[0] == 0:
        raise Decision9ScaleError("no vertices to scale")
    if float(scale) <= 0.0:
        raise Decision9ScaleError(f"scale must be positive, got {scale}")
    if center is None:
        origin = pts.mean(axis=0)
    else:
        origin = np.asarray(center, dtype=float).reshape(-1)
        if origin.shape[0] != pts.shape[1]:
            raise Decision9ScaleError(
                f"center has {origin.shape[0]} components, points have {pts.shape[1]}"
            )
    return origin + float(scale) * (pts - origin)


def signed_surface_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume of a closed triangle surface, in input-length units cubed."""
    verts = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise Decision9ScaleError(
            f"volume needs 3-D vertices (N, 3), got {tuple(verts.shape)}"
        )
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise Decision9ScaleError(
            f"faces must be (F, 3) integer triangles, got {tuple(tris.shape)}"
        )
    if int(tris.min(initial=0)) < 0 or int(tris.max(initial=0)) >= verts.shape[0]:
        raise Decision9ScaleError("face index out of range for vertices")
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    return float(np.einsum("ij,ij->", v0, np.cross(v1, v2)) / 6.0)


def volume_ml(
    vertices: np.ndarray,
    faces: np.ndarray,
    millimetres: bool = False,
) -> float:
    """Absolute cavity volume in millilitres."""
    raw = abs(signed_surface_volume(vertices, faces))
    return raw / 1000.0 if millimetres else raw * 1.0e6


def apply_isotropic_v0(
    points: np.ndarray,
    *,
    v0_ml: float = KLOTZ_V0_ML,
    image_volume_ml: float = IMAGE_VOLUME_ML_PAT12,
    p_img_mmhg: Sequence[float] = P_IMG_MMHG_ASSUMED,
    center: Optional[np.ndarray] = None,
    faces: Optional[np.ndarray] = None,
    millimetres: bool = False,
) -> Tuple[np.ndarray, IsotropicV0Report]:
    """Scale a reference mesh to V0 and return ``(scaled_points, report)``."""
    scale = isotropic_linear_scale(v0_ml, image_volume_ml)
    pts = np.asarray(points, dtype=float)
    if center is None:
        origin = pts.mean(axis=0)
    else:
        origin = np.asarray(center, dtype=float).reshape(-1)
    scaled = isotropic_scale_points(pts, scale, center=origin)

    measured_img = measured_v0 = None
    if faces is not None:
        measured_img = volume_ml(pts, faces, millimetres=millimetres)
        measured_v0 = volume_ml(scaled, faces, millimetres=millimetres)

    report = IsotropicV0Report(
        scale=float(scale),
        v0_ml=float(v0_ml),
        image_volume_ml=float(image_volume_ml),
        p_img_mmhg=tuple(float(p) for p in p_img_mmhg),
        center=tuple(float(c) for c in origin),
        n_points=int(pts.shape[0]),
        spatial_dim=int(pts.shape[1]),
        millimetres=bool(millimetres),
        measured_image_volume_ml=measured_img,
        measured_v0_ml=measured_v0,
        wall_thickness_image_mm=WALL_THICKNESS_IMAGE_MM,
        wall_thickness_approx_mm=WALL_THICKNESS_IMAGE_MM * float(scale),
    )
    return scaled, report


def is_ibfe_export_npz(path: str | Path) -> bool:
    """True if ``path`` looks like a simulated IBFEFrames bundle."""
    path = Path(path)
    if path.suffix != ".npz":
        return False
    with np.load(path) as data:
        return IBFE_EXPORT_KEYS.issubset(set(data.files))


def refuse_if_ibfe_export(path: str | Path) -> None:
    if is_ibfe_export_npz(path):
        raise Decision9ScaleError(
            f"{path} is a simulated IBFEFrames export (has fluid velocity). "
            "Do not isotropically scale a solved beat. Scale the reference "
            "mesh, then run the solver, then tag the new export with "
            "decision_9=approx."
        )


def load_vertices(path: str | Path, key: str = "vertices") -> np.ndarray:
    """Load (N, 2|3) vertices from CSV / NPY / NPZ."""
    path = Path(path)
    refuse_if_ibfe_export(path)
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".npz":
        data = np.load(path)
        if key not in data.files:
            raise Decision9ScaleError(
                f"{path}: missing array {key!r}; found {data.files}"
            )
        arr = data[key]
    else:
        arr = _load_csv_table(path, expect_cols=("x", "y", "z"))
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise Decision9ScaleError(
            f"{path}: expected (N, 2) or (N, 3) vertices, got {tuple(arr.shape)}"
        )
    return arr


def load_faces(path: str | Path, key: str = "faces") -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".npz":
        data = np.load(path)
        if key not in data.files:
            raise Decision9ScaleError(
                f"{path}: missing array {key!r}; found {data.files}"
            )
        arr = data[key]
    else:
        arr = _load_csv_table(path, expect_cols=("i", "j", "k"), integer=True)
    arr = np.asarray(arr, dtype=int)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise Decision9ScaleError(
            f"{path}: expected (F, 3) faces, got {tuple(arr.shape)}"
        )
    return arr


def write_scaled_mesh(
    out_dir: str | Path,
    scaled: np.ndarray,
    report: IsotropicV0Report,
    faces: Optional[np.ndarray] = None,
) -> Path:
    """Write scaled vertices, optional faces, fidelity sidecar, and report."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    header = "x,y,z" if scaled.shape[1] == 3 else "x,y"
    np.savetxt(
        out / "vertices_scaled.csv",
        scaled,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.8e",
    )
    np.save(out / "vertices_scaled.npy", scaled)
    if faces is not None:
        np.savetxt(
            out / "faces.csv",
            np.asarray(faces, dtype=int),
            delimiter=",",
            header="i,j,k",
            comments="",
            fmt="%d",
        )
    fid = decision_9_approx_fidelity(
        v0_ml=report.v0_ml,
        image_volume_ml=report.image_volume_ml,
        p_img_mmhg=report.p_img_mmhg,
    )
    (out / "fidelity.json").write_text(dumps_fidelity(fid) + "\n")
    (out / "report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    (out / "report.md").write_text(_report_markdown(report))
    return out


def tag_ibfe_export_approx(
    frames,
    *,
    v0_ml: float = KLOTZ_V0_ML,
    image_volume_ml: float = IMAGE_VOLUME_ML_PAT12,
    p_img_mmhg: Sequence[float] = P_IMG_MMHG_ASSUMED,
):
    """Attach Decision 9 approx fidelity to a *new* beat solved on scaled V0."""
    import dataclasses

    return dataclasses.replace(
        frames,
        fidelity=decision_9_approx_fidelity(
            v0_ml=v0_ml,
            image_volume_ml=image_volume_ml,
            p_img_mmhg=p_img_mmhg,
        ),
    )


def _load_csv_table(
    path: Path,
    expect_cols: Sequence[str],
    integer: bool = False,
) -> np.ndarray:
    raw = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if getattr(raw, "dtype", None) is not None and raw.dtype.names:
        names = [n.strip() for n in raw.dtype.names]
        use = [c for c in expect_cols if c in names]
        if len(use) >= 2:
            arr = np.column_stack([np.asarray(raw[c], dtype=float) for c in use])
        else:
            arr = np.column_stack(
                [np.asarray(raw[n], dtype=float) for n in raw.dtype.names]
            )
    else:
        arr = np.atleast_2d(np.loadtxt(path, delimiter=",", dtype=float))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if integer:
        return np.asarray(np.rint(arr), dtype=int)
    return np.asarray(arr, dtype=float)


def _report_markdown(report: IsotropicV0Report) -> str:
    measured = ""
    if report.measured_image_volume_ml is not None:
        measured = (
            f"- Measured image volume (from faces): "
            f"{report.measured_image_volume_ml:.3f} mL\n"
            f"- Measured V0 (from scaled faces): "
            f"{report.measured_v0_ml:.3f} mL\n"
        )
    units = "mm" if report.millimetres else "m"
    return (
        "# Decision 9 approximate V0 (isotropic scale)\n\n"
        "This is an **approximation**. Wall thickness scales with the mesh "
        f"({report.wall_thickness_image_mm:.1f} mm → "
        f"{report.wall_thickness_approx_mm:.1f} mm). The strain state is "
        "not the in-vivo loaded state. Formal backward displacement is not "
        "applied.\n\n"
        f"- Assumed image volume: {report.image_volume_ml} mL\n"
        f"- Assumed V0: {report.v0_ml} mL (Klotz single-beat empiric; "
        "large error; pat12 may differ)\n"
        f"- Exposed linear scale `(V0/V_img)^(1/3)`: **{report.scale:.6f}**\n"
        f"- Assumed p_img (mid–late diastole): {list(report.p_img_mmhg)} mmHg\n"
        f"- Scale centre ({units}): {list(report.center)}\n"
        f"- Vertices: {report.n_points}\n"
        f"{measured}\n"
        "Next on the solver: fill the scaled reference with atrial pressure "
        "toward the image volume. Run the sensitivity at **5 mmHg and "
        "10 mmHg**. Tag each export with `decision_9: approx` from "
        "`fidelity.json`. Do not use a pre-D9 beat for E-wave / IVPG / "
        "vortex-ring claims.\n"
    )
