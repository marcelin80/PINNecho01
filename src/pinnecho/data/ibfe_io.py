"""On-disk I/O for :class:`~pinnecho.data.load_ibfe_output.IBFEFrames`.

This module defines the *concrete on-disk contract* a real IBAMR/IBFE export
must match to be consumed by PINNecho, and implements two portable, dependency-
light loaders plus a saver:

1. **NPZ bundle** (single ``.npz`` file) -- the simplest handoff. All arrays in
   one file with fixed keys (see :data:`NPZ_TENSOR_KEYS`). Round-trippable with
   :func:`save_ibfe_npz` / :func:`load_ibfe_npz`.

2. **Manifest + per-frame CSV directory** -- for exports written frame-by-frame.
   A ``manifest.yaml`` (see :func:`load_ibfe_manifest`) lists the cycle metadata
   and, per time frame, the CSV files for the fluid volume, wall interface and
   (optionally) the mitral/aortic valves. Column names are matched by header, so
   extra columns are ignored and order does not matter.

3. **VTK/Exodus point clouds** -- an *optional* hook (:func:`load_frame_vtk`)
   guarded behind ``meshio``; used when a manifest frame points at a ``.vtu`` /
   ``.vtk`` / ``.e`` file plus a field-name mapping.

Units & conventions (identical to the :class:`IBFEFrames` docstring): SI metres,
seconds, m/s, Pa, N/m^3; coordinate columns are spatial-first then time; 2D uses
``(x, y[, t])`` and 3D appends ``z``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


# NPZ keys for the array fields (order-independent; loaded by name).
NPZ_TENSOR_KEYS = [
    "coords_fluid", "velocity_fluid", "pressure_fluid", "forcing_fluid",
    "coords_wall", "normals_wall", "velocity_wall", "traction_wall",
    "coords_mitral", "velocity_mitral", "coords_aortic", "velocity_aortic",
]


def frames_to_dtype(frames, dtype: torch.dtype):
    """Return a copy of ``frames`` with every tensor cast to ``dtype``."""
    import dataclasses

    updates = {k: getattr(frames, k).to(dtype) for k in NPZ_TENSOR_KEYS}
    return dataclasses.replace(frames, **updates)


def _spatial_cols(spatial_dim: int) -> List[str]:
    return ["x", "y", "z"][:spatial_dim]


def _vel_cols(spatial_dim: int) -> List[str]:
    return ["u", "v", "w"][:spatial_dim]


# --------------------------------------------------------------------------- #
# NPZ bundle
# --------------------------------------------------------------------------- #
def save_ibfe_npz(frames, path: str | Path) -> Path:
    """Serialise an :class:`IBFEFrames` to a single ``.npz`` bundle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: getattr(frames, k).detach().cpu().numpy() for k in NPZ_TENSOR_KEYS}
    arrays["_meta"] = np.array(
        [float(frames.cycle_period), float(frames.rho), float(frames.mu),
         float(frames.spatial_dim)], dtype=np.float64)
    np.savez(path, **arrays)
    return path


def load_ibfe_npz(path: str | Path, dtype: torch.dtype = torch.float64):
    """Load an :class:`IBFEFrames` from a ``.npz`` bundle written by :func:`save_ibfe_npz`."""
    from .load_ibfe_output import IBFEFrames

    data = np.load(path)
    meta = data["_meta"]
    kw = {k: torch.as_tensor(data[k], dtype=dtype) for k in NPZ_TENSOR_KEYS}
    return IBFEFrames(
        cycle_period=float(meta[0]), rho=float(meta[1]), mu=float(meta[2]),
        spatial_dim=int(round(float(meta[3]))), **kw,
    )


# --------------------------------------------------------------------------- #
# Manifest + per-frame CSV
# --------------------------------------------------------------------------- #
def _read_csv_named(path: Path) -> Dict[str, np.ndarray]:
    """Read a headered CSV into a ``{column_name: 1D array}`` dict."""
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float,
                        deletechars="", replace_space="_")
    if arr.dtype.names is None:
        raise ValueError(f"{path}: expected a header row with named columns")
    out = {}
    for name in arr.dtype.names:
        col = np.atleast_1d(arr[name]).astype(float)
        out[name.strip()] = col
    return out


def _stack(cols: Dict[str, np.ndarray], names: List[str], src: Path) -> np.ndarray:
    missing = [n for n in names if n not in cols]
    if missing:
        raise ValueError(f"{src}: missing required columns {missing}; "
                         f"found {sorted(cols)}")
    return np.stack([cols[n] for n in names], axis=1)


def _frame_time_column(n: int, t: float) -> np.ndarray:
    return np.full((n, 1), float(t))


def load_ibfe_manifest(manifest_path: str | Path,
                       dtype: torch.dtype = torch.float64):
    """Load an :class:`IBFEFrames` from a ``manifest.yaml`` + per-frame CSVs.

    Manifest schema (YAML)::

        spatial_dim: 2            # 2 or 3
        cycle_period: 0.9         # seconds
        rho: 1060.0               # kg/m^3
        mu: 0.0035                # Pa*s
        frames:
          - t: 0.00
            fluid:  frames/fluid_000.csv    # x,y[,z],u,v[,w],p,fx,fy[,fz]
            wall:   frames/wall_000.csv      # x,y[,z],nx,ny[,nz],uw,vw[,ww],tx,ty[,tz]
            mitral: frames/mitral_000.csv    # x,y[,z],u,v[,w]      (optional)
            aortic: frames/aortic_000.csv    # x,y[,z],u,v[,w]      (optional)
          - t: 0.05
            ...

    Paths in the manifest are resolved relative to the manifest's directory.
    CSV columns are matched by *name* (header row required); unknown columns are
    ignored. Times are taken from each frame's ``t`` and appended as the last
    coordinate column.
    """
    import yaml

    from .load_ibfe_output import IBFEFrames

    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    with open(manifest_path) as fh:
        spec = yaml.safe_load(fh)

    dim = int(spec["spatial_dim"])
    sp, ve = _spatial_cols(dim), _vel_cols(dim)
    nrm = [f"n{a}" for a in sp]
    uw = [f"{a}w" for a in "uvw"[:dim]]  # uw, vw, ww
    trac = [f"t{a}" for a in "xyz"[:dim]]
    frc = [f"f{a}" for a in "xyz"[:dim]]

    buf: Dict[str, List[np.ndarray]] = {k: [] for k in NPZ_TENSOR_KEYS}

    def _resolve(rel):
        return (root / rel) if rel is not None else None

    for fr in spec["frames"]:
        t = float(fr["t"])

        fluid = _read_csv_named(_resolve(fr["fluid"]))
        n = len(next(iter(fluid.values())))
        buf["coords_fluid"].append(
            np.concatenate([_stack(fluid, sp, Path(fr["fluid"])),
                            _frame_time_column(n, t)], axis=1))
        buf["velocity_fluid"].append(_stack(fluid, ve, Path(fr["fluid"])))
        buf["pressure_fluid"].append(_stack(fluid, ["p"], Path(fr["fluid"])))
        buf["forcing_fluid"].append(_stack(fluid, frc, Path(fr["fluid"])))

        wall = _read_csv_named(_resolve(fr["wall"]))
        nw = len(next(iter(wall.values())))
        buf["coords_wall"].append(
            np.concatenate([_stack(wall, sp, Path(fr["wall"])),
                            _frame_time_column(nw, t)], axis=1))
        buf["normals_wall"].append(_stack(wall, nrm, Path(fr["wall"])))
        buf["velocity_wall"].append(_stack(wall, uw, Path(fr["wall"])))
        buf["traction_wall"].append(_stack(wall, trac, Path(fr["wall"])))

        for key, cprefix in (("mitral", "coords_mitral"), ("aortic", "coords_aortic")):
            rel = fr.get(key)
            if rel is None:
                continue
            valve = _read_csv_named(_resolve(rel))
            nv = len(next(iter(valve.values())))
            buf[cprefix].append(
                np.concatenate([_stack(valve, sp, Path(rel)),
                                _frame_time_column(nv, t)], axis=1))
            vkey = "velocity_mitral" if key == "mitral" else "velocity_aortic"
            buf[vkey].append(_stack(valve, ve, Path(rel)))

    def _cat(key, ncols):
        if buf[key]:
            return torch.as_tensor(np.concatenate(buf[key], axis=0), dtype=dtype)
        return torch.zeros(0, ncols, dtype=dtype)

    return IBFEFrames(
        coords_fluid=_cat("coords_fluid", dim + 1),
        velocity_fluid=_cat("velocity_fluid", dim),
        pressure_fluid=_cat("pressure_fluid", 1),
        forcing_fluid=_cat("forcing_fluid", dim),
        coords_wall=_cat("coords_wall", dim + 1),
        normals_wall=_cat("normals_wall", dim),
        velocity_wall=_cat("velocity_wall", dim),
        traction_wall=_cat("traction_wall", dim),
        coords_mitral=_cat("coords_mitral", dim + 1),
        velocity_mitral=_cat("velocity_mitral", dim),
        coords_aortic=_cat("coords_aortic", dim + 1),
        velocity_aortic=_cat("velocity_aortic", dim),
        cycle_period=float(spec["cycle_period"]),
        rho=float(spec["rho"]), mu=float(spec["mu"]), spatial_dim=dim,
    )


def load_frame_vtk(path: str | Path, field_map: Dict[str, str]):
    """Read points + named point-data arrays from a VTK/Exodus file (optional).

    Requires ``meshio``. ``field_map`` maps logical names to the point-data array
    names in the file, e.g. ``{"velocity": "U", "pressure": "p", "forcing": "F"}``.
    Returns ``(points, {logical_name: array})``. This is a convenience for wiring
    a VTK-based manifest loader; the CSV/NPZ paths cover the common case without
    extra dependencies.
    """
    try:
        import meshio
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "load_frame_vtk requires the optional 'meshio' package "
            "(pip install meshio). Use the CSV/NPZ loaders otherwise."
        ) from exc

    mesh = meshio.read(str(path))
    points = np.asarray(mesh.points, dtype=float)
    out = {}
    for logical, arr_name in field_map.items():
        if arr_name not in mesh.point_data:
            raise KeyError(f"{path}: point-data array '{arr_name}' not found; "
                           f"available: {list(mesh.point_data)}")
        out[logical] = np.asarray(mesh.point_data[arr_name], dtype=float)
    return points, out


def save_ibfe_manifest_csv(frames, out_dir: str | Path,
                           t_values: Optional[np.ndarray] = None) -> Path:
    """Write an :class:`IBFEFrames` as a ``manifest.yaml`` + per-frame CSVs.

    Groups points by their (rounded) time column into frames. Primarily used to
    emit a *concrete template* of the expected directory layout that a real
    export can be matched against.
    """
    import yaml

    out_dir = Path(out_dir)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    dim = int(frames.spatial_dim)
    sp, ve = _spatial_cols(dim), _vel_cols(dim)
    nrm = [f"n{a}" for a in sp]
    uw = [f"{a}w" for a in "uvw"[:dim]]
    trac = [f"t{a}" for a in "xyz"[:dim]]
    frc = [f"f{a}" for a in "xyz"[:dim]]

    def _np(x):
        return x.detach().cpu().numpy()

    cf = _np(frames.coords_fluid)
    times = np.unique(np.round(cf[:, dim], 6)) if t_values is None else np.asarray(t_values)

    def _sel(coords, t):
        return np.isclose(coords[:, dim], t, atol=1e-5)

    def _write(path, header, matrix):
        np.savetxt(path, matrix, delimiter=",", header=",".join(header),
                   comments="", fmt="%.8e")

    manifest_frames = []
    cf_all = _np(frames.coords_fluid); vf = _np(frames.velocity_fluid)
    pf = _np(frames.pressure_fluid); ff = _np(frames.forcing_fluid)
    cw = _np(frames.coords_wall); nw = _np(frames.normals_wall)
    vw = _np(frames.velocity_wall); tw = _np(frames.traction_wall)
    have_mitral = frames.coords_mitral.shape[0] > 0
    cm = _np(frames.coords_mitral); vm = _np(frames.velocity_mitral)

    for i, t in enumerate(times):
        fm = _sel(cf_all, t)
        fluid_mat = np.concatenate(
            [cf_all[fm, :dim], vf[fm], pf[fm], ff[fm]], axis=1)
        _write(out_dir / "frames" / f"fluid_{i:03d}.csv", sp + ve + ["p"] + frc, fluid_mat)

        wm = _sel(cw, t)
        wall_mat = np.concatenate([cw[wm, :dim], nw[wm], vw[wm], tw[wm]], axis=1)
        _write(out_dir / "frames" / f"wall_{i:03d}.csv", sp + nrm + uw + trac, wall_mat)

        entry = {"t": float(t), "fluid": f"frames/fluid_{i:03d}.csv",
                 "wall": f"frames/wall_{i:03d}.csv"}
        if have_mitral:
            mm = _sel(cm, t)
            if mm.any():
                mit_mat = np.concatenate([cm[mm, :dim], vm[mm]], axis=1)
                _write(out_dir / "frames" / f"mitral_{i:03d}.csv", sp + ve, mit_mat)
                entry["mitral"] = f"frames/mitral_{i:03d}.csv"
        manifest_frames.append(entry)

    manifest = {"spatial_dim": dim, "cycle_period": float(frames.cycle_period),
                "rho": float(frames.rho), "mu": float(frames.mu),
                "frames": manifest_frames}
    mpath = out_dir / "manifest.yaml"
    with open(mpath, "w") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False)
    return mpath
