"""Corrado LV-phantom NIfTI (OSF 2naw4) → public-volume arrays.

The Wisconsin in-vitro LV release (Corrado et al., Magn Reson Med 2021,
DOI 10.1002/mrm.28698; data https://osf.io/2naw4/) ships reconstructed
4D-flow and PIV as ``VELX/VELY/VELZ`` NIfTI plus a cavity mask. Velocities
are stored in **mm/s** (see ``computeKineticEnergy.m`` in the analysis
repo). This adapter converts one scan directory into the public-volume
contract (m/s, metres, ``f = 0``).

It is **not** an IBFE dump and must not be treated as one. The informative
follow-up is window-count observability on the MRI velocity, not A vs
``fsi_param``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from .public_volume import save_public_volume_npz


_VEL_STEMS = ("VELX", "VELY", "VELZ")


def _require_nibabel():
    try:
        import nibabel as nib  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Corrado NIfTI conversion needs nibabel:  pip install nibabel"
        ) from exc
    return nib


def resolve_nii(directory: str | Path, stem: str) -> Path:
    """``STEM.nii.gz`` or ``STEM.nii`` in ``directory``."""
    directory = Path(directory)
    for name in (f"{stem}.nii.gz", f"{stem}.nii"):
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"{directory} has no {stem}.nii.gz / {stem}.nii")


def discover_mask(scan_dir: str | Path) -> Path:
    """Find the LV cavity mask that belongs to a Corrado scan directory.

    Layout on OSF::

        MRI_Data/Day 1/PC VIPR/30_Minutes/VELX.nii.gz
        MRI_Data/Day 1/PC VIPR/Mask_VIPR_Day1.nii.gz
        PIV_Data/Registered to Day 1/PC VIPR/Average Beat/VELX.nii.gz
        PIV_Data/Registered to Day 1/PC VIPR/MASK.nii.gz
    """
    scan_dir = Path(scan_dir)
    parent = scan_dir.parent
    text = str(scan_dir)
    traj = "VIPR" if "VIPR" in text else ("SOS" if "SOS" in text else None)
    if "Day 1" in text or "Day1" in text:
        day = "Day1"
    elif "Day 2" in text or "Day2" in text:
        day = "Day2"
    else:
        day = None
    guesses = []
    if traj and day:
        guesses.extend([
            parent / f"Mask_{traj}_{day}.nii.gz",
            parent / f"Mask_{traj}_{day}.nii",
        ])
    guesses.extend([
        parent / "MASK.nii.gz", parent / "MASK.nii",
        scan_dir / "MASK.nii.gz", scan_dir / "MASK.nii",
        parent / "Mask.nii.gz", parent / "Mask.nii",
    ])
    for path in guesses:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"no cavity mask next to {scan_dir}. Pass --mask explicitly.")


def _load_nifti(path: Path):
    nib = _require_nibabel()
    return nib.load(str(path))


def _as_meters(values: np.ndarray) -> np.ndarray:
    """Convert mm → m when the header is in millimetres (Corrado default)."""
    values = np.asarray(values, dtype=np.float64)
    typical = float(np.median(np.abs(values)[np.abs(values) > 0])) if np.any(
        values) else 0.0
    if typical >= 0.1:  # 0.1 mm or larger: header is in mm
        return values / 1000.0
    return values


def _time_step_seconds(zooms: Sequence[float], n_times: int,
                       period: Optional[float]) -> float:
    dt = float(zooms[3]) if len(zooms) > 3 else 0.0
    if dt > 2.0:  # almost certainly milliseconds
        dt = dt / 1000.0
    if dt > 1e-6:
        return dt
    cycle = float(period) if period is not None else 1.0
    return cycle / max(n_times, 1)


def _velocity_to_mps(vel: np.ndarray, unit: str) -> np.ndarray:
    unit = unit.lower().replace(" ", "")
    scale = {"mm/s": 1e-3, "mmps": 1e-3, "cm/s": 1e-2, "cmps": 1e-2,
             "m/s": 1.0, "mps": 1.0}.get(unit)
    if scale is None:
        raise ValueError(f"unknown velocity unit {unit!r}")
    return np.asarray(vel, dtype=np.float64) * scale


def load_corrado_scan(
    scan_dir: str | Path, *,
    mask_path: Optional[str | Path] = None,
    velocity_unit: str = "mm/s",
    period: Optional[float] = None,
    stride: int = 1,
    rho: float = 1060.0,
    mu: float = 0.0035,
) -> dict:
    """Load one Corrado scan directory into a public-volume dict (SI units)."""
    scan_dir = Path(scan_dir)
    if not scan_dir.is_dir():
        raise FileNotFoundError(scan_dir)
    comps = [_load_nifti(resolve_nii(scan_dir, stem)) for stem in _VEL_STEMS]
    arrays = [np.asanyarray(img.get_fdata()) for img in comps]
    if any(a.shape != arrays[0].shape for a in arrays[1:]):
        raise ValueError(
            "VELX/VELY/VELZ shapes differ: "
            + ", ".join(str(a.shape) for a in arrays))
    raw = arrays[0]
    if raw.ndim == 3:
        raw = raw[..., None]
        arrays = [a[..., None] for a in arrays]
    if raw.ndim != 4:
        raise ValueError(
            f"velocity NIfTI must be (X,Y,Z) or (X,Y,Z,T), got {raw.shape}")
    vel = np.stack(arrays, axis=-1)  # (X, Y, Z, T, 3)
    vel = np.moveaxis(vel, 3, 0)  # (T, X, Y, Z, 3)
    vel = _velocity_to_mps(vel, velocity_unit)

    mask_file = Path(mask_path) if mask_path else discover_mask(scan_dir)
    mask_img = _load_nifti(mask_file)
    mask = np.asanyarray(mask_img.get_fdata())
    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim == 3:
        pass
    elif mask.ndim == 4:
        mask = np.moveaxis(mask, 3, 0)
        if mask.shape[0] != vel.shape[0] or mask.shape[1:] != vel.shape[1:4]:
            raise ValueError(
                f"mask {tuple(mask.shape)} does not match velocity "
                f"{tuple(vel.shape[:4])}")
    else:
        raise ValueError(f"mask must be 3D or 4D, got {mask.shape}")
    if mask.ndim == 3 and mask.shape != vel.shape[1:4]:
        raise ValueError(
            f"mask {tuple(mask.shape)} != velocity spatial {tuple(vel.shape[1:4])}")

    header = comps[0].header
    zooms = header.get_zooms()
    spacing = _as_meters(np.asarray(zooms[:3], dtype=np.float64))
    affine = np.asarray(comps[0].affine, dtype=np.float64)
    origin = _as_meters(affine[:3, 3])

    n_full = vel.shape[0]
    dt = _time_step_seconds(zooms, n_full, period)
    times = np.arange(n_full, dtype=np.float64) * dt
    if period is None:
        period = float(times[-1] + dt) if n_full else 1.0

    stride = max(1, int(stride))
    vel = vel[::stride]
    times = times[::stride]
    if mask.ndim == 4:
        mask = mask[::stride]

    return {
        "velocity": vel,
        "mask": (mask != 0),
        "spacing": spacing,
        "origin": origin,
        "times": times,
        "cycle_period": float(period),
        "rho": float(rho),
        "mu": float(mu),
        "source": str(scan_dir),
        "mask_path": str(mask_file),
    }


def convert_corrado_scan(
    scan_dir: str | Path, out_path: str | Path, **kwargs
) -> Tuple[Path, dict]:
    """Write a public-volume NPZ from one Corrado scan directory."""
    volume = load_corrado_scan(scan_dir, **kwargs)
    path = save_public_volume_npz(
        out_path, velocity=volume["velocity"], mask=volume["mask"],
        spacing=volume["spacing"], origin=volume["origin"],
        times=volume["times"], cycle_period=volume["cycle_period"],
        rho=volume["rho"], mu=volume["mu"])
    return path, volume
