"""Pack an OSU-MR CORe recon (phase, not m/s) into a public-volume NPZ.

``outputs.thetaX/Y/Z`` from ``main_recon.m`` are background-corrected
phase in radians. Dataset #11 (Zenodo ``4D_Flow_Cartesian_Dataset_11.mat``)
has VENC **200 cm/s** (Arshad et al., *Magn Reson Med* 2024, Fig. 4).
Do not rewrite CORe in Python.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np


DATASET11_VENC_MPS = 2.00  # 200 cm/s; paper Fig. 4, dataset #11


def phase_to_velocity_mps(theta: np.ndarray, venc_mps: float) -> np.ndarray:
    """Map phase in ``(-π, π]`` to velocity. ``v = θ/π · VENC``."""
    return np.asarray(theta, dtype=np.float64) / np.pi * float(venc_mps)


def _as_txyz(arr: np.ndarray) -> np.ndarray:
    """MATLAB ``(X, Y, Z, T)`` → ``(T, X, Y, Z)``."""
    arr = np.asarray(arr)
    if arr.ndim != 4:
        raise ValueError(f"expected (X,Y,Z,T) or (T,X,Y,Z), got {arr.shape}")
    # Heuristic: cardiac bins are 20 in this dataset; put the size-20 axis first
    # only when the caller did not already pass (T, X, Y, Z).
    return np.transpose(arr, (3, 0, 1, 2))


def velocity_from_thetas(
    theta_x: np.ndarray,
    theta_y: np.ndarray,
    theta_z: np.ndarray,
    venc_mps: float,
) -> np.ndarray:
    """Stack three phase volumes into ``(T, Nx, Ny, Nz, 3)`` m/s."""
    vx = phase_to_velocity_mps(_as_txyz(theta_x), venc_mps)
    vy = phase_to_velocity_mps(_as_txyz(theta_y), venc_mps)
    vz = phase_to_velocity_mps(_as_txyz(theta_z), venc_mps)
    if vx.shape != vy.shape or vx.shape != vz.shape:
        raise ValueError(f"theta shapes {vx.shape} {vy.shape} {vz.shape}")
    return np.stack([vx, vy, vz], axis=-1)


def magnitude_mask(xhat: np.ndarray, percentile: float = 70.0) -> np.ndarray:
    """Time-mean |xHat| threshold. **Not** an LV cavity segmentation."""
    mag = np.abs(_as_txyz(xhat))
    mean = mag.mean(axis=0)
    cut = float(np.percentile(mean[np.isfinite(mean)], percentile))
    return mean >= cut


def load_core_outputs(path) -> Dict[str, np.ndarray]:
    """Read ``outputs.thetaX/Y/Z`` and ``outputs.xHat`` from a CORe ``.mat``."""
    from scipy.io import loadmat
    data = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if "outputs" not in data:
        raise KeyError(f"{path} has no 'outputs' (is this the CORe save, not k-space?)")
    out = data["outputs"]
    needed = ("thetaX", "thetaY", "thetaZ", "xHat")
    missing = [k for k in needed if not hasattr(out, k)]
    if missing:
        raise KeyError(f"{path} outputs missing {missing}")
    return {k: np.asarray(getattr(out, k)) for k in needed}


def pack_osu_public_volume(
    thetas: Dict[str, np.ndarray],
    *,
    venc_mps: float,
    spacing_m: Sequence[float],
    origin_m: Sequence[float] = (0.0, 0.0, 0.0),
    period_s: float = 1.0,
    mask: Optional[np.ndarray] = None,
    mag_percentile: float = 70.0,
) -> Dict:
    vel = velocity_from_thetas(
        thetas["thetaX"], thetas["thetaY"], thetas["thetaZ"], venc_mps)
    if mask is None:
        if "xHat" not in thetas:
            raise ValueError("need xHat for a magnitude mask, or pass mask=")
        mask = magnitude_mask(thetas["xHat"], mag_percentile)
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 4:
        mask = mask.any(axis=0)
    if mask.shape != vel.shape[1:4]:
        raise ValueError(f"mask {mask.shape} vs grid {vel.shape[1:4]}")
    t_len = vel.shape[0]
    times = np.linspace(0.0, float(period_s), t_len, endpoint=False)
    return {
        "velocity": vel,
        "mask": mask,
        "spacing": np.asarray(spacing_m, dtype=np.float64).reshape(3),
        "origin": np.asarray(origin_m, dtype=np.float64).reshape(3),
        "times": times,
        "cycle_period": float(period_s),
        "venc_mps": float(venc_mps),
        "mask_is_lv": False,
    }
