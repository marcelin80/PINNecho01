"""Public 4D-flow / phantom volumes → :class:`IBFEFrames` (no oracle forcing).

This is the *non-IBFE* on-ramp: a structured Eulerian velocity volume plus a
cavity mask, the form public 4D-flow MRI and in-vitro LV phantoms actually
ship. It is **not** an IBAMR/IBFE dump and must not be treated as one.

What is filled
--------------
* ``velocity_fluid`` from the volume (the only measured / published field).
* ``coords_wall`` / ``normals_wall`` / ``velocity_wall`` from the mask boundary
  (no-slip proxy: wall velocity = fluid velocity on the boundary voxels).
* ``forcing_fluid = 0`` and ``traction_wall = 0`` -- there is no IB-spread
  structural force and no Cauchy traction in a 4D-flow / phantom release.
* ``pressure_fluid = 0`` unless the volume carries a pressure array (some
  phantoms do; 4D-flow MRI does not).

What this is for
----------------
A vs deliverable-B (``fsi_param``) on a **non-manufactured** 3D field. Do
**not** train ``fsi_informed`` on these frames: the oracle forcing is absent
by construction, so the FSI backbone collapses to the kinematic baseline.

On-disk contract (``*public*.npz`` or any NPZ with ``velocity`` + ``mask``)
--------------------------------------------------------------------------
::

    velocity   (T, Nx, Ny, Nz, 3)   m/s
    mask       (T, Nx, Ny, Nz) or (Nx, Ny, Nz)   cavity interior (nonzero)
    spacing    (3,)                 m
    origin     (3,)                 m   (voxel-centre of index (0,0,0))
    times      (T,)                 s
    pressure   (T, Nx, Ny, Nz)      Pa   (optional)
    _meta      [cycle_period, rho, mu, spatial_dim]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.ndimage import binary_erosion

from .load_ibfe_output import IBFEFrames

PUBLIC_VOLUME_KEYS = ("velocity", "mask", "spacing", "origin", "times")


def is_public_volume_npz(path: str | Path) -> bool:
    """True if ``path`` is an NPZ with the public-volume keys (not an IBFE bundle)."""
    path = Path(path)
    if path.suffix != ".npz" or not path.is_file():
        return False
    with np.load(path) as data:
        return all(k in data.files for k in ("velocity", "mask"))


def has_oracle_forcing(frames: IBFEFrames) -> bool:
    """False for public-volume frames (``f`` is identically zero)."""
    if frames.forcing_fluid.numel() == 0:
        return False
    return float(frames.forcing_fluid.abs().max()) > 0.0


def frames_from_public_volume(
    velocity: np.ndarray,
    mask: np.ndarray,
    spacing: Sequence[float],
    origin: Sequence[float],
    times: Sequence[float],
    pressure: Optional[np.ndarray] = None,
    cycle_period: Optional[float] = None,
    rho: float = 1060.0,
    mu: float = 0.0035,
    n_fluid: Optional[int] = None,
    n_wall: Optional[int] = None,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> IBFEFrames:
    """Pack a structured velocity volume into :class:`IBFEFrames`.

    ``forcing_fluid`` and ``traction_wall`` are zeros. ``pressure_fluid`` is
    zero unless ``pressure`` is given. ``n_fluid`` / ``n_wall`` cap the *total*
    point counts (evenly across frames) so a fine 4D-flow grid stays tractable.
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 5 or velocity.shape[-1] != 3:
        raise ValueError(
            f"velocity must be (T, Nx, Ny, Nz, 3), got {velocity.shape}")
    t_len, nx, ny, nz, _ = velocity.shape
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = np.broadcast_to(mask.astype(bool), (t_len, nx, ny, nz)).copy()
    elif mask.ndim == 4:
        mask = mask.astype(bool)
        if mask.shape != (t_len, nx, ny, nz):
            raise ValueError(f"mask shape {mask.shape} != {(t_len, nx, ny, nz)}")
    else:
        raise ValueError(f"mask must be (Nx,Ny,Nz) or (T,Nx,Ny,Nz), got {mask.shape}")

    spacing = np.asarray(spacing, dtype=np.float64).reshape(3)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if times.shape[0] != t_len:
        raise ValueError(f"times length {times.shape[0]} != T={t_len}")
    if pressure is not None:
        pressure = np.asarray(pressure, dtype=np.float64)
        if pressure.shape != (t_len, nx, ny, nz):
            raise ValueError(
                f"pressure shape {pressure.shape} != {(t_len, nx, ny, nz)}")

    period = float(cycle_period) if cycle_period is not None else float(
        max(times[-1] - times[0], times[1] - times[0] if t_len > 1 else 1.0))
    rng = np.random.default_rng(seed)
    n_f_each = None if n_fluid is None else max(1, int(n_fluid) // t_len)
    n_w_each = None if n_wall is None else max(1, int(n_wall) // t_len)

    fluid_c, fluid_v, fluid_p = [], [], []
    wall_c, wall_n, wall_v = [], [], []
    for k in range(t_len):
        fc, fv, fp, wc, wn, wv = _frame_points(
            velocity[k], mask[k],
            None if pressure is None else pressure[k],
            spacing, origin, float(times[k]))
        fc, fv, fp = _cap(rng, n_f_each, fc, fv, fp)
        wc, wn, wv = _cap(rng, n_w_each, wc, wn, wv)
        fluid_c.append(fc); fluid_v.append(fv); fluid_p.append(fp)
        wall_c.append(wc); wall_n.append(wn); wall_v.append(wv)

    coords_fluid = torch.as_tensor(np.concatenate(fluid_c, 0), dtype=dtype)
    velocity_fluid = torch.as_tensor(np.concatenate(fluid_v, 0), dtype=dtype)
    pressure_fluid = torch.as_tensor(np.concatenate(fluid_p, 0), dtype=dtype)
    coords_wall = torch.as_tensor(np.concatenate(wall_c, 0), dtype=dtype)
    normals_wall = torch.as_tensor(np.concatenate(wall_n, 0), dtype=dtype)
    velocity_wall = torch.as_tensor(np.concatenate(wall_v, 0), dtype=dtype)
    n_f, n_w = coords_fluid.shape[0], coords_wall.shape[0]
    if n_f == 0:
        raise ValueError("public volume has no interior (mask) voxels")
    if n_w == 0:
        raise ValueError("public volume has no wall voxels (mask fills the grid "
                         "without a boundary, or is empty after erosion)")

    return IBFEFrames(
        coords_fluid=coords_fluid,
        velocity_fluid=velocity_fluid,
        pressure_fluid=pressure_fluid,
        forcing_fluid=torch.zeros(n_f, 3, dtype=dtype),
        coords_wall=coords_wall,
        normals_wall=normals_wall,
        velocity_wall=velocity_wall,
        traction_wall=torch.zeros(n_w, 3, dtype=dtype),
        coords_mitral=torch.zeros(0, 4, dtype=dtype),
        velocity_mitral=torch.zeros(0, 3, dtype=dtype),
        coords_aortic=torch.zeros(0, 4, dtype=dtype),
        velocity_aortic=torch.zeros(0, 3, dtype=dtype),
        cycle_period=period, rho=float(rho), mu=float(mu),
        spatial_dim=3,
    )


def _cap(rng, n_keep, *arrays):
    n = arrays[0].shape[0]
    if n_keep is None or n <= n_keep or n == 0:
        return arrays
    idx = rng.choice(n, size=n_keep, replace=False)
    return tuple(a[idx] for a in arrays)


def _frame_points(vel, mask, pressure, spacing, origin, t):
    """Interior + boundary point clouds for one time frame."""
    nx, ny, nz = mask.shape
    ix, iy, iz = np.nonzero(mask)
    xyz = origin + np.stack([ix, iy, iz], axis=1) * spacing
    coords = np.concatenate([xyz, np.full((ix.size, 1), t)], axis=1)
    velocity = vel[ix, iy, iz]
    if pressure is None:
        p = np.zeros((ix.size, 1), dtype=np.float64)
    else:
        p = pressure[ix, iy, iz].reshape(-1, 1)

    boundary = _boundary_mask(mask)
    bx, by, bz = np.nonzero(boundary)
    if bx.size == 0:
        return (coords, velocity, p,
                np.zeros((0, 4)), np.zeros((0, 3)), np.zeros((0, 3)))
    wxyz = origin + np.stack([bx, by, bz], axis=1) * spacing
    wcoords = np.concatenate([wxyz, np.full((bx.size, 1), t)], axis=1)
    wvel = vel[bx, by, bz]
    normals = _outward_normals(mask, bx, by, bz)
    return coords, velocity, p, wcoords, normals, wvel


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    """Interior voxels that touch the exterior (6-connected)."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, iterations=1, border_value=0)
    return mask & ~eroded


def _outward_normals(mask: np.ndarray, ix, iy, iz) -> np.ndarray:
    """Outward unit normals from ``-∇ mask`` (mask = 1 inside)."""
    m = mask.astype(np.float64)
    gx, gy, gz = np.gradient(m)
    vec = -np.stack([gx[ix, iy, iz], gy[ix, iy, iz], gz[ix, iy, iz]], axis=1)
    nrm = np.linalg.norm(vec, axis=1, keepdims=True)
    # Degenerate gradient (flat mask, 1-voxel islands): fall back to
    # displacement from the interior centroid.
    centroid = np.array(np.nonzero(mask), dtype=np.float64).mean(axis=1)
    fallback = np.stack([ix, iy, iz], axis=1).astype(np.float64) - centroid
    degenerates = nrm[:, 0] < 1e-12
    vec[degenerates] = fallback[degenerates]
    nrm = np.linalg.norm(vec, axis=1, keepdims=True).clip(min=1e-12)
    return vec / nrm


# --------------------------------------------------------------------------- #
# On-disk I/O
# --------------------------------------------------------------------------- #
def save_public_volume_npz(
    path: str | Path, *,
    velocity: np.ndarray, mask: np.ndarray,
    spacing: Sequence[float], origin: Sequence[float], times: Sequence[float],
    pressure: Optional[np.ndarray] = None,
    cycle_period: float, rho: float, mu: float,
) -> Path:
    """Write the public-volume NPZ contract."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "velocity": np.asarray(velocity),
        "mask": np.asarray(mask).astype(np.uint8),
        "spacing": np.asarray(spacing, dtype=np.float64).reshape(3),
        "origin": np.asarray(origin, dtype=np.float64).reshape(3),
        "times": np.asarray(times, dtype=np.float64).reshape(-1),
        "_meta": np.array([float(cycle_period), float(rho), float(mu), 3.0],
                          dtype=np.float64),
    }
    if pressure is not None:
        arrays["pressure"] = np.asarray(pressure)
    np.savez(path, **arrays)
    return path


def load_public_volume_npz(
    path: str | Path, *,
    n_fluid: Optional[int] = None, n_wall: Optional[int] = None,
    seed: int = 0, dtype: torch.dtype = torch.float32,
) -> IBFEFrames:
    """Load a public-volume NPZ into :class:`IBFEFrames` (``f = 0``)."""
    path = Path(path)
    data = np.load(path)
    try:
        missing = [k for k in PUBLIC_VOLUME_KEYS if k not in data.files]
        if missing:
            raise KeyError(f"{path} is missing public-volume keys {missing}")
        meta = data["_meta"] if "_meta" in data.files else np.array(
            [float(data["times"][-1]), 1060.0, 0.0035, 3.0])
        return frames_from_public_volume(
            data["velocity"], data["mask"], data["spacing"], data["origin"],
            data["times"],
            pressure=data["pressure"] if "pressure" in data.files else None,
            cycle_period=float(meta[0]), rho=float(meta[1]), mu=float(meta[2]),
            n_fluid=n_fluid, n_wall=n_wall, seed=seed, dtype=dtype)
    finally:
        data.close()


# --------------------------------------------------------------------------- #
# Synthetic stand-in (tests / example export; not a public dataset)
# --------------------------------------------------------------------------- #
def rasterize_synthetic_lv_3d(
    config=None, *,
    nx: int = 16, ny: int = 16, nz: int = 20, n_frames: int = 4,
    pad: float = 0.12, with_pressure: bool = False,
    dtype: torch.dtype = torch.float64,
) -> Tuple[dict, object]:
    """Sample :class:`SyntheticLV3D` onto a Cartesian grid + cavity mask.

    Returns ``(volume_dict, lv)`` where ``volume_dict`` has the public-volume
    keys. Pressure is omitted unless ``with_pressure=True`` (4D-flow default:
    no pressure). Forcing is never written -- that is the point.
    """
    from ..config import Config
    from .synthetic_lv_3d import SyntheticLV3D

    if config is None:
        config = Config()
    lv = SyntheticLV3D(config.geometry, config.flow, dtype=dtype)
    # Bounding box: peak semi-axes plus pad, so the wall stays inside the grid.
    smax = 1.0 + float(config.geometry.strain_amplitude)
    rx = float(config.geometry.r0_x) * (smax ** 0.5) * (1.0 + pad)
    ry = float(config.geometry.r0_y) * (smax ** 0.5) * (1.0 + pad)
    rz = float(config.geometry.r0_z) * smax * (1.0 + pad)
    cx, cy, cz = (float(config.geometry.center_x),
                  float(config.geometry.center_y),
                  float(config.geometry.center_z))
    origin = np.array([cx - rx, cy - ry, cz - rz], dtype=np.float64)
    spacing = np.array([2 * rx / max(nx - 1, 1),
                        2 * ry / max(ny - 1, 1),
                        2 * rz / max(nz - 1, 1)], dtype=np.float64)
    times = np.linspace(0.0, float(config.flow.period), n_frames)
    xs = origin[0] + np.arange(nx) * spacing[0]
    ys = origin[1] + np.arange(ny) * spacing[1]
    zs = origin[2] + np.arange(nz) * spacing[2]
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")

    velocity = np.zeros((n_frames, nx, ny, nz, 3), dtype=np.float64)
    mask = np.zeros((n_frames, nx, ny, nz), dtype=bool)
    pressure = np.zeros((n_frames, nx, ny, nz), dtype=np.float64) if with_pressure else None
    for k, t in enumerate(times):
        coords = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel(),
                           np.full(XX.size, t)], axis=1)
        X = torch.as_tensor(coords, dtype=dtype)
        _, _, _, rho2 = lv._normalised(X)
        inside = (rho2.reshape(-1) <= 1.0).cpu().numpy()
        mask[k].ravel()[:] = inside
        uvw = lv._velocity_from_graph(X).detach().cpu().numpy()
        velocity[k].reshape(-1, 3)[:] = uvw
        if with_pressure:
            pressure[k].ravel()[:] = lv.pressure(X).detach().cpu().numpy().reshape(-1)

    volume = {
        "velocity": velocity, "mask": mask, "spacing": spacing,
        "origin": origin, "times": times,
        "cycle_period": float(config.flow.period),
        "rho": float(config.flow.density), "mu": float(config.flow.viscosity),
    }
    if with_pressure:
        volume["pressure"] = pressure
    return volume, lv


def synthetic_public_volume_frames(config=None, *,
                                   nx: int = 16, ny: int = 16, nz: int = 20,
                                   n_frames: int = 4, n_fluid: int = 2000,
                                   n_wall: int = 500, with_pressure: bool = False,
                                   seed: int = 0,
                                   dtype: torch.dtype = torch.float32) -> IBFEFrames:
    """End-to-end stand-in: rasterise the 3D synthetic LV as a public volume."""
    volume, _ = rasterize_synthetic_lv_3d(
        config, nx=nx, ny=ny, nz=nz, n_frames=n_frames,
        with_pressure=with_pressure)
    return frames_from_public_volume(
        volume["velocity"], volume["mask"], volume["spacing"], volume["origin"],
        volume["times"], pressure=volume.get("pressure"),
        cycle_period=volume["cycle_period"], rho=volume["rho"], mu=volume["mu"],
        n_fluid=n_fluid, n_wall=n_wall, seed=seed, dtype=dtype)
