"""Echo-plane slab coverage of the Eulerian jet (no training, no thrombus).

The 1/2/3-window Corrado runs used idealized *volume* beams. Clinical echo
samples thin slabs (A4C / A2C / PLAX / PSAX). This module asks whether the
10 cm/s through-flow jet even sits in those slabs.

Views are the project's AABB-canonical set
(:func:`~pinnecho.data.acquisition.standard_views_from_extents`): long axis
``z``, A4C = ``y``-midplane. That is the same convention as
``plane_coverage``, not a cardiologist placing probes on the Corrado phantom.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from pinnecho.data.acquisition import standard_views_from_extents
from pinnecho.eval.stasis import (
    DEFAULT_PERSIST_FRAC,
    DEFAULT_SPEED_THRESH,
    _mask_t,
    persistent_stasis_mask,
)

# Same planar protocols as ``plane_coverage.default_coverage_conditions``
# (without the idealized whole-volume point-window row).
DEFAULT_PROTOCOLS: Tuple[Tuple[str, ...], ...] = (
    ("a4c",),
    ("a4c", "a2c"),
    ("a4c", "plax"),
    ("a4c", "a2c", "plax", "psax"),
)

DEFAULT_THICKNESS_FRAC = 0.12


def voxel_centers(origin, spacing, spatial_shape) -> np.ndarray:
    """Voxel-centre coordinates ``(Nx, Ny, Nz, 3)`` in metres."""
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    spacing = np.asarray(spacing, dtype=np.float64).reshape(3)
    nx, ny, nz = (int(s) for s in spatial_shape)
    xs = origin[0] + np.arange(nx) * spacing[0]
    ys = origin[1] + np.arange(ny) * spacing[1]
    zs = origin[2] + np.arange(nz) * spacing[2]
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([xx, yy, zz], axis=-1)


def _spatial_cavity(mask: np.ndarray, spatial_shape) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if m.ndim == 4:
        m = m.any(axis=0)
    spatial_shape = tuple(int(s) for s in spatial_shape)
    if m.shape != spatial_shape:
        raise ValueError(f"mask {m.shape} vs grid {spatial_shape}")
    return m


def views_from_cavity(
    mask: np.ndarray,
    origin,
    spacing,
    *,
    thickness_frac: float = DEFAULT_THICKNESS_FRAC,
    views: Sequence[str] | None = None,
):
    """AABB-canonical echo views sized to the cavity mask."""
    cavity = _spatial_cavity(mask, mask.shape[-3:] if mask.ndim >= 3 else mask.shape)
    xyz = voxel_centers(origin, spacing, cavity.shape)
    pts = xyz[cavity]
    if pts.size == 0:
        raise ValueError("cavity mask is empty")
    center = pts.mean(axis=0)
    extents = pts.max(axis=0) - pts.min(axis=0)
    thickness = float(thickness_frac) * float(np.mean(extents))
    return standard_views_from_extents(
        center.tolist(), extents.tolist(), 3, thickness=thickness,
        views=views, dtype=torch.float64), float(thickness)


def union_slab_mask(xyz: np.ndarray, planes) -> np.ndarray:
    """Boolean ``xyz.shape[:-1]``: voxel centre lies in at least one slab."""
    flat = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    coords = torch.as_tensor(flat, dtype=torch.float64)
    hit = np.zeros(flat.shape[0], dtype=bool)
    for plane in planes:
        hit |= plane.select_mask(coords).cpu().numpy()
    return hit.reshape(xyz.shape[:-1])


def _along_beam_frac(vbar: np.ndarray, xyz: np.ndarray, planes, sel: np.ndarray) -> float:
    """Mean over ``sel`` of max_plane |v̄·b| / |v̄|."""
    idx = np.flatnonzero(sel.ravel())
    if idx.size == 0:
        return float("nan")
    pts = xyz.reshape(-1, 3)[idx]
    vel = vbar.reshape(-1, 3)[idx]
    speed = np.linalg.norm(vel, axis=1)
    best = np.zeros(idx.size, dtype=np.float64)
    coords = torch.as_tensor(pts, dtype=torch.float64)
    for plane in planes:
        in_p = plane.select_mask(coords).cpu().numpy()
        if not in_p.any():
            continue
        beam = plane.beams(coords).cpu().numpy()
        along = np.abs((vel * beam).sum(axis=1)) / np.clip(speed, 1e-12, None)
        best[in_p] = np.maximum(best[in_p], along[in_p])
    return float(best.mean())


def protocol_coverage(
    xyz: np.ndarray,
    cavity: np.ndarray,
    persist: np.ndarray,
    jet: np.ndarray,
    vbar: np.ndarray,
    planes,
    voxel_ml: float,
    thickness_m: float,
    name: str,
) -> Dict[str, Any]:
    slab = union_slab_mask(xyz, planes) & cavity
    n_jet = int(jet.sum())
    n_jet_hit = int((jet & slab).sum())
    return {
        "name": name,
        "n_views": float(len(planes)),
        "thickness_m": float(thickness_m),
        "cavity_ml_in_slab": float(slab.sum()) * voxel_ml,
        "persist_ml_in_slab": float((persist & slab).sum()) * voxel_ml,
        "jet_ml": float(n_jet) * voxel_ml,
        "jet_ml_in_slab": float(n_jet_hit) * voxel_ml,
        "jet_frac_in_slab": (
            float(n_jet_hit / n_jet) if n_jet else float("nan")),
        "jet_along_beam_frac": _along_beam_frac(vbar, xyz, planes, jet & slab),
    }


def slab_coverage_report(
    velocity: np.ndarray,
    mask: np.ndarray,
    spacing,
    origin,
    *,
    speed_thresh: float = DEFAULT_SPEED_THRESH,
    persist_frac: float = DEFAULT_PERSIST_FRAC,
    thickness_frac: float = DEFAULT_THICKNESS_FRAC,
    protocols: Sequence[Sequence[str]] = DEFAULT_PROTOCOLS,
) -> List[Dict[str, Any]]:
    """Coverage of the 10 cm/s jet by each echo-plane protocol. No training."""
    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 5 or velocity.shape[-1] != 3:
        raise ValueError(f"velocity must be (T,Nx,Ny,Nz,3), got {velocity.shape}")
    spacing = np.asarray(spacing, dtype=np.float64).reshape(3)
    cavity = _mask_t(velocity, mask).any(axis=0)
    persist = persistent_stasis_mask(
        velocity, mask, speed_thresh=float(speed_thresh),
        persist_frac=persist_frac)
    jet = cavity & np.logical_not(persist)
    xyz = voxel_centers(origin, spacing, cavity.shape)
    vbar = velocity.mean(axis=0)
    voxel_ml = float(np.prod(spacing)) * 1e6
    rows = []
    for views in protocols:
        views = tuple(views)
        planes, thickness = views_from_cavity(
            cavity, origin, spacing, thickness_frac=thickness_frac, views=views)
        rows.append(protocol_coverage(
            xyz, cavity, persist, jet, vbar, planes, voxel_ml, thickness,
            name="+".join(views)))
    return rows
