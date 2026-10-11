"""Eulerian stasis / residence-time *proxy* on a public velocity volume.

This is layer 1 of the CHF-thrombus motivation: a 3-component field is turned
into a **stasis map**, not a coagulation cascade. 4D-flow LV papers typically
flag cavity voxels with speed below ~10 cm/s for a large fraction of the RR
interval. That is a kinematic risk marker. It is **not** a thrombus.

A reconstructed field with speed relL2 ~ 0.4 will generally scramble this map.
The comparison here exists so that claim can be tested instead of assumed.

Units: velocity m/s, spacing m, times s. ``rho`` is kg/m³ for kinetic energy.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


DEFAULT_SPEED_THRESH = 0.10  # m/s = 10 cm/s (common 4D-flow LV stasis cut)
DEFAULT_PERSIST_FRAC = 0.50
DEFAULT_REL_P99 = 0.25


def resolve_speed_thresh(speed: np.ndarray, *,
                         speed_thresh: float = DEFAULT_SPEED_THRESH,
                         rel_p99: float = DEFAULT_REL_P99) -> float:
    """Clinical 10 cm/s on real 4D-flow; a relative cut on slow toy fields."""
    interior = np.asarray(speed, dtype=np.float64).ravel()
    interior = interior[np.isfinite(interior)]
    if interior.size == 0:
        return float(speed_thresh)
    p99 = float(np.quantile(interior, 0.99))
    # Real 4D-flow (p99 at or above the clinical cut): use 10 cm/s.
    # Slow manufactured toys (p99 << 10 cm/s): a relative cut, else every voxel
    # would be "stasis".
    if p99 >= speed_thresh:
        return float(speed_thresh)
    return float(rel_p99 * max(p99, 1e-12))


def _interior_speed(velocity: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Speed on masked voxels; ``velocity`` is ``(..., 3)``, mask broadcasts."""
    velocity = np.asarray(velocity, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    speed = np.linalg.norm(velocity, axis=-1)
    if mask.shape != speed.shape:
        if mask.ndim == speed.ndim - 1 and speed.shape[1:] == mask.shape:
            mask = np.broadcast_to(mask, speed.shape)
        else:
            raise ValueError(f"mask {mask.shape} vs speed {speed.shape}")
    return speed[mask]


def _mask_t(velocity: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 3:
        return np.broadcast_to(mask, velocity.shape[:4])
    if mask.ndim == 4 and mask.shape == velocity.shape[:4]:
        return mask
    raise ValueError(f"mask {mask.shape} vs velocity {velocity.shape}")


def persistent_stasis_mask(
    velocity: np.ndarray,
    mask: np.ndarray,
    *,
    speed_thresh: float,
    persist_frac: float = DEFAULT_PERSIST_FRAC,
) -> np.ndarray:
    """Boolean ``(Nx, Ny, Nz)`` persistent-stasis mask at a *fixed* cut."""
    velocity = np.asarray(velocity, dtype=np.float64)
    interior = _mask_t(velocity, mask)
    speed = np.linalg.norm(velocity, axis=-1)
    stasis_t = interior & (speed < float(speed_thresh))
    persist = stasis_t.mean(axis=0) >= persist_frac
    return persist & interior.any(axis=0)


def volume_stasis_report(
    velocity: np.ndarray,
    mask: np.ndarray,
    spacing: np.ndarray,
    *,
    speed_thresh: float = DEFAULT_SPEED_THRESH,
    persist_frac: float = DEFAULT_PERSIST_FRAC,
    rho: float = 1060.0,
    absolute: bool = False,
) -> Dict[str, float]:
    """Eulerian stasis on a public-volume array ``(T, Nx, Ny, Nz, 3)``.

    ``absolute=True`` keeps ``speed_thresh`` as given (use this for a
    reconstruction scored at the GT clinical cut).
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 5 or velocity.shape[-1] != 3:
        raise ValueError(f"velocity must be (T,Nx,Ny,Nz,3), got {velocity.shape}")
    interior = _mask_t(velocity, mask)
    spacing = np.asarray(spacing, dtype=np.float64).reshape(3)
    voxel_vol = float(np.prod(spacing))  # m³

    speed = np.linalg.norm(velocity, axis=-1)
    n_in = int(interior.sum())
    if n_in == 0:
        raise ValueError("cavity mask is empty")
    speed_in = speed[interior]
    thresh = (float(speed_thresh) if absolute
              else resolve_speed_thresh(speed_in, speed_thresh=speed_thresh))
    persist = persistent_stasis_mask(
        velocity, mask, speed_thresh=thresh, persist_frac=persist_frac)
    cavity_any = interior.any(axis=0)
    persist_vox = int(persist.sum())
    cavity_vox = int(cavity_any.sum())
    ke = 0.5 * rho * (speed_in ** 2)
    return {
        "speed_thresh": thresh,
        "stasis_frac": float((interior & (speed < thresh)).sum() / n_in),
        "persist_stasis_frac": float(persist_vox / max(cavity_vox, 1)),
        "persist_stasis_ml": persist_vox * voxel_vol * 1e6,
        "jet_ml": (cavity_vox - persist_vox) * voxel_vol * 1e6,
        "cavity_ml": cavity_vox * voxel_vol * 1e6,
        "speed_mean": float(speed_in.mean()),
        "speed_p99": float(np.quantile(speed_in, 0.99)),
        "ke_mean": float(ke.mean()),
        "n_interior": n_in,
        "n_cavity_voxels": cavity_vox,
    }


def compare_volume_stasis(
    vel_true: np.ndarray,
    vel_pred: np.ndarray,
    mask: np.ndarray,
    spacing: np.ndarray,
    *,
    speed_thresh: float = DEFAULT_SPEED_THRESH,
    persist_frac: float = DEFAULT_PERSIST_FRAC,
    rho: float = 1060.0,
) -> Dict[str, float]:
    """Score a reconstructed volume at the GT-resolved stasis cut."""
    gt = volume_stasis_report(
        vel_true, mask, spacing, speed_thresh=speed_thresh,
        persist_frac=persist_frac, rho=rho)
    thresh = float(gt["speed_thresh"])
    pred = volume_stasis_report(
        vel_pred, mask, spacing, speed_thresh=thresh,
        persist_frac=persist_frac, rho=rho, absolute=True)
    a = persistent_stasis_mask(
        vel_true, mask, speed_thresh=thresh, persist_frac=persist_frac)
    b = persistent_stasis_mask(
        vel_pred, mask, speed_thresh=thresh, persist_frac=persist_frac)
    inter = float(np.logical_and(a, b).sum())
    na, nb = float(a.sum()), float(b.sum())
    dice = (2.0 * inter / (na + nb)) if (na + nb) > 0 else 1.0
    return {
        "speed_thresh": thresh,
        "persist_stasis_ml_true": gt["persist_stasis_ml"],
        "persist_stasis_ml_pred": pred["persist_stasis_ml"],
        "jet_ml_true": gt["jet_ml"],
        "jet_ml_pred": pred["jet_ml"],
        "persist_stasis_dice": dice,
        "stasis_frac_true": gt["stasis_frac"],
        "stasis_frac_pred": pred["stasis_frac"],
        "speed_mean_true": gt["speed_mean"],
        "speed_mean_pred": pred["speed_mean"],
        "cavity_ml": gt["cavity_ml"],
    }


@torch.no_grad()
def predict_volume_velocity(
    model,
    origin,
    spacing,
    times,
    spatial_shape,
    *,
    batch_size: int = 32768,
) -> np.ndarray:
    """Evaluate a trained PINN on the public-volume grid ``(T, Nx, Ny, Nz, 3)``."""
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    spacing = np.asarray(spacing, dtype=np.float64).reshape(3)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    nx, ny, nz = (int(s) for s in spatial_shape)
    xs = origin[0] + np.arange(nx) * spacing[0]
    ys = origin[1] + np.arange(ny) * spacing[1]
    zs = origin[2] + np.arange(nz) * spacing[2]
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    xyz = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    n = xyz.shape[0]
    vel = np.zeros((times.shape[0], nx, ny, nz, 3), dtype=np.float32)
    for it, t in enumerate(times):
        chunks = []
        for i in range(0, n, batch_size):
            sl = xyz[i:i + batch_size]
            X = np.concatenate(
                [sl, np.full((sl.shape[0], 1), t, dtype=np.float64)], axis=1)
            xt = torch.as_tensor(X, dtype=dtype, device=device)
            f = model.fields(xt)
            keys = ("u", "v", "w") if "w" in f else ("u", "v")
            chunks.append(torch.cat([f[k] for k in keys], dim=1).cpu().numpy())
        pred = np.concatenate(chunks, axis=0)
        if pred.shape[1] == 2:
            pred = np.concatenate([pred, np.zeros((pred.shape[0], 1))], axis=1)
        vel[it] = pred.reshape(nx, ny, nz, 3)
    return vel


def compare_point_stasis(
    vel_true: torch.Tensor,
    vel_pred: torch.Tensor,
    *,
    speed_thresh: float = DEFAULT_SPEED_THRESH,
) -> Dict[str, float]:
    """Dice / fractions of the Eulerian stasis set on fluid point clouds."""
    vt = vel_true.detach()
    vp = vel_pred.detach()
    if vt.shape != vp.shape or vt.shape[-1] < 2:
        raise ValueError(f"velocity shapes {tuple(vt.shape)} vs {tuple(vp.shape)}")
    st = torch.linalg.norm(vt, dim=1)
    sp = torch.linalg.norm(vp, dim=1)
    thresh = resolve_speed_thresh(st.cpu().numpy(), speed_thresh=speed_thresh)
    a = st < thresh
    b = sp < thresh
    inter = float((a & b).sum().item())
    na = float(a.sum().item())
    nb = float(b.sum().item())
    dice = (2.0 * inter / (na + nb)) if (na + nb) > 0 else 1.0
    # Eulerian RT proxy 1/max(|u|, eps): wrong where the field is slow.
    eps = max(0.05 * float(st.quantile(0.5).clamp_min(1e-4)), 1e-4)
    rt_t = 1.0 / st.clamp_min(eps)
    rt_p = 1.0 / sp.clamp_min(eps)
    rt_rel = float(torch.linalg.norm(rt_p - rt_t) / (torch.linalg.norm(rt_t) + 1e-12))
    return {
        "stasis_speed_thresh": thresh,
        "stasis_frac_true": float(na / max(st.numel(), 1)),
        "stasis_frac_pred": float(nb / max(sp.numel(), 1)),
        "stasis_dice": dice,
        "rt_proxy_relL2": rt_rel,
        "speed_mean_true": float(st.mean()),
        "speed_mean_pred": float(sp.mean()),
    }


def evaluate_stasis_ibfe(vel_true: torch.Tensor, vel_pred: torch.Tensor,
                         **kwargs) -> Dict[str, float]:
    """Alias used from :func:`evaluate_ibfe`."""
    return compare_point_stasis(vel_true, vel_pred, **kwargs)


def volume_from_public_npz(path) -> Dict:
    """Load the arrays ``volume_stasis_report`` needs from a public-volume NPZ."""
    data = np.load(path)
    try:
        return {
            "velocity": data["velocity"],
            "mask": data["mask"],
            "spacing": data["spacing"],
            "origin": data["origin"],
            "times": data["times"],
            "rho": float(data["_meta"][1]) if "_meta" in data.files else 1060.0,
        }
    finally:
        data.close()
