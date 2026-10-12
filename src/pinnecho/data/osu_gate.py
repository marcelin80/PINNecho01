"""Pre-registered fail-cheap gate for OSU rest dataset #11.

Corrado A/B/C/blur are closed. After MATLAB CORe, this is the next measured
check — not a train, not a thrombus, not an LV claim from a magnitude cut.

The original check 1 (LV ``speed_p99`` ≥ 10 cm/s) is **retired**. Peak
``max_t |u|`` is positively biased over 20 frames; the placeholder p99 is
already 35 cm/s and unmasked air hits VENC·√3 ≈ 344 cm/s. That cut cannot
fail on noise. Locked replacements, written before any painted LV is scored:

1. ``lv_vs_tissue`` — spatial median of per-voxel **time-RMS** |u| on an
   LV mask must be ≥ **3×** the same statistic on a static-tissue ROI
   (chest-wall muscle, not lung). No tissue ROI is PENDING. Reported
   ``legacy_speed_p99`` is diagnostic only.
2. ``time_consistency`` — AC power of the spatial-mean velocity curve
   must have ≥ **0.55** in Fourier harmonics 1–4 (cardiac). Flat /
   DC-only is noise or static tissue.
3. ``aao_sv`` — ascending-aorta stroke volume vs the paper 2D-PC ref
   **93.6 mL/beat** (Arshad et al., *Magn Reson Med* 2024, Fig. 4 /
   dataset #11). Relative error > 0.40 is FAIL. No Aao plane is PENDING.
4. ``valve_balance`` — |SV_mitral − SV_aorta| / max(|SV|) ≤ **0.40**.
   Integration cancels zero-mean noise. Missing mitral or Aao plane is
   PENDING.
5. ``jet_chamber`` — 10 cm/s remainder (``cavity & ~persist``) ≥ 50 %
   inside the LV and < 50 % inside an optional aorta/LVOT *region*.
   A thin Aao plane is **only** for stroke volume.

``outputs.theta*`` from ``main_recon`` already have cmapx/y/z applied.
This packer does not re-estimate background phase; residual static-tissue
speed is the leftover.

PASS is all five. PASS does not authorize a thrombus solver or the 5.1 GB
cine. FAIL or no confident LV contour in 30–45 min → Charité inquiry, not
CFD. The discarded auto-box ``43:60 59:69 15:37`` is not an LV.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from scipy.ndimage import binary_fill_holes, label as cc_label

from pinnecho.eval.stasis import (
    peak_speed, persistent_stasis_mask, volume_stasis_report,
)


DATASET11_AAO_SV_ML = 93.6
DATASET11_AAO_PEAK_MPS = 1.80
# Retired as a pass/fail cut. Kept so reports can show that noise clears it.
LEGACY_SPEED_P99_MIN_MPS = 0.10
LV_TISSUE_RATIO_MIN = 3.0
FOURIER_LOW_HARMONICS = 4
FOURIER_LOW_FRAC_MIN = 0.55
AAO_SV_REL_TOL = 0.40
VALVE_SV_REL_TOL = 0.40
JET_IN_LV_MIN_FRAC = 0.50
JET_IN_AORTA_MAX_FRAC = 0.50
JET_SPEED_THRESH = 0.10
JET_PERSIST_FRAC = 0.50
# Compat alias — do not use as a gate cut.
SPEED_P99_MIN_MPS = LEGACY_SPEED_P99_MIN_MPS

THRESHOLDS = {
    "lv_tissue_ratio_min": LV_TISSUE_RATIO_MIN,
    "fourier_low_harmonics": FOURIER_LOW_HARMONICS,
    "fourier_low_frac_min": FOURIER_LOW_FRAC_MIN,
    "aao_sv_ref_ml": DATASET11_AAO_SV_ML,
    "aao_sv_rel_tol": AAO_SV_REL_TOL,
    "valve_sv_rel_tol": VALVE_SV_REL_TOL,
    "jet_in_lv_min_frac": JET_IN_LV_MIN_FRAC,
    "jet_in_aorta_max_frac": JET_IN_AORTA_MAX_FRAC,
    "jet_speed_thresh_mps": JET_SPEED_THRESH,
    "jet_persist_frac": JET_PERSIST_FRAC,
    "legacy_speed_p99_min_mps": LEGACY_SPEED_P99_MIN_MPS,
}

GATE_NOTES = (
    "Magnitude mask is not an LV segmentation.",
    "QA NIfTI intensities are display-scaled 0-1000, not cm/s.",
    "Hide peak_speed while painting; do not trace PC-MRA as a wall.",
    "Save rest_lv_mask_v1 / v2; do not overwrite after seeing scores.",
    "Auto-box 43:60 59:69 15:37 (~53 mL, anterior) is not LV.",
    "CORe theta* already has cmap background-phase correction.",
    "PASS does not authorize a thrombus solver.",
    "Do not download the 5.1 GB cine until PASS.",
    "FAIL or no confident LV in 30-45 min → Charité inquiry, not CFD.",
)


def as_spatial_mask(mask: np.ndarray, grid_shape: Sequence[int]) -> np.ndarray:
    """Broadcast a 3-D or 4-D cavity mask onto ``(Nx, Ny, Nz)``."""
    mask = np.asarray(mask)
    if mask.dtype != bool:
        mask = mask.astype(bool)
    grid = tuple(int(s) for s in grid_shape)
    if mask.ndim == 4:
        if mask.shape[1:] != grid and mask.shape[:3] != grid:
            raise ValueError(f"mask {mask.shape} vs grid {grid}")
        return mask.any(axis=0) if mask.shape[1:] == grid else mask.any(axis=-1)
    if mask.shape != grid:
        raise ValueError(f"mask {mask.shape} vs grid {grid}")
    return mask


def load_binary_mask(path, grid_shape: Sequence[int]) -> np.ndarray:
    raw = np.load(path)
    try:
        arr = raw["mask"] if isinstance(raw, np.lib.npyio.NpzFile) else raw
        return as_spatial_mask(arr, grid_shape)
    finally:
        if hasattr(raw, "close"):
            raw.close()


def load_volume_arrays(path) -> Dict[str, np.ndarray]:
    """Load public-volume arrays (copied off the NPZ handle)."""
    data = np.load(path)
    try:
        return {
            "velocity": np.asarray(data["velocity"]),
            "mask": np.asarray(data["mask"]).astype(bool),
            "spacing": np.asarray(data["spacing"], dtype=np.float64).reshape(3),
            "origin": np.asarray(data["origin"], dtype=np.float64).reshape(3),
            "times": np.asarray(data["times"], dtype=np.float64).reshape(-1),
        }
    finally:
        data.close()


def _unit(normal: Sequence[float]) -> np.ndarray:
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    return n / float(np.linalg.norm(n).clip(min=1e-12))


def analysis_plane_mask(
    mask: np.ndarray,
    normal: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Keep the max-area slice along the through-plane axis.

    An Aao *tube* would over-count flux. 2D-PC analogue is one plane.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"expected (Nx,Ny,Nz) mask, got {mask.shape}")
    if not mask.any():
        raise ValueError("Aao lumen mask is empty")
    if normal is None:
        occupied = [
            int((mask.sum(axis=tuple(i for i in range(3) if i != ax)) > 0).sum())
            for ax in range(3)
        ]
        axis = int(np.argmin(occupied))
    else:
        axis = int(np.argmax(np.abs(_unit(normal))))
    other = tuple(i for i in range(3) if i != axis)
    per_slice = mask.sum(axis=other)
    k = int(np.argmax(per_slice))
    out = np.zeros_like(mask)
    sl: list = [slice(None)] * 3
    sl[axis] = k
    out[tuple(sl)] = mask[tuple(sl)]
    return out


def voxel_face_area_m2(spacing_m: Sequence[float], normal: Sequence[float]) -> float:
    spacing = np.asarray(spacing_m, dtype=np.float64).reshape(3)
    n = _unit(normal)
    thickness = float(np.abs(n) @ spacing)
    return float(np.prod(spacing)) / max(thickness, 1e-12)


def _frame_dt(times: Sequence[float]) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise ValueError("times is empty")
    if times.size == 1:
        return np.array([1.0])
    dt = np.diff(times)
    return np.concatenate([dt, dt[:1]])


def plane_stroke_volume_ml(
    velocity: np.ndarray,
    lumen_mask: np.ndarray,
    spacing_m: Sequence[float],
    times: Sequence[float],
    normal: Optional[Sequence[float]] = None,
) -> float:
    """Cycle-integrated flux through an Aao analysis plane, in mL/beat.

    ``SV = Σ_t Σ_{plane} (v · n) ΔA Δt``. ``lumen_mask`` is thinned to one
    slice so a short tube does not over-count.
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 5 or velocity.shape[-1] != 3:
        raise ValueError(f"velocity must be (T,Nx,Ny,Nz,3), got {velocity.shape}")
    mask = as_spatial_mask(lumen_mask, velocity.shape[1:4])
    plane = analysis_plane_mask(mask, normal)
    if normal is None:
        v_mean = velocity[:, plane, :].mean(axis=(0, 1))
        if float(np.linalg.norm(v_mean)) < 1e-8:
            occupied = [
                int((plane.sum(axis=tuple(i for i in range(3) if i != ax)) > 0).sum())
                for ax in range(3)
            ]
            normal = np.eye(3)[int(np.argmin(occupied))]
        else:
            normal = v_mean
    n = _unit(normal)
    dA = voxel_face_area_m2(spacing_m, n)
    v_dot_n = np.tensordot(velocity, n, axes=([-1], [0]))
    flux = v_dot_n[:, plane].sum(axis=1) * dA
    sv_m3 = float(np.sum(flux * _frame_dt(times)))
    return sv_m3 * 1e6


def masked_rms_speed_median(velocity: np.ndarray, mask: np.ndarray) -> float:
    """Spatial median of per-voxel time-RMS |u|.

    Not ``max_t |u|``: that peak is positively biased over 20 frames and
    is what made the retired 10 cm/s p99 cut unable to fail on noise.
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 5 or velocity.shape[-1] != 3:
        raise ValueError(f"velocity must be (T,Nx,Ny,Nz,3), got {velocity.shape}")
    mask3 = as_spatial_mask(mask, velocity.shape[1:4])
    if not mask3.any():
        raise ValueError("mask is empty")
    speed = np.linalg.norm(velocity, axis=-1)
    rms = np.sqrt(np.mean(np.square(speed[:, mask3]), axis=0))
    return float(np.median(rms))


def temporal_low_harmonic_frac(
    velocity: np.ndarray,
    mask: np.ndarray,
    *,
    max_harmonic: int = FOURIER_LOW_HARMONICS,
) -> float:
    """Fraction of AC power in harmonics 1..``max_harmonic``.

    Uses the spatial-mean velocity curve ``v(t)`` inside ``mask`` (three
    components, power summed). Cardiac RR modulation piles into 1–4.
    White noise is flat (~0.40 of rfft AC bins when T=20). DC-only
    (static residual) returns 0.
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 5 or velocity.shape[-1] != 3:
        raise ValueError(f"velocity must be (T,Nx,Ny,Nz,3), got {velocity.shape}")
    mask3 = as_spatial_mask(mask, velocity.shape[1:4])
    if not mask3.any():
        raise ValueError("mask is empty")
    v_t = velocity[:, mask3, :].mean(axis=1)
    v_t = v_t - v_t.mean(axis=0, keepdims=True)
    spec = np.fft.rfft(v_t, axis=0)
    power = np.square(np.abs(spec)).sum(axis=1)
    ac = float(power[1:].sum())
    if ac < 1e-16:
        return 0.0
    hi = min(int(max_harmonic), int(power.size) - 1)
    if hi < 1:
        return 0.0
    return float(power[1:hi + 1].sum()) / ac


def jet_remainder_mask(
    velocity: np.ndarray,
    cavity: np.ndarray,
    *,
    speed_thresh: float = JET_SPEED_THRESH,
    persist_frac: float = JET_PERSIST_FRAC,
) -> np.ndarray:
    """``cavity & ~persist`` at a fixed cut (not “ever exceeded”)."""
    persist = persistent_stasis_mask(
        velocity, cavity, speed_thresh=speed_thresh, persist_frac=persist_frac)
    cavity3 = as_spatial_mask(cavity, velocity.shape[1:4])
    return cavity3 & ~persist


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if obj is None:
        return None
    return obj


def _combine(verdicts: Sequence[str]) -> str:
    if any(v == "fail" for v in verdicts):
        return "fail"
    if any(v == "pending" for v in verdicts):
        return "pending"
    return "pass"


def evaluate_osu_gate(
    velocity: np.ndarray,
    eval_mask: np.ndarray,
    spacing_m: Sequence[float],
    times: Sequence[float],
    *,
    lv_mask: Optional[np.ndarray] = None,
    tissue_mask: Optional[np.ndarray] = None,
    aorta_mask: Optional[np.ndarray] = None,
    aao_plane_mask: Optional[np.ndarray] = None,
    aao_normal: Optional[Sequence[float]] = None,
    mv_plane_mask: Optional[np.ndarray] = None,
    mv_normal: Optional[Sequence[float]] = None,
    mask_is_lv: bool = False,
    persist_frac: float = JET_PERSIST_FRAC,
) -> Dict:
    """Structured PASS / FAIL / PENDING for dataset #11.

    ``eval_mask`` should be the whole-heart (magnitude) cavity so the jet
    location check can see aorta contamination. An LV-only packed volume
    cannot fail ``jet_chamber`` unless ``aorta_mask`` overlaps that volume.

    ``tissue_mask`` is a static-tissue ROI (chest-wall muscle). It is
    required to pass ``lv_vs_tissue``. ``mv_plane_mask`` is a thin mitral
    plane for ``valve_balance``.
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    grid = velocity.shape[1:4]
    eval3 = as_spatial_mask(eval_mask, grid)
    spacing = np.asarray(spacing_m, dtype=np.float64).reshape(3)
    voxel_ml = float(np.prod(spacing)) * 1e6

    lv3 = as_spatial_mask(lv_mask, grid) if lv_mask is not None else None
    if lv3 is None and mask_is_lv:
        lv3 = eval3
        mask_is_lv = True
    elif lv3 is not None:
        mask_is_lv = True
    else:
        mask_is_lv = False

    tissue3 = (as_spatial_mask(tissue_mask, grid)
               if tissue_mask is not None else None)
    if tissue3 is not None and lv3 is not None:
        tissue3 = tissue3 & ~lv3

    ao3 = as_spatial_mask(aorta_mask, grid) if aorta_mask is not None else None
    aao3 = (as_spatial_mask(aao_plane_mask, grid)
            if aao_plane_mask is not None else ao3)
    mv3 = (as_spatial_mask(mv_plane_mask, grid)
           if mv_plane_mask is not None else None)

    p99_mask = lv3 if mask_is_lv and lv3 is not None else eval3
    stasis = volume_stasis_report(
        velocity, p99_mask, spacing, speed_thresh=JET_SPEED_THRESH,
        persist_frac=persist_frac, absolute=True)
    legacy_p99 = float(stasis["speed_p99"])

    lv_rms = None
    tissue_rms = None
    ratio = None
    if mask_is_lv and lv3 is not None:
        lv_rms = masked_rms_speed_median(velocity, lv3)
    if tissue3 is not None and tissue3.any():
        tissue_rms = masked_rms_speed_median(velocity, tissue3)
    if lv_rms is not None and tissue_rms is not None:
        ratio = lv_rms / max(tissue_rms, 1e-12)

    if not mask_is_lv or lv3 is None:
        tissue_check = {
            "verdict": "pending",
            "lv_rms_mps": lv_rms,
            "tissue_rms_mps": tissue_rms,
            "ratio": ratio,
            "min_ratio": LV_TISSUE_RATIO_MIN,
            "legacy_speed_p99": legacy_p99,
            "on_lv": False,
            "detail": (
                f"legacy_speed_p99 {legacy_p99:.4f} m/s is on a magnitude "
                "placeholder; the retired 10 cm/s cut cannot fail on noise"
            ),
        }
    elif tissue3 is None or not tissue3.any():
        tissue_check = {
            "verdict": "pending",
            "lv_rms_mps": lv_rms,
            "tissue_rms_mps": None,
            "ratio": None,
            "min_ratio": LV_TISSUE_RATIO_MIN,
            "legacy_speed_p99": legacy_p99,
            "on_lv": True,
            "detail": (
                f"LV time-RMS median {lv_rms:.4f} m/s; need a static-tissue "
                "ROI (chest-wall muscle) as the noise floor"
            ),
        }
    elif ratio < LV_TISSUE_RATIO_MIN:
        tissue_check = {
            "verdict": "fail",
            "lv_rms_mps": lv_rms,
            "tissue_rms_mps": tissue_rms,
            "ratio": ratio,
            "min_ratio": LV_TISSUE_RATIO_MIN,
            "legacy_speed_p99": legacy_p99,
            "on_lv": True,
            "detail": (
                f"LV time-RMS {lv_rms:.4f} / tissue {tissue_rms:.4f} "
                f"= {ratio:.2f} < {LV_TISSUE_RATIO_MIN:.1f}× "
                f"(legacy p99 {legacy_p99:.4f} m/s, not a cut)"
            ),
        }
    else:
        tissue_check = {
            "verdict": "pass",
            "lv_rms_mps": lv_rms,
            "tissue_rms_mps": tissue_rms,
            "ratio": ratio,
            "min_ratio": LV_TISSUE_RATIO_MIN,
            "legacy_speed_p99": legacy_p99,
            "on_lv": True,
            "detail": (
                f"LV time-RMS {lv_rms:.4f} / tissue {tissue_rms:.4f} "
                f"= {ratio:.2f} ≥ {LV_TISSUE_RATIO_MIN:.1f}×"
            ),
        }

    if not mask_is_lv or lv3 is None:
        fourier_check = {
            "verdict": "pending",
            "low_frac": None,
            "min_frac": FOURIER_LOW_FRAC_MIN,
            "harmonics": FOURIER_LOW_HARMONICS,
            "detail": "need an LV mask for the temporal Fourier check",
        }
    else:
        low_frac = temporal_low_harmonic_frac(velocity, lv3)
        if low_frac < FOURIER_LOW_FRAC_MIN:
            fourier_check = {
                "verdict": "fail",
                "low_frac": low_frac,
                "min_frac": FOURIER_LOW_FRAC_MIN,
                "harmonics": FOURIER_LOW_HARMONICS,
                "detail": (
                    f"harmonics 1–{FOURIER_LOW_HARMONICS} hold {low_frac:.2f} "
                    f"of AC power (< {FOURIER_LOW_FRAC_MIN:.2f}; "
                    "flat/DC is noise, not a cardiac waveform)"
                ),
            }
        else:
            fourier_check = {
                "verdict": "pass",
                "low_frac": low_frac,
                "min_frac": FOURIER_LOW_FRAC_MIN,
                "harmonics": FOURIER_LOW_HARMONICS,
                "detail": (
                    f"harmonics 1–{FOURIER_LOW_HARMONICS} hold {low_frac:.2f} "
                    f"of AC power (≥ {FOURIER_LOW_FRAC_MIN:.2f})"
                ),
            }

    sv_ml = None
    if aao3 is not None:
        sv_ml = plane_stroke_volume_ml(
            velocity, aao3, spacing, times, normal=aao_normal)
        rel = abs(sv_ml - DATASET11_AAO_SV_ML) / DATASET11_AAO_SV_ML
        if rel > AAO_SV_REL_TOL:
            sv_check = {
                "verdict": "fail",
                "value_ml": sv_ml,
                "ref_ml": DATASET11_AAO_SV_ML,
                "rel_err": rel,
                "rel_tol": AAO_SV_REL_TOL,
                "detail": (
                    f"Aao SV {sv_ml:.1f} mL vs 93.6 "
                    f"(rel {rel:.2f} > {AAO_SV_REL_TOL:.2f})"
                ),
            }
        else:
            sv_check = {
                "verdict": "pass",
                "value_ml": sv_ml,
                "ref_ml": DATASET11_AAO_SV_ML,
                "rel_err": rel,
                "rel_tol": AAO_SV_REL_TOL,
                "detail": f"Aao SV {sv_ml:.1f} mL vs 93.6 (rel {rel:.2f})",
            }
    else:
        sv_check = {
            "verdict": "pending",
            "value_ml": None,
            "ref_ml": DATASET11_AAO_SV_ML,
            "rel_err": None,
            "rel_tol": AAO_SV_REL_TOL,
            "detail": "no Aao analysis plane — cannot check 93.6 mL/beat",
        }

    sv_mv = None
    if mv3 is not None:
        sv_mv = plane_stroke_volume_ml(
            velocity, mv3, spacing, times, normal=mv_normal)
    if sv_ml is None or sv_mv is None:
        valve_check = {
            "verdict": "pending",
            "aao_ml": sv_ml,
            "mv_ml": sv_mv,
            "rel_err": None,
            "rel_tol": VALVE_SV_REL_TOL,
            "detail": "need both a mitral plane and an Aao plane for SV balance",
        }
    else:
        denom = max(abs(sv_ml), abs(sv_mv), 1e-12)
        valve_rel = abs(abs(sv_mv) - abs(sv_ml)) / denom
        if valve_rel > VALVE_SV_REL_TOL:
            valve_check = {
                "verdict": "fail",
                "aao_ml": sv_ml,
                "mv_ml": sv_mv,
                "rel_err": valve_rel,
                "rel_tol": VALVE_SV_REL_TOL,
                "detail": (
                    f"|MV {sv_mv:.1f} − Aao {sv_ml:.1f}| / max "
                    f"= {valve_rel:.2f} > {VALVE_SV_REL_TOL:.2f}"
                ),
            }
        else:
            valve_check = {
                "verdict": "pass",
                "aao_ml": sv_ml,
                "mv_ml": sv_mv,
                "rel_err": valve_rel,
                "rel_tol": VALVE_SV_REL_TOL,
                "detail": (
                    f"MV {sv_mv:.1f} mL vs Aao {sv_ml:.1f} mL "
                    f"(rel {valve_rel:.2f})"
                ),
            }

    jet = jet_remainder_mask(
        velocity, eval3, speed_thresh=JET_SPEED_THRESH,
        persist_frac=persist_frac)
    jet_ml = float(jet.sum()) * voxel_ml
    jet_in_lv_ml = None
    jet_in_lv_frac = None
    jet_in_aorta_ml = None
    jet_in_aorta_frac = None
    if lv3 is not None:
        in_lv = jet & lv3
        jet_in_lv_ml = float(in_lv.sum()) * voxel_ml
        jet_in_lv_frac = float(in_lv.sum()) / max(int(jet.sum()), 1)
    if ao3 is not None:
        in_ao = jet & ao3
        jet_in_aorta_ml = float(in_ao.sum()) * voxel_ml
        jet_in_aorta_frac = float(in_ao.sum()) / max(int(jet.sum()), 1)

    lv_only_eval = bool(mask_is_lv and ao3 is None and lv3 is not None
                        and np.array_equal(eval3, lv3))
    if lv_only_eval:
        jet_check = {
            "verdict": "pending",
            "jet_ml": jet_ml,
            "jet_in_lv_ml": jet_in_lv_ml,
            "jet_in_lv_frac": jet_in_lv_frac,
            "jet_in_aorta_ml": None,
            "jet_in_aorta_frac": None,
            "detail": (
                "LV-only volume cannot test aorta/LVOT contamination; "
                "re-run on the whole-heart NPZ + LV + Aao masks"
            ),
        }
    elif lv3 is None:
        jet_check = {
            "verdict": "pending",
            "jet_ml": jet_ml,
            "jet_in_lv_ml": None,
            "jet_in_lv_frac": None,
            "jet_in_aorta_ml": jet_in_aorta_ml,
            "jet_in_aorta_frac": jet_in_aorta_frac,
            "detail": (
                f"whole-heart 10 cm/s remainder {jet_ml:.2f} mL; "
                "need an LV mask to locate it (an Aao plane is only for SV)"
            ),
        }
    elif jet.sum() == 0:
        jet_check = {
            "verdict": "fail",
            "jet_ml": 0.0,
            "jet_in_lv_ml": jet_in_lv_ml,
            "jet_in_lv_frac": jet_in_lv_frac,
            "jet_in_aorta_ml": jet_in_aorta_ml,
            "jet_in_aorta_frac": jet_in_aorta_frac,
            "detail": "no 10 cm/s remainder in the eval mask",
        }
    elif jet_in_aorta_frac is not None and jet_in_aorta_frac >= JET_IN_AORTA_MAX_FRAC:
        jet_check = {
            "verdict": "fail",
            "jet_ml": jet_ml,
            "jet_in_lv_ml": jet_in_lv_ml,
            "jet_in_lv_frac": jet_in_lv_frac,
            "jet_in_aorta_ml": jet_in_aorta_ml,
            "jet_in_aorta_frac": jet_in_aorta_frac,
            "detail": (
                f"{jet_in_aorta_frac:.0%} of the 10 cm/s remainder "
                f"({jet_in_aorta_ml:.2f}/{jet_ml:.2f} mL) is in the aorta/LVOT"
            ),
        }
    elif jet_in_lv_frac is not None and jet_in_lv_frac < JET_IN_LV_MIN_FRAC:
        jet_check = {
            "verdict": "fail",
            "jet_ml": jet_ml,
            "jet_in_lv_ml": jet_in_lv_ml,
            "jet_in_lv_frac": jet_in_lv_frac,
            "jet_in_aorta_ml": jet_in_aorta_ml,
            "jet_in_aorta_frac": jet_in_aorta_frac,
            "detail": (
                f"only {jet_in_lv_frac:.0%} of the 10 cm/s remainder "
                f"({jet_in_lv_ml:.2f}/{jet_ml:.2f} mL) is inside the LV"
            ),
        }
    else:
        jet_check = {
            "verdict": "pass",
            "jet_ml": jet_ml,
            "jet_in_lv_ml": jet_in_lv_ml,
            "jet_in_lv_frac": jet_in_lv_frac,
            "jet_in_aorta_ml": jet_in_aorta_ml,
            "jet_in_aorta_frac": jet_in_aorta_frac,
            "detail": (
                f"10 cm/s remainder {jet_ml:.2f} mL "
                + (f"({jet_in_lv_frac:.0%} in LV) " if jet_in_lv_frac is not None else "")
                + (f"({jet_in_aorta_frac:.0%} in aorta)" if jet_in_aorta_frac is not None else "")
            ).strip(),
        }

    report = {
        "dataset": "osu_zenodo_12515230_rest11",
        "verdict": _combine((
            tissue_check["verdict"], fourier_check["verdict"],
            sv_check["verdict"], valve_check["verdict"],
            jet_check["verdict"])),
        "mask_is_lv": bool(mask_is_lv),
        "cavity_ml_eval": float(eval3.sum()) * voxel_ml,
        "cavity_ml_lv": float(lv3.sum()) * voxel_ml if lv3 is not None else None,
        "cavity_ml_tissue": (
            float(tissue3.sum()) * voxel_ml
            if tissue3 is not None and tissue3.any() else None),
        "cavity_ml_p99": float(stasis["cavity_ml"]),
        "checks": {
            "lv_vs_tissue": tissue_check,
            "time_consistency": fourier_check,
            "aao_sv": sv_check,
            "valve_balance": valve_check,
            "jet_chamber": jet_check,
        },
        "thresholds": dict(THRESHOLDS),
        "background_phase": {
            "source": "core_outputs.theta_after_cmap",
            "applied_in_python": False,
            "note": (
                "main_recon writes theta* after cmapx/y/z. Packer maps "
                "θ/π·VENC. Residual static-tissue speed is the leftover."
            ),
        },
        "units": {
            "velocity": "m/s",
            "qa_nifti": "display 0-1000, not cm/s",
        },
        "notes": list(GATE_NOTES),
    }
    return _jsonable(report)


def format_osu_gate(report: Dict) -> str:
    lines = [f"OSU gate: {str(report['verdict']).upper()}"]
    for name, check in report["checks"].items():
        lines.append(f"  {name}: {check['verdict']} — {check['detail']}")
    return "\n".join(lines)


def write_gate_json(path, report: Dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return path


def parse_index_span(text: str, length: int) -> slice:
    """``'10:40'`` → inclusive SNAP cursor range as a Python slice."""
    parts = str(text).replace("-", ":").split(":")
    if len(parts) != 2:
        raise ValueError(f"expected lo:hi, got {text!r}")
    lo, hi = int(parts[0]), int(parts[1])
    if lo > hi:
        lo, hi = hi, lo
    if lo < 0 or hi >= length:
        raise ValueError(f"{text!r} outside 0..{length - 1}")
    return slice(lo, hi + 1)


def box_threshold_mask(
    image: np.ndarray,
    box_xyz,
    *,
    percentile: float = 55.0,
) -> np.ndarray:
    """Largest filled component of ``image >= p`` inside an index box.

    This is a **coarse cavity**, not an endocardial segmentation. 2.3 mm
    CORe magnitude cannot support a publication wall.
    """
    vol = np.asarray(image, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(f"expected (Nx,Ny,Nz), got {vol.shape}")
    sx, sy, sz = box_xyz
    roi = vol[sx, sy, sz]
    finite = roi[np.isfinite(roi) & (roi > 0)]
    if finite.size == 0:
        raise ValueError("box has no positive voxels")
    cut = float(np.percentile(finite, percentile))
    inside = np.zeros(vol.shape, dtype=bool)
    inside[sx, sy, sz] = np.isfinite(roi) & (roi >= cut)
    labeled, n = cc_label(inside)
    if n == 0:
        raise ValueError("no voxels above the percentile in the box")
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    out = labeled == int(counts.argmax())
    return binary_fill_holes(out)


def write_osu_qa(
    qa_dir,
    *,
    velocity: np.ndarray,
    spacing_m: Sequence[float],
    origin_m: Sequence[float],
    mag: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
) -> Path:
    """Mid-slice PNGs + optional NIfTI for drawing LV / Aao in ITK-SNAP."""
    qa_dir = Path(qa_dir)
    qa_dir.mkdir(parents=True, exist_ok=True)
    peak = peak_speed(velocity)
    grid = peak.shape
    np.save(qa_dir / "peak_speed.npy", peak)
    template = np.zeros(grid, dtype=np.uint8)
    np.save(qa_dir / "mask_template.npy", template)
    peak_draw = peak
    if mask is not None:
        mask3 = as_spatial_mask(mask, grid)
        np.save(qa_dir / "eval_mask.npy", mask3.astype(np.uint8))
        # Air/noise hits VENC·√3 and washes ITK-SNAP contrast. Draw on the mask.
        peak_draw = np.where(mask3, peak, 0.0)
    mag_mean = None
    mag_xyz_t = None
    if mag is not None:
        mag = np.asarray(mag)
        t_len = int(velocity.shape[0])
        if mag.ndim == 4:
            if mag.shape[0] == t_len:
                mag_xyz_t = np.moveaxis(np.abs(mag), 0, -1)
            elif mag.shape[-1] == t_len:
                mag_xyz_t = np.abs(mag)
            else:
                mag_xyz_t = np.abs(mag)
            mag_mean = mag_xyz_t.mean(axis=-1)
        else:
            mag_mean = np.abs(mag)
        np.save(qa_dir / "mag_mean.npy", mag_mean)

    _write_mid_slices(qa_dir / "slices_peak.png", peak_draw, title="peak |u| (m/s, masked)")
    if mag_mean is not None:
        _write_mid_slices(qa_dir / "slices_mag.png", mag_mean, title="mean |xHat|")

    spacing = np.asarray(spacing_m, dtype=np.float64).reshape(3)
    origin = np.asarray(origin_m, dtype=np.float64).reshape(3)
    # ITK-SNAP default window is ~0–255. Physical m/s and CORe |xHat|~0.03
    # look black. NIfTIs are display-scaled; .npy stay physical.
    peak_nifti = _display_scale(peak_draw, hi=1000.0)
    nifti_note = _try_write_nifti(qa_dir / "peak_speed.nii.gz", peak_nifti, spacing, origin)
    if mag_mean is not None:
        mag_nifti = _display_scale(mag_mean, hi=1000.0)
        _try_write_nifti(qa_dir / "mag_mean.nii.gz", mag_nifti, spacing, origin)
        if mag_xyz_t is not None:
            mip = mag_xyz_t.max(axis=-1)
            _try_write_nifti(
                qa_dir / "mag_mip.nii.gz", _display_scale(mip, hi=1000.0),
                spacing, origin)
            _try_write_nifti(
                qa_dir / "mag_4d.nii.gz", _display_scale(mag_xyz_t, hi=1000.0),
                spacing, origin)
    _try_write_nifti(qa_dir / "mask_template.nii.gz", template, spacing, origin)

    (qa_dir / "ORIENTATION.txt").write_text(
        "Affine is spacing-only (identity orientation, mm). R/L/A/P/S/I\n"
        "labels in ITK-SNAP are NOT scanner anatomy. Confirm the descending\n"
        "aorta runs superior-inferior on the posterior-left of the heart\n"
        "before trusting those labels.\n"
    )
    (qa_dir / "README.txt").write_text(
        "OSU dataset #11 QA — draw masks AFTER the gate-1 revision.\n"
        "Magnitude is not LV. Do not train. Do not start a thrombus solver.\n\n"
        "UNITS: NIfTI peak_speed / mag_mean / mag_mip are display-scaled\n"
        "0-1000 (1-99% stretch). They are NOT cm/s. A SNAP cursor of 576\n"
        "is not 576 cm/s. Physical m/s stay in the sibling .npy files.\n\n"
        "ITK-SNAP: open mag_mean.nii.gz only. HIDE peak_speed.nii.gz while\n"
        "painting. Do not trace PC-MRA / speed-weighted edges as a wall —\n"
        "that selects high-speed voxels and inflates LV stats. Use speed\n"
        "or PC-MRA only to find the chamber, then draw on magnitude.\n"
        "Walls are soft at 2.3 mm — do not freehand a publication contour.\n\n"
        "Also draw:\n"
        "  - static-tissue ROI (chest-wall muscle, not lung / not the box)\n"
        "  - thin Aao analysis plane (one slice; stroke volume only)\n"
        "  - thin mitral analysis plane (inflow; valve-balance check)\n"
        "A one-slice box is a valve plane. Chest-wall box = tissue ROI.\n"
        "Note cursor (x,y,z) corners and run osu_box_mask.py, or paint into\n"
        "mask_template.npy (uint8, same grid) and save as .npy.\n\n"
        "SAVE VERSIONED: rest_lv_mask_v1.nii.gz, then v2 if needed.\n"
        "Do not overwrite. Do not redraw after reading the gate JSON.\n"
        "Confirm millilitres in SNAP Segmentation → Volumes and Statistics.\n"
        "Adult LVEDV is typically ≥ 100 mL; the discarded auto-box\n"
        "43:60 59:69 15:37 is ~53 mL and sits anterior (likely RV / wall),\n"
        "so it is not LV. Do not score it.\n\n"
        "If no confident LV contour in 30-45 min, stop and send\n"
        "docs/data/CHARITE_INQUIRY.md. Do not keep painting.\n\n"
        f"grid={tuple(grid)}  spacing_m={spacing.tolist()}\n"
        f"nifti={nifti_note}\n"
        "Then:\n"
        "  PYTHONPATH=src python3 scripts/osu_failcheap_gate.py \\\n"
        "      --volume data/osu_4dflow/rest_wholeheart.npz \\\n"
        "      --lv-mask path/to/rest_lv_mask_v1.nii.gz \\\n"
        "      --tissue-mask path/to/rest_tissue_mask_v1.nii.gz \\\n"
        "      --aao-mask path/to/rest_aao_mask_v1.nii.gz \\\n"
        "      --mv-mask path/to/rest_mv_mask_v1.nii.gz \\\n"
        "      --out docs/results/public_volume/osu_gate.json\n"
        "Optional: --aorta-region for a filled Aao/LVOT (not the thin plane).\n"
    )
    return qa_dir


def _write_mid_slices(path: Path, vol: np.ndarray, *, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vol = np.asarray(vol, dtype=np.float64)
    mids = [s // 2 for s in vol.shape]
    planes = (
        vol[mids[0], :, :],
        vol[:, mids[1], :],
        vol[:, :, mids[2]],
    )
    labels = ("x mid", "y mid", "z mid")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    vmax = float(np.nanpercentile(vol[np.isfinite(vol)], 99)) if np.isfinite(vol).any() else 1.0
    for ax, sl, lab in zip(axes, planes, labels):
        ax.imshow(sl.T, origin="lower", cmap="magma", vmin=0.0, vmax=max(vmax, 1e-6))
        ax.set_title(lab)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _display_scale(arr: np.ndarray, *, hi: float = 1000.0) -> np.ndarray:
    """Stretch finite voxels to ``[0, hi]`` so ITK-SNAP is not black."""
    a = np.asarray(arr, dtype=np.float64)
    sel = a[np.isfinite(a)]
    sel = sel[sel > 0]
    if sel.size == 0:
        return np.zeros_like(a, dtype=np.float32)
    lo, hi_p = np.percentile(sel, (1.0, 99.0))
    if hi_p <= lo:
        hi_p = lo + 1e-6
    out = (a - lo) / (hi_p - lo) * float(hi)
    return np.clip(out, 0.0, float(hi)).astype(np.float32)


def _try_write_nifti(path: Path, array: np.ndarray, spacing_m, origin_m) -> str:
    try:
        import nibabel as nib  # type: ignore
    except ImportError:
        return "skipped (pip install nibabel for ITK-SNAP)"
    spacing_mm = np.asarray(spacing_m, dtype=np.float64).reshape(3) * 1000.0
    origin_mm = np.asarray(origin_m, dtype=np.float64).reshape(3) * 1000.0
    aff = np.eye(4)
    aff[0, 0], aff[1, 1], aff[2, 2] = spacing_mm
    aff[:3, 3] = origin_mm
    img = nib.Nifti1Image(np.asarray(array, dtype=np.float32), aff)
    img.header.set_xyzt_units(xyz="mm")
    nib.save(img, str(path))
    return str(path.name)


def load_mask_any(path, grid_shape: Sequence[int]) -> np.ndarray:
    """Load a boolean mask from ``.npy``, ``.npz``, or NIfTI."""
    path = Path(path)
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".nii") or suffix.endswith(".nii.gz"):
        try:
            import nibabel as nib  # type: ignore
        except ImportError as exc:
            raise ImportError("NIfTI masks need nibabel: pip install nibabel") from exc
        arr = np.asanyarray(nib.load(str(path)).dataobj)
        return as_spatial_mask(arr, grid_shape)
    return load_binary_mask(path, grid_shape)
