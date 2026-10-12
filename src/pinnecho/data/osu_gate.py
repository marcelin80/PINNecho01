"""Pre-registered fail-cheap gate for OSU rest dataset #11.

Corrado A/B/C/blur are closed. After MATLAB CORe, this is the next measured
check — not a train, not a thrombus, not an LV claim from a magnitude cut.

Three checks, written down before the laptop recon exists:

1. ``speed_p99`` on an **LV** mask ≥ 10 cm/s. The same number on a magnitude
   placeholder is PENDING: dataset #11 Aao peak is ~180 cm/s, so whole-heart
   p99 will clear the cut even if the LV jet is unusable.
2. Ascending-aorta stroke volume vs the paper 2D-PC ref **93.6 mL/beat**
   (Arshad et al., *Magn Reson Med* 2024, Fig. 4 / dataset #11). Relative
   error > 0.40 is FAIL. No Aao analysis plane is PENDING.
3. The 10 cm/s Eulerian remainder (``cavity & ~persist``) must live in the
   LV chamber, not the aorta / LVOT. Computed on the whole-heart eval mask.
   A thin Aao plane is **only** for check 2; pass ``aorta_mask`` if you
   have a filled LVOT/arch region.

PASS is all three. PASS does not authorize a thrombus solver or the 5.1 GB
cine. FAIL → Charité 4D-flow inquiry, not CFD.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from pinnecho.eval.stasis import (
    peak_speed, persistent_stasis_mask, volume_stasis_report,
)


DATASET11_AAO_SV_ML = 93.6
DATASET11_AAO_PEAK_MPS = 1.80
SPEED_P99_MIN_MPS = 0.10
AAO_SV_REL_TOL = 0.40
JET_IN_LV_MIN_FRAC = 0.50
JET_IN_AORTA_MAX_FRAC = 0.50
JET_SPEED_THRESH = 0.10
JET_PERSIST_FRAC = 0.50

THRESHOLDS = {
    "speed_p99_min_mps": SPEED_P99_MIN_MPS,
    "aao_sv_ref_ml": DATASET11_AAO_SV_ML,
    "aao_sv_rel_tol": AAO_SV_REL_TOL,
    "jet_in_lv_min_frac": JET_IN_LV_MIN_FRAC,
    "jet_in_aorta_max_frac": JET_IN_AORTA_MAX_FRAC,
    "jet_speed_thresh_mps": JET_SPEED_THRESH,
    "jet_persist_frac": JET_PERSIST_FRAC,
}

GATE_NOTES = (
    "Magnitude mask is not an LV segmentation.",
    "PASS does not authorize a thrombus solver.",
    "Do not download the 5.1 GB cine until PASS.",
    "FAIL → Charité 4D-flow inquiry, not CFD.",
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
    aorta_mask: Optional[np.ndarray] = None,
    aao_plane_mask: Optional[np.ndarray] = None,
    aao_normal: Optional[Sequence[float]] = None,
    mask_is_lv: bool = False,
    persist_frac: float = JET_PERSIST_FRAC,
) -> Dict:
    """Structured PASS / FAIL / PENDING for dataset #11.

    ``eval_mask`` should be the whole-heart (magnitude) cavity so the jet
    location check can see aorta contamination. An LV-only packed volume
    cannot fail check 3 unless ``aorta_mask`` overlaps that volume.
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

    ao3 = as_spatial_mask(aorta_mask, grid) if aorta_mask is not None else None
    aao3 = (as_spatial_mask(aao_plane_mask, grid)
            if aao_plane_mask is not None else ao3)

    p99_mask = lv3 if mask_is_lv and lv3 is not None else eval3
    stasis = volume_stasis_report(
        velocity, p99_mask, spacing, speed_thresh=JET_SPEED_THRESH,
        persist_frac=persist_frac, absolute=True)
    p99 = float(stasis["speed_p99"])
    if p99 < SPEED_P99_MIN_MPS:
        p99_check = {
            "verdict": "fail",
            "value": p99,
            "min": SPEED_P99_MIN_MPS,
            "on_lv": bool(mask_is_lv),
            "detail": f"speed_p99 {p99:.4f} m/s < 0.10 m/s",
        }
    elif not mask_is_lv:
        p99_check = {
            "verdict": "pending",
            "value": p99,
            "min": SPEED_P99_MIN_MPS,
            "on_lv": False,
            "detail": (
                f"speed_p99 {p99:.4f} m/s is on a magnitude placeholder; "
                "dataset #11 Aao ~180 cm/s will clear 10 cm/s without an LV jet"
            ),
        }
    else:
        p99_check = {
            "verdict": "pass",
            "value": p99,
            "min": SPEED_P99_MIN_MPS,
            "on_lv": True,
            "detail": f"speed_p99 {p99:.4f} m/s on LV mask",
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
            p99_check["verdict"], sv_check["verdict"], jet_check["verdict"])),
        "mask_is_lv": bool(mask_is_lv),
        "cavity_ml_eval": float(eval3.sum()) * voxel_ml,
        "cavity_ml_p99": float(stasis["cavity_ml"]),
        "checks": {
            "speed_p99": p99_check,
            "aao_sv": sv_check,
            "jet_chamber": jet_check,
        },
        "thresholds": dict(THRESHOLDS),
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
    if mag is not None:
        mag = np.asarray(mag)
        if mag.ndim == 4:
            # (T,X,Y,Z) or (X,Y,Z,T) — mean over the size-matching time axis
            if mag.shape[0] == velocity.shape[0]:
                mag_mean = np.abs(mag).mean(axis=0)
            elif mag.shape[-1] == velocity.shape[0]:
                mag_mean = np.abs(mag).mean(axis=-1)
            else:
                mag_mean = np.abs(mag).mean(axis=0)
        else:
            mag_mean = np.abs(mag)
        np.save(qa_dir / "mag_mean.npy", mag_mean)

    _write_mid_slices(qa_dir / "slices_peak.png", peak_draw, title="peak |u| (m/s, masked)")
    if mag_mean is not None:
        _write_mid_slices(qa_dir / "slices_mag.png", mag_mean, title="mean |xHat|")

    spacing = np.asarray(spacing_m, dtype=np.float64).reshape(3)
    origin = np.asarray(origin_m, dtype=np.float64).reshape(3)
    # ITK-SNAP expects the NIfTI affine in millimetres, not metres.
    nifti_note = _try_write_nifti(qa_dir / "peak_speed.nii.gz", peak_draw, spacing, origin)
    if mag_mean is not None:
        _try_write_nifti(qa_dir / "mag_mean.nii.gz", mag_mean, spacing, origin)
    _try_write_nifti(qa_dir / "mask_template.nii.gz", template, spacing, origin)

    (qa_dir / "README.txt").write_text(
        "OSU dataset #11 QA — draw an LV cavity and an Aao analysis plane.\n"
        "Magnitude is not LV. Do not train. Do not start a thrombus solver.\n\n"
        "ITK-SNAP: open mag_mean.nii.gz first (anatomy). Add peak_speed.nii.gz\n"
        "as a separate image. Affine is millimetres. peak_speed is magnitude-\n"
        "masked so air/VENC noise does not wash the contrast.\n"
        "save LV as rest_lv_mask.nii.gz and a thin Aao plane as "
        "rest_aao_mask.nii.gz.\n"
        "Or paint into mask_template.npy (uint8, same grid) and save as .npy.\n\n"
        f"grid={tuple(grid)}  spacing_m={spacing.tolist()}\n"
        f"nifti={nifti_note}\n"
        "Then:\n"
        "  PYTHONPATH=src python3 scripts/osu_failcheap_gate.py \\\n"
        "      --volume data/osu_4dflow/rest_wholeheart.npz \\\n"
        "      --lv-mask path/to/rest_lv_mask.nii.gz \\\n"
        "      --aao-mask path/to/rest_aao_mask.nii.gz \\\n"
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
