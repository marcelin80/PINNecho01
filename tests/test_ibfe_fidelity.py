"""Decision 9 /meta/fidelity: pre-D9 diastole is unusable; approx exposes V0."""

import dataclasses
import json

import numpy as np
import torch

from pinnecho.config import Config
from pinnecho.data.ibfe_fidelity import (
    ATTEMPT_2_SCOPE,
    DIASTOLE_UNUSABLE_WARNING,
    IMAGE_VOLUME_ML_PAT12,
    KLOTZ_V0_ML,
    P_IMG_MMHG_ASSUMED,
    Q_CLOSE_RATIONALE,
    V0_SCALE_ISOTROPIC,
    MONK_ALPHA,
    MONK_V0_ML,
    MONK_V_IMG_ML,
    decision_9_approx_fidelity,
    default_pre_decision_9,
    fidelity_from_scale_reference,
    diastole_usable,
    isotropic_linear_scale,
    loads_fidelity,
    manufactured_fidelity,
    resolve_fidelity,
)
from pinnecho.data.ibfe_io import (
    NPZ_TENSOR_KEYS,
    load_ibfe_manifest,
    load_ibfe_npz,
    save_ibfe_manifest_csv,
    save_ibfe_npz,
)
from pinnecho.data.ibfe_validate import validate_ibfe_frames
from pinnecho.data.load_ibfe_output import synthetic_ibfe_frames
from pinnecho.data.public_volume import frames_from_public_volume


def test_isotropic_scale_is_exposed_assumption():
    scale = isotropic_linear_scale(KLOTZ_V0_ML, IMAGE_VOLUME_ML_PAT12)
    assert abs(scale - (58.0 / 105.0) ** (1.0 / 3.0)) < 1e-12
    assert abs(scale - 0.82) < 0.01
    assert abs(V0_SCALE_ISOTROPIC - scale) < 1e-12


def test_q_close_rationale_is_honest():
    assert "p_lv = p_la" in Q_CLOSE_RATIONALE
    assert "q_close" in Q_CLOSE_RATIONALE
    assert "R_la=0.03" in ATTEMPT_2_SCOPE
    assert "q_close=10" in ATTEMPT_2_SCOPE
    assert "end=0.33" in ATTEMPT_2_SCOPE
    note = default_pre_decision_9().notes
    assert Q_CLOSE_RATIONALE in note
    assert "systole" in note and "ejection" in note


def test_missing_fidelity_is_pre_decision_9_systole_only():
    fid = loads_fidelity(None)
    assert fid.diastole_usable is False
    assert fid.decision_9 == "not_applied"
    assert fid.reference == "image_as_unstressed"
    assert fid.usable_phases == ("systole", "ejection")
    assert fid.p_img_mmhg == P_IMG_MMHG_ASSUMED


def test_monk_scale_reference_is_v0_mesh_not_a_filled_beat():
    raw = {
        "decision_9": "a",
        "decision_9_stage": "a1-isotropic-scale",
        "approximation": True,
        "algorithm": "isotropic_scale",
        "V_img_mL": MONK_V_IMG_ML,
        "V0_target_mL": 58.0,
        "V0_actual_mL": MONK_V0_ML,
        "alpha": MONK_ALPHA,
        "p_img_mmHg_assumed": 8.0,
        "p_img_range_mmHg": [5.0, 10.0],
        "image": "/home/marce/pat12_meshes/LV.e",
        "out": "/home/marce/pat12_meshes/LV_V0.e",
        "pass_v0": True,
        "n_inverted": 0,
        "note": "Approximation: homothety about the cavity centroid.",
    }
    fid = fidelity_from_scale_reference(raw)
    assert fid.decision_9 == "approx"
    assert fid.approximation is True
    assert fid.diastole_usable is False
    assert fid.usable_phases == ("reference",)
    assert abs(fid.v0_scale - MONK_ALPHA) < 1e-12
    assert abs(fid.v0_scale - isotropic_linear_scale(MONK_V0_ML, MONK_V_IMG_ML)) < 1e-12
    # Formula 58/105 is close but not the applied mesh scale.
    assert abs(fid.v0_scale - V0_SCALE_ISOTROPIC) > 1e-4
    assert fid.extra["p_img_mmHg_assumed"] == 8.0


def test_approx_fidelity_exposes_v0_and_p_img():
    fid = decision_9_approx_fidelity()
    assert fid.decision_9 == "approx"
    assert fid.approximation is True
    assert fid.diastole_usable is True
    assert abs(fid.v0_scale - V0_SCALE_ISOTROPIC) < 1e-12
    assert fid.v0_ml == KLOTZ_V0_ML
    assert fid.p_img_mmhg == P_IMG_MMHG_ASSUMED
    assert "approximation" in fid.notes.lower()


def test_synthetic_frames_are_not_pre_d9(tmp_path):
    torch.set_default_dtype(torch.float64)
    frames = synthetic_ibfe_frames(Config(), n_fluid=80, n_wall=40, n_frames=3)
    fid = resolve_fidelity(frames)
    assert fid.source == "manufactured"
    assert fid.decision_9 == "not_applicable"
    assert diastole_usable(frames) is True
    report = validate_ibfe_frames(frames)
    assert report.ok
    assert not any("diastolic filling is unphysiological" in w for w in report.warnings)

    path = save_ibfe_npz(frames, tmp_path / "syn.npz")
    loaded = load_ibfe_npz(path)
    assert resolve_fidelity(loaded).source == "manufactured"
    assert diastole_usable(loaded) is True


def test_legacy_npz_without_fidelity_warns_diastole(tmp_path):
    torch.set_default_dtype(torch.float64)
    frames = synthetic_ibfe_frames(Config(), n_fluid=80, n_wall=40, n_frames=3)
    path = tmp_path / "legacy.npz"
    arrays = {k: getattr(frames, k).detach().cpu().numpy() for k in NPZ_TENSOR_KEYS}
    arrays["_meta"] = np.array(
        [frames.cycle_period, frames.rho, frames.mu, float(frames.spatial_dim)],
        dtype=np.float64,
    )
    np.savez(path, **arrays)

    loaded = load_ibfe_npz(path)
    assert loaded.fidelity.diastole_usable is False
    assert diastole_usable(loaded) is False
    report = validate_ibfe_frames(loaded)
    assert report.ok
    assert any(DIASTOLE_UNUSABLE_WARNING[:40] in w for w in report.warnings)
    assert report.stats["fidelity_diastole_usable"] == 0.0


def test_manifest_roundtrip_preserves_approx(tmp_path):
    torch.set_default_dtype(torch.float64)
    frames = synthetic_ibfe_frames(Config(), n_fluid=80, n_wall=40, n_frames=3)
    tagged = dataclasses.replace(frames, fidelity=decision_9_approx_fidelity())
    manifest = save_ibfe_manifest_csv(tagged, tmp_path / "m")
    loaded = load_ibfe_manifest(manifest)
    fid = resolve_fidelity(loaded)
    assert fid.decision_9 == "approx"
    assert abs(fid.v0_scale - V0_SCALE_ISOTROPIC) < 1e-12
    text = manifest.read_text()
    assert "decision_9" in text
    assert "v0_scale" in text


def test_public_volume_fidelity_skips_decision_9():
    vel = np.zeros((2, 6, 6, 6, 3), dtype=np.float32)
    vel[..., 0] = 0.2
    mask = np.zeros((2, 6, 6, 6), dtype=np.uint8)
    mask[:, 1:5, 1:5, 1:5] = 1
    frames = frames_from_public_volume(
        vel, mask, spacing=(0.002, 0.002, 0.002),
        origin=(0.0, 0.0, 0.0), times=(0.0, 0.1),
    )
    fid = resolve_fidelity(frames)
    assert fid.source == "public_volume"
    assert fid.decision_9 == "not_applicable"
    assert diastole_usable(frames) is True
    report = validate_ibfe_frames(frames)
    assert not any("diastolic filling is unphysiological" in w for w in report.warnings)


def test_fidelity_json_roundtrip():
    raw = json.dumps(decision_9_approx_fidelity(v0_ml=50.0, image_volume_ml=100.0).to_dict())
    fid = loads_fidelity(raw)
    assert abs(fid.v0_scale - (0.5) ** (1.0 / 3.0)) < 1e-12
    assert fid.approximation is True
