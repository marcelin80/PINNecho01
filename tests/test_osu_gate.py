"""OSU fail-cheap gate (no MATLAB, no Zenodo download)."""

import json
import math

import numpy as np
import pytest

from pinnecho.data.osu_core import pack_osu_public_volume
from pinnecho.data.osu_gate import (
    AAO_SV_REL_TOL, DATASET11_AAO_SV_ML, FOURIER_LOW_FRAC_MIN,
    LV_TISSUE_RATIO_MIN, VALVE_SV_REL_TOL, analysis_plane_mask,
    box_threshold_mask, evaluate_osu_gate, format_osu_gate,
    jet_remainder_mask, masked_rms_speed_median, parse_index_span,
    plane_stroke_volume_ml, temporal_low_harmonic_frac, write_gate_json,
    write_osu_qa,
)


def _times(t_len, period=1.0):
    return np.linspace(0.0, period, t_len, endpoint=False)


def test_plane_stroke_volume_uniform_slab():
    t_len, nx = 10, 12
    vel = np.zeros((t_len, nx, nx, nx, 3))
    vel[..., 2] = 0.50  # +z
    mask = np.zeros((nx, nx, nx), dtype=bool)
    mask[1:11, 1:11, 5] = True  # 10×10 plane
    sv = plane_stroke_volume_ml(
        vel, mask, spacing_m=(0.002, 0.002, 0.002), times=_times(t_len))
    # area = 10*10*0.002*0.002 = 4e-4 m²; flux = 0.5 * 4e-4 = 2e-4 m³/s
    # period 1 s → 200 mL
    assert sv == pytest.approx(200.0, rel=1e-6)


def test_analysis_plane_thins_a_tube():
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[3:6, 3:6, 2:6] = True
    plane = analysis_plane_mask(mask, normal=(0, 0, 1))
    assert plane[:, :, 2:6].sum() == plane.sum()
    assert int((plane.sum(axis=(0, 1)) > 0).sum()) == 1


def test_pack_explicit_mask_can_claim_lv():
    n, t_len = 6, 4
    th = np.zeros((n, n, n, t_len))
    mask = np.zeros((n, n, n), dtype=bool)
    mask[1:4, 1:4, 1:4] = True
    vol = pack_osu_public_volume(
        {"thetaX": th, "thetaY": th, "thetaZ": th, "xHat": np.ones_like(th)},
        venc_mps=2.0, spacing_m=(0.002, 0.002, 0.002),
        mask=mask, mask_is_lv=True)
    assert vol["mask_is_lv"] is True
    vol_ph = pack_osu_public_volume(
        {"thetaX": th, "thetaY": th, "thetaZ": th, "xHat": np.ones_like(th)},
        venc_mps=2.0, spacing_m=(0.002, 0.002, 0.002))
    assert vol_ph["mask_is_lv"] is False


def _two_chamber(
    t_len=8, n=16, lv_speed=0.40, ao_speed=0.80, tissue_speed=0.04,
    pulsatile=False, mv_speed=None,
):
    """Whole-heart eval: LV (left) + aorta (right) + tissue corner + mitral plane."""
    vel = np.zeros((t_len, n, n, n, 3))
    lv = np.zeros((n, n, n), dtype=bool)
    ao = np.zeros((n, n, n), dtype=bool)
    tissue = np.zeros((n, n, n), dtype=bool)
    lv[2:8, 4:12, 4:12] = True
    ao[10:14, 6:10, 6:10] = True
    tissue[0:2, 0:4, 0:4] = True
    eval_mask = lv | ao
    t = np.arange(t_len, dtype=np.float64)
    wave = (1.0 + 0.8 * np.sin(2.0 * np.pi * t / t_len)) if pulsatile else np.ones(t_len)
    vel[:, lv, 0] = lv_speed * wave[:, None]
    vel[:, ao, 2] = ao_speed
    vel[:, tissue, :] = tissue_speed
    aao_plane = np.zeros_like(ao)
    aao_plane[10:14, 6:10, 8] = True
    mv_plane = np.zeros((n, n, n), dtype=bool)
    # Same 4×4 area as the Aao plane, outside the LV so we can set v_z independently.
    mv_plane[2:6, 4:8, 1] = True
    vel[:, mv_plane, :] = 0.0
    vel[:, mv_plane, 2] = ao_speed if mv_speed is None else mv_speed
    spacing = (0.002, 0.002, 0.002)
    times = _times(t_len)
    return vel, eval_mask, lv, ao, aao_plane, tissue, mv_plane, spacing, times


def _score(vel, eval_mask, lv, ao, aao_plane, tissue, mv_plane, spacing, times,
           **kwargs):
    return evaluate_osu_gate(
        vel, eval_mask, spacing, times,
        lv_mask=lv, tissue_mask=tissue, aorta_mask=ao,
        aao_plane_mask=aao_plane, mv_plane_mask=mv_plane, mask_is_lv=True,
        **kwargs)


def test_gate_placeholder_is_pending():
    vel, eval_mask, *_rest, spacing, times = _two_chamber()
    report = evaluate_osu_gate(vel, eval_mask, spacing, times)
    assert report["verdict"] == "pending"
    assert report["checks"]["lv_vs_tissue"]["verdict"] == "pending"
    assert report["checks"]["time_consistency"]["verdict"] == "pending"
    assert report["checks"]["aao_sv"]["verdict"] == "pending"
    assert report["checks"]["valve_balance"]["verdict"] == "pending"
    assert report["checks"]["jet_chamber"]["verdict"] == "pending"
    assert report["mask_is_lv"] is False
    assert "speed_p99" not in report["checks"]


def test_gate_pass_when_jet_in_lv_and_sv_in_band():
    # Aao plane is 4×4 voxels × 0.002² = 6.4e-5 m².
    # 93.6 mL/beat = 9.36e-5 m³/s → v = 1.4625 m/s.
    packed = _two_chamber(ao_speed=1.4625, pulsatile=True)
    report = _score(*packed)
    assert report["checks"]["lv_vs_tissue"]["verdict"] == "pass"
    assert report["checks"]["lv_vs_tissue"]["ratio"] >= LV_TISSUE_RATIO_MIN
    assert report["checks"]["time_consistency"]["verdict"] == "pass"
    assert report["checks"]["aao_sv"]["verdict"] == "pass"
    assert report["checks"]["valve_balance"]["verdict"] == "pass"
    assert report["checks"]["jet_chamber"]["verdict"] == "pass"
    assert report["verdict"] == "pass"
    assert report["checks"]["jet_chamber"]["jet_in_lv_frac"] > 0.5
    assert report["background_phase"]["applied_in_python"] is False


def test_gate_fail_when_remainder_is_aorta():
    packed = _two_chamber(lv_speed=0.0, ao_speed=1.4625, pulsatile=True)
    report = _score(*packed)
    assert report["checks"]["jet_chamber"]["verdict"] == "fail"
    assert report["verdict"] == "fail"
    assert report["checks"]["jet_chamber"]["jet_in_aorta_frac"] >= 0.50


def test_gate_fail_lv_not_above_tissue_floor():
    packed = _two_chamber(lv_speed=0.02, ao_speed=1.4625, pulsatile=True)
    report = _score(*packed)
    assert report["checks"]["lv_vs_tissue"]["verdict"] == "fail"
    assert report["verdict"] == "fail"
    assert report["checks"]["lv_vs_tissue"]["ratio"] < LV_TISSUE_RATIO_MIN


def test_gate_lv_without_tissue_is_pending():
    vel, eval_mask, lv, ao, aao_plane, _tissue, mv_plane, spacing, times = (
        _two_chamber(ao_speed=1.4625, pulsatile=True))
    report = evaluate_osu_gate(
        vel, eval_mask, spacing, times,
        lv_mask=lv, aorta_mask=ao, aao_plane_mask=aao_plane,
        mv_plane_mask=mv_plane, mask_is_lv=True)
    assert report["checks"]["lv_vs_tissue"]["verdict"] == "pending"
    assert report["verdict"] == "pending"
    assert "static-tissue" in report["checks"]["lv_vs_tissue"]["detail"]


def test_legacy_p99_cannot_fail_on_flat_spectrum():
    """Equal harmonics 1–9 clear retired 10 cm/s p99 but fail new checks."""
    t_len, n = 20, 16
    vel, eval_mask, lv, ao, aao_plane, tissue, mv_plane, spacing, times = (
        _two_chamber(t_len=t_len, n=n, ao_speed=1.4625))
    t = np.arange(t_len, dtype=np.float64)
    sig = np.zeros(t_len)
    for k in range(1, 10):
        sig += np.sin(2.0 * np.pi * k * t / t_len)
    sig = 0.30 * sig / max(float(sig.std()), 1e-12)
    vel[:, lv, :] = 0.0
    vel[:, lv, 0] = sig[:, None]
    vel[:, tissue, :] = 0.04
    report = evaluate_osu_gate(
        vel, eval_mask, spacing, times,
        lv_mask=lv, tissue_mask=tissue, aorta_mask=ao,
        aao_plane_mask=aao_plane, mv_plane_mask=mv_plane, mask_is_lv=True)
    assert report["checks"]["lv_vs_tissue"]["legacy_speed_p99"] >= 0.10
    assert report["checks"]["time_consistency"]["low_frac"] < FOURIER_LOW_FRAC_MIN
    assert report["checks"]["time_consistency"]["verdict"] == "fail"
    assert report["verdict"] == "fail"


def test_gate_fail_sv_out_of_band():
    packed = _two_chamber(
        ao_speed=3.00, pulsatile=True)  # ~192 mL, well outside ±40 % of 93.6
    report = _score(*packed)
    rel = abs(report["checks"]["aao_sv"]["value_ml"] - DATASET11_AAO_SV_ML) / DATASET11_AAO_SV_ML
    assert rel > AAO_SV_REL_TOL
    assert report["checks"]["aao_sv"]["verdict"] == "fail"
    assert report["verdict"] == "fail"


def test_gate_fail_valve_imbalance():
    packed = _two_chamber(ao_speed=1.4625, pulsatile=True, mv_speed=0.20)
    report = _score(*packed)
    assert report["checks"]["valve_balance"]["verdict"] == "fail"
    assert report["checks"]["valve_balance"]["rel_err"] > VALVE_SV_REL_TOL
    assert report["verdict"] == "fail"


def test_aao_plane_alone_does_not_locate_jet():
    vel, eval_mask, _lv, _ao, aao_plane, _tissue, _mv, spacing, times = _two_chamber()
    report = evaluate_osu_gate(
        vel, eval_mask, spacing, times, aao_plane_mask=aao_plane)
    assert report["checks"]["jet_chamber"]["verdict"] == "pending"
    assert report["checks"]["aao_sv"]["verdict"] != "pending"
    assert "LV mask" in report["checks"]["jet_chamber"]["detail"]


def test_lv_only_volume_cannot_locate_aorta():
    vel, eval_mask, lv, *_rest, spacing, times = _two_chamber()
    report = evaluate_osu_gate(
        vel, lv, spacing, times, lv_mask=lv, mask_is_lv=True)
    assert report["checks"]["jet_chamber"]["verdict"] == "pending"
    assert "whole-heart" in report["checks"]["jet_chamber"]["detail"]


def test_jet_is_persist_complement_not_ever_exceeded():
    t_len, n = 10, 6
    vel = np.zeros((t_len, n, n, n, 3))
    mask = np.ones((n, n, n), dtype=bool)
    vel[0, ..., 0] = 0.40  # one-frame spike
    jet = jet_remainder_mask(vel, mask)
    assert not jet.any()


def test_rms_median_is_not_peak_biased():
    t_len, n = 20, 8
    vel = np.zeros((t_len, n, n, n, 3))
    mask = np.zeros((n, n, n), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    vel[:, mask, 0] = 0.10
    vel[0, mask, 0] = 0.80  # one-frame spike
    rms = masked_rms_speed_median(vel, mask)
    # time-RMS of mostly 0.10 with one 0.80 frame: sqrt((19*0.01 + 0.64)/20) ≈ 0.20
    assert rms == pytest.approx(math.sqrt((19 * 0.01 + 0.64) / 20), rel=1e-6)
    assert rms < 0.80


def test_fourier_harmonic1_vs_flat():
    t_len, n = 20, 8
    vel = np.zeros((t_len, n, n, n, 3))
    mask = np.zeros((n, n, n), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    t = np.arange(t_len, dtype=np.float64)
    vel[:, mask, 0] = np.sin(2.0 * np.pi * t / t_len)[:, None]
    assert temporal_low_harmonic_frac(vel, mask) == pytest.approx(1.0, abs=1e-6)

    vel_flat = np.zeros_like(vel)
    sig = np.zeros(t_len)
    for k in range(1, 10):
        sig += np.sin(2.0 * np.pi * k * t / t_len)
    vel_flat[:, mask, 0] = sig[:, None]
    assert temporal_low_harmonic_frac(vel_flat, mask) < FOURIER_LOW_FRAC_MIN

    vel_dc = np.zeros_like(vel)
    vel_dc[:, mask, 0] = 0.30
    assert temporal_low_harmonic_frac(vel_dc, mask) == 0.0


def test_write_gate_json_and_qa(tmp_path):
    packed = _two_chamber(ao_speed=1.4625, pulsatile=True)
    vel, eval_mask, *_rest, spacing, _times = packed
    report = _score(*packed)
    path = write_gate_json(tmp_path / "osu_gate.json", report)
    loaded = json.loads(path.read_text())
    assert loaded["verdict"] == "pass"
    assert "OSU gate: PASS" in format_osu_gate(report)
    qdir = write_osu_qa(
        tmp_path / "qa", velocity=vel, spacing_m=spacing,
        origin_m=(0, 0, 0), mask=eval_mask)
    assert (qdir / "peak_speed.npy").is_file()
    assert (qdir / "slices_peak.png").is_file()
    assert (qdir / "README.txt").is_file()
    assert (qdir / "ORIENTATION.txt").is_file()
    readme = (qdir / "README.txt").read_text()
    assert "display-scaled" in readme
    assert "HIDE peak_speed" in readme
    nibabel = pytest.importorskip("nibabel")
    img = nibabel.load(str(qdir / "peak_speed.nii.gz"))
    assert img.header.get_zooms()[:3] == pytest.approx((2.0, 2.0, 2.0))
    data = np.asanyarray(img.dataobj)
    assert float(data.max()) > 100.0  # display-scaled, not raw m/s
    # Masked draw volume must not keep the unmasked VENC·√3 noise tail.
    assert float(data[~eval_mask].max()) == 0.0


def test_parse_index_span_is_inclusive():
    assert parse_index_span("10:12", 20) == slice(10, 13)


def test_box_threshold_keeps_bright_core():
    img = np.zeros((12, 12, 12))
    img[2:4, 2:4, 2:4] = 1.0
    img[6:10, 6:10, 6:10] = 5.0
    m = box_threshold_mask(
        img,
        (slice(0, 12), slice(0, 12), slice(0, 12)),
        percentile=50.0)
    assert m[8, 8, 8]
    assert not m[0, 0, 0]
    assert not m[3, 3, 3]
    assert int(m.sum()) == 64


def test_sv_rel_tol_is_pre_registered():
    assert math.isclose(DATASET11_AAO_SV_ML, 93.6)
    assert math.isclose(AAO_SV_REL_TOL, 0.40)
    assert math.isclose(VALVE_SV_REL_TOL, 0.40)
    assert math.isclose(LV_TISSUE_RATIO_MIN, 3.0)
    assert math.isclose(FOURIER_LOW_FRAC_MIN, 0.55)
