"""OSU fail-cheap gate (no MATLAB, no Zenodo download)."""

import json
import math

import numpy as np
import pytest

from pinnecho.data.osu_core import pack_osu_public_volume
from pinnecho.data.osu_gate import (
    AAO_SV_REL_TOL, DATASET11_AAO_SV_ML, analysis_plane_mask,
    box_threshold_mask, evaluate_osu_gate, format_osu_gate,
    jet_remainder_mask, parse_index_span, plane_stroke_volume_ml,
    write_gate_json, write_osu_qa,
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


def _two_chamber(t_len=8, n=16, lv_speed=0.40, ao_speed=0.80):
    """Whole-heart eval: LV (left) + aorta (right)."""
    vel = np.zeros((t_len, n, n, n, 3))
    lv = np.zeros((n, n, n), dtype=bool)
    ao = np.zeros((n, n, n), dtype=bool)
    lv[2:8, 4:12, 4:12] = True
    ao[10:14, 6:10, 6:10] = True
    eval_mask = lv | ao
    vel[:, lv, 0] = lv_speed
    vel[:, ao, 2] = ao_speed
    # Aao analysis plane: one z-slice of the aorta lumen
    aao_plane = np.zeros_like(ao)
    aao_plane[10:14, 6:10, 8] = True
    spacing = (0.002, 0.002, 0.002)
    times = _times(t_len)
    return vel, eval_mask, lv, ao, aao_plane, spacing, times


def test_gate_placeholder_is_pending():
    vel, eval_mask, *_rest, spacing, times = _two_chamber()
    report = evaluate_osu_gate(vel, eval_mask, spacing, times)
    assert report["verdict"] == "pending"
    assert report["checks"]["speed_p99"]["verdict"] == "pending"
    assert report["checks"]["aao_sv"]["verdict"] == "pending"
    assert report["checks"]["jet_chamber"]["verdict"] == "pending"
    assert report["mask_is_lv"] is False


def test_gate_pass_when_jet_in_lv_and_sv_in_band():
    # Aao plane is 4×4 voxels × 0.002² = 6.4e-5 m².
    # 93.6 mL/beat = 9.36e-5 m³/s → v = 1.4625 m/s.
    vel, eval_mask, lv, ao, aao_plane, spacing, times = _two_chamber(
        ao_speed=1.4625)
    report = evaluate_osu_gate(
        vel, eval_mask, spacing, times,
        lv_mask=lv, aorta_mask=ao, aao_plane_mask=aao_plane, mask_is_lv=True)
    assert report["checks"]["speed_p99"]["verdict"] == "pass"
    assert report["checks"]["aao_sv"]["verdict"] == "pass"
    assert report["checks"]["jet_chamber"]["verdict"] == "pass"
    assert report["verdict"] == "pass"
    assert report["checks"]["jet_chamber"]["jet_in_lv_frac"] > 0.5


def test_gate_fail_when_remainder_is_aorta():
    vel, eval_mask, lv, ao, aao_plane, spacing, times = _two_chamber(
        lv_speed=0.0, ao_speed=1.4625)
    report = evaluate_osu_gate(
        vel, eval_mask, spacing, times,
        lv_mask=lv, aorta_mask=ao, aao_plane_mask=aao_plane, mask_is_lv=True)
    assert report["checks"]["jet_chamber"]["verdict"] == "fail"
    assert report["verdict"] == "fail"
    assert report["checks"]["jet_chamber"]["jet_in_aorta_frac"] >= 0.50


def test_gate_fail_p99_on_slow_lv():
    vel, eval_mask, lv, ao, aao_plane, spacing, times = _two_chamber(
        lv_speed=0.02, ao_speed=1.4625)
    # p99 is scored on the LV mask
    report = evaluate_osu_gate(
        vel, eval_mask, spacing, times,
        lv_mask=lv, aorta_mask=ao, aao_plane_mask=aao_plane, mask_is_lv=True)
    assert report["checks"]["speed_p99"]["verdict"] == "fail"
    assert report["verdict"] == "fail"


def test_gate_fail_sv_out_of_band():
    vel, eval_mask, lv, ao, aao_plane, spacing, times = _two_chamber(
        ao_speed=3.00)  # ~192 mL, well outside ±40 % of 93.6
    report = evaluate_osu_gate(
        vel, eval_mask, spacing, times,
        lv_mask=lv, aorta_mask=ao, aao_plane_mask=aao_plane, mask_is_lv=True)
    rel = abs(report["checks"]["aao_sv"]["value_ml"] - DATASET11_AAO_SV_ML) / DATASET11_AAO_SV_ML
    assert rel > AAO_SV_REL_TOL
    assert report["checks"]["aao_sv"]["verdict"] == "fail"
    assert report["verdict"] == "fail"


def test_aao_plane_alone_does_not_locate_jet():
    vel, eval_mask, _lv, _ao, aao_plane, spacing, times = _two_chamber()
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


def test_write_gate_json_and_qa(tmp_path):
    vel, eval_mask, lv, ao, aao_plane, spacing, times = _two_chamber(
        ao_speed=1.4625)
    report = evaluate_osu_gate(
        vel, eval_mask, spacing, times,
        lv_mask=lv, aorta_mask=ao, aao_plane_mask=aao_plane, mask_is_lv=True)
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
