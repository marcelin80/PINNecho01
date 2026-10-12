"""Eulerian stasis / RT proxy on public volumes (not a thrombus model)."""

import math

import numpy as np
import pytest
import torch

from pinnecho.config import Config
from pinnecho.data import synthetic_public_volume_frames
from pinnecho.data.ibfe_dataset import evaluate_ibfe, train_ibfe
from pinnecho.eval.stasis import (
    blur_velocity, compare_point_stasis, compare_registered_volumes,
    compare_volume_stasis, peak_speed, peak_speed_report,
    persistent_stasis_mask, resolve_speed_thresh, volume_stasis_report,
)


def test_resolve_thresh_is_clinical_on_fast_fields():
    speed = np.linspace(0.0, 1.0, 1000)
    assert resolve_speed_thresh(speed, speed_thresh=0.10) == 0.10


def test_resolve_thresh_shrinks_on_slow_toy_fields():
    speed = np.linspace(0.0, 0.04, 1000)
    t = resolve_speed_thresh(speed, speed_thresh=0.10)
    assert t < 0.10
    assert t == 0.25 * np.quantile(speed, 0.99)


def test_resolve_thresh_clinical_on_corrado_like_p99():
    # p99 = 26 cm/s (measured Corrado interior) must keep 10 cm/s,
    # not drop to 0.25*p99 ≈ 6.5 cm/s.
    speed = np.linspace(0.0, 0.26, 10_000)
    assert resolve_speed_thresh(speed, speed_thresh=0.10) == 0.10


def test_volume_stasis_fraction_on_split_cavity():
    t_len, n = 4, 8
    vel = np.zeros((t_len, n, n, n, 3))
    mask = np.zeros((n, n, n), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    # Left half of the box is stagnant; right half is 0.4 m/s.
    vel[:, 2:4, 2:6, 2:6, 0] = 0.0
    vel[:, 4:6, 2:6, 2:6, 0] = 0.40
    report = volume_stasis_report(
        vel, mask, spacing=(0.002, 0.002, 0.002), speed_thresh=0.10)
    assert report["stasis_frac"] == 0.5
    assert report["persist_stasis_frac"] == 0.5
    assert report["cavity_ml"] > 0.0
    assert report["speed_mean"] > 0.0


def test_identical_fields_have_perfect_stasis_dice():
    vel = torch.tensor([[0.02, 0.0, 0.0], [0.30, 0.0, 0.0],
                        [0.01, 0.0, 0.0], [0.40, 0.0, 0.0]])
    m = compare_point_stasis(vel, vel, speed_thresh=0.10)
    assert m["stasis_dice"] == 1.0
    assert m["rt_proxy_relL2"] < 1e-12
    assert m["stasis_frac_true"] == m["stasis_frac_pred"]


def test_scrambled_slow_fast_lowers_dice():
    true = torch.tensor([[0.02, 0.0, 0.0], [0.30, 0.0, 0.0],
                         [0.01, 0.0, 0.0], [0.40, 0.0, 0.0]])
    pred = torch.tensor([[0.30, 0.0, 0.0], [0.02, 0.0, 0.0],
                         [0.40, 0.0, 0.0], [0.01, 0.0, 0.0]])
    m = compare_point_stasis(true, pred, speed_thresh=0.10)
    assert m["stasis_dice"] == 0.0


def _split_cavity_volume():
    t_len, n = 4, 8
    vel = np.zeros((t_len, n, n, n, 3))
    mask = np.zeros((n, n, n), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    vel[:, 2:4, 2:6, 2:6, 0] = 0.0
    vel[:, 4:6, 2:6, 2:6, 0] = 0.40
    return vel, mask


def test_compare_volume_identical_is_perfect():
    vel, mask = _split_cavity_volume()
    m = compare_volume_stasis(vel, vel, mask, spacing=(0.002, 0.002, 0.002))
    assert m["persist_stasis_dice"] == 1.0
    assert m["jet_dice"] == 1.0
    assert m["jet_ml_true"] == m["jet_ml_pred"]
    assert m["persist_stasis_ml_true"] == m["persist_stasis_ml_pred"]


def test_absolute_report_keeps_clinical_cut_on_slow_field():
    vel, mask = _split_cavity_volume()
    vel = vel * 0.1  # p99 well below 10 cm/s
    auto = volume_stasis_report(
        vel, mask, spacing=(0.002, 0.002, 0.002), speed_thresh=0.10)
    fixed = volume_stasis_report(
        vel, mask, spacing=(0.002, 0.002, 0.002), speed_thresh=0.10,
        absolute=True)
    assert auto["speed_thresh"] < 0.10
    assert fixed["speed_thresh"] == 0.10
    assert fixed["jet_ml"] < auto["jet_ml"]


def test_registered_compare_uses_overlap_mask():
    vel, mask = _split_cavity_volume()
    other = vel.copy()
    other_mask = np.zeros_like(mask)
    other_mask[2:6, 4:6, 2:6] = True  # jet half only
    other[:, other_mask, :] *= 0.2  # 0.40 → 0.08, below the 10 cm/s cut
    m = compare_registered_volumes(
        vel, mask, other, other_mask, (0.002, 0.002, 0.002),
        origin_a=(0.0, 0.0, 0.0), origin_b=(0.0, 0.0, 0.0),
        name_a="mri", name_b="piv")
    assert m["cavity_ml_piv"] < m["cavity_ml_mri"]
    assert m["overlap_ml"] == m["cavity_ml_piv"]
    assert m["origin_delta_vox"] == 0.0
    assert m["overlap_jet_ml_true"] > m["overlap_jet_ml_pred"]
    assert 0.0 <= m["overlap_jet_dice"] <= 1.0


def test_registered_compare_rejects_unaligned_grids():
    vel, mask = _split_cavity_volume()
    try:
        compare_registered_volumes(
            vel, mask, vel[:, :-1], mask[:-1], (0.002, 0.002, 0.002))
    except ValueError as exc:
        assert "not registered" in str(exc)
    else:
        raise AssertionError("expected grid mismatch")


def test_compare_volume_zero_pred_eats_the_jet():
    vel, mask = _split_cavity_volume()
    m = compare_volume_stasis(
        vel, np.zeros_like(vel), mask, spacing=(0.002, 0.002, 0.002))
    assert m["jet_ml_pred"] == 0.0
    assert m["jet_dice"] == 0.0
    assert m["signed_jet_err"] == -1.0
    assert m["jet_recall"] == 0.0
    assert math.isnan(m["jet_precision"])
    assert m["persist_stasis_ml_pred"] > m["persist_stasis_ml_true"]
    assert m["persist_stasis_dice"] < 1.0
    assert m["slow_speed_relL2"] >= 0.0
    assert m["peak_ratio_median"] == 0.0


def test_compare_volume_identical_has_unit_precision_recall():
    vel, mask = _split_cavity_volume()
    m = compare_volume_stasis(vel, vel, mask, spacing=(0.002, 0.002, 0.002))
    assert m["signed_jet_err"] == 0.0
    assert m["jet_precision"] == 1.0
    assert m["jet_recall"] == 1.0
    assert m["slow_speed_relL2"] == 0.0
    assert m["peak_ratio_mean"] == 1.0
    assert m["peak_ratio_median"] == 1.0


def test_jet_is_majority_above_cut_not_ever_above():
    # 4 frames. Voxel A is fast in 1/4 → persist. Voxel B is fast in 3/4 → jet.
    t_len, n = 4, 4
    vel = np.zeros((t_len, n, n, n, 3))
    mask = np.zeros((n, n, n), dtype=bool)
    mask[1, 1, 1] = True  # A
    mask[2, 2, 2] = True  # B
    vel[0, 1, 1, 1, 0] = 0.40
    vel[:3, 2, 2, 2, 0] = 0.40
    persist = persistent_stasis_mask(vel, mask, speed_thresh=0.10, persist_frac=0.50)
    assert persist[1, 1, 1]
    assert not persist[2, 2, 2]
    report = volume_stasis_report(
        vel, mask, spacing=(0.002, 0.002, 0.002), speed_thresh=0.10, absolute=True)
    voxel_ml = 0.002 ** 3 * 1e6
    assert report["jet_ml"] == pytest.approx(voxel_ml)
    assert report["persist_stasis_ml"] == pytest.approx(voxel_ml)


def test_peak_speed_ratio_on_scaled_field():
    vel, mask = _split_cavity_volume()
    persist = persistent_stasis_mask(vel, mask, speed_thresh=0.10)
    cavity = mask if mask.ndim == 3 else mask.any(axis=0)
    jet = cavity & np.logical_not(persist)
    half = peak_speed_report(vel, 0.5 * vel, jet)
    assert half["peak_ratio_mean"] == pytest.approx(0.5)
    assert half["peak_ratio_median"] == pytest.approx(0.5)
    assert half["n_jet_peak"] > 0
    ident = peak_speed_report(vel, vel, jet)
    assert ident["peak_ratio_mean"] == pytest.approx(1.0)
    assert peak_speed(vel).shape == vel.shape[1:4]


def test_blur_gt_is_a_resolution_upper_bound():
    vel, mask = _split_cavity_volume()
    sharp = compare_volume_stasis(vel, vel, mask, spacing=(0.002, 0.002, 0.002))
    blurred = blur_velocity(vel, sigma_vox=1.0)
    soft = compare_volume_stasis(
        vel, blurred, mask, spacing=(0.002, 0.002, 0.002))
    assert sharp["jet_dice"] == 1.0
    assert soft["jet_dice"] < 1.0
    assert soft["peak_ratio_median"] < 1.0


def test_evaluate_ibfe_includes_stasis_keys():
    frames = synthetic_public_volume_frames(
        Config(), nx=10, ny=10, nz=12, n_frames=2, n_fluid=400, n_wall=80)
    _, metrics = train_ibfe(
        frames, backbone="baseline", steps=6, lbfgs_iters=1,
        use_traction=False, seed=0, verbose=False)
    for k in ("stasis_dice", "stasis_frac_true", "stasis_frac_pred",
              "rt_proxy_relL2"):
        assert k in metrics
        assert math.isfinite(metrics[k])
    assert 0.0 <= metrics["stasis_dice"] <= 1.0
