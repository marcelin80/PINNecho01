"""Eulerian stasis / RT proxy on public volumes (not a thrombus model)."""

import math

import numpy as np
import torch

from pinnecho.config import Config
from pinnecho.data import synthetic_public_volume_frames
from pinnecho.data.ibfe_dataset import evaluate_ibfe, train_ibfe
from pinnecho.eval.stasis import (
    compare_point_stasis, compare_registered_volumes, compare_volume_stasis,
    resolve_speed_thresh, volume_stasis_report,
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
    assert m["persist_stasis_ml_pred"] > m["persist_stasis_ml_true"]
    assert m["persist_stasis_dice"] < 1.0


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
