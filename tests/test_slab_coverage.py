"""Echo-plane slab coverage of a synthetic jet (no training)."""

import numpy as np
import pytest

from pinnecho.eval.slab_coverage import (
    slab_coverage_report,
    union_slab_mask,
    views_from_cavity,
    voxel_centers,
)


def _sheet_jet_volume():
    """Cavity cube; fast sheet on the A4C midplane (y = centre)."""
    n, t_len = 16, 4
    vel = np.zeros((t_len, n, n, n, 3))
    mask = np.ones((n, n, n), dtype=bool)
    # A4C is y = centre. Put a 0.40 m/s sheet on the two mid y-indices.
    vel[:, :, 7:9, :, 2] = 0.40
    origin = np.array([0.0, 0.0, 0.0])
    spacing = np.array([0.002, 0.002, 0.002])
    return vel, mask, spacing, origin


def test_voxel_centers_match_origin_and_spacing():
    xyz = voxel_centers((1.0, 2.0, 3.0), (0.1, 0.2, 0.3), (2, 2, 2))
    assert xyz.shape == (2, 2, 2, 3)
    assert xyz[0, 0, 0].tolist() == [1.0, 2.0, 3.0]
    assert xyz[1, 1, 1].tolist() == pytest.approx([1.1, 2.2, 3.3])


def test_a4c_covers_midplane_jet():
    vel, mask, spacing, origin = _sheet_jet_volume()
    rows = slab_coverage_report(
        vel, mask, spacing, origin, speed_thresh=0.10, thickness_frac=0.12)
    by = {r["name"]: r for r in rows}
    assert by["a4c"]["jet_ml"] > 0.0
    assert by["a4c"]["jet_frac_in_slab"] == pytest.approx(1.0, abs=1e-9)
    assert by["a4c+a2c+plax+psax"]["jet_frac_in_slab"] == pytest.approx(1.0, abs=1e-9)
    assert by["a4c+a2c+plax+psax"]["jet_ml_in_slab"] >= by["a4c"]["jet_ml_in_slab"]


def test_a2c_sheet_is_missed_by_a4c():
    n, t_len = 16, 4
    vel = np.zeros((t_len, n, n, n, 3))
    mask = np.ones((n, n, n), dtype=bool)
    # Fast sheet on the A2C midplane (x = centre), away from y = centre.
    vel[:, 7:9, 13:, :, 0] = 0.40
    rows = slab_coverage_report(
        vel, mask, np.array([0.002, 0.002, 0.002]), np.zeros(3),
        speed_thresh=0.10, thickness_frac=0.12)
    by = {r["name"]: r for r in rows}
    assert by["a4c"]["jet_frac_in_slab"] < 0.25
    assert by["a4c+a2c"]["jet_frac_in_slab"] > 0.9
    assert by["a4c+a2c+plax+psax"]["jet_frac_in_slab"] >= by["a4c+a2c"]["jet_frac_in_slab"]


def test_four_view_is_union_of_slabs():
    vel, mask, spacing, origin = _sheet_jet_volume()
    cavity = mask
    xyz = voxel_centers(origin, spacing, cavity.shape)
    planes4, _ = views_from_cavity(
        cavity, origin, spacing, views=("a4c", "a2c", "plax", "psax"))
    planes_a4c, _ = views_from_cavity(cavity, origin, spacing, views=("a4c",))
    u4 = union_slab_mask(xyz, planes4)
    ua = union_slab_mask(xyz, planes_a4c)
    assert ua.sum() <= u4.sum()
    assert np.all(ua <= u4)


def test_along_beam_frac_is_a_fraction():
    vel, mask, spacing, origin = _sheet_jet_volume()
    rows = slab_coverage_report(vel, mask, spacing, origin, speed_thresh=0.10)
    for r in rows:
        if np.isfinite(r["jet_along_beam_frac"]):
            assert 0.0 <= r["jet_along_beam_frac"] <= 1.0
