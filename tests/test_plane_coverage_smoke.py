"""Smoke tests for the 3D plane-coverage observability sweep plumbing."""

import math

import torch

from pinnecho.config import Config
from pinnecho.train.plane_coverage import (
    CoverageCondition,
    default_coverage_conditions,
    run_coverage_sweep,
)

torch.set_default_dtype(torch.float32)


def test_default_conditions_shape():
    conds = default_coverage_conditions()
    names = [c.name for c in conds]
    assert any(c.is_planar() for c in conds)
    assert any(not c.is_planar() for c in conds)   # idealized point-window ref
    # The full 4-view protocol is present.
    assert any(c.views == ("a4c", "a2c", "plax", "psax") for c in conds)
    assert len(set(names)) == len(names)           # unique labels


def test_coverage_sweep_runs_and_reports():
    recs = run_coverage_sweep(
        Config(), backbone="baseline", steps=15, lbfgs_iters=2, seeds=(0,),
        use_traction=False, n_fluid=800, n_wall=200, n_frames=3, verbose=False)
    assert len(recs) == len(default_coverage_conditions())
    for r in recs:
        for k in ("vel_relL2_speed", "vel_relL2_u", "vel_relL2_v", "vel_relL2_w",
                  "pressure_relL2"):
            assert math.isfinite(r[f"{k}_mean"]) and f"{k}_std" in r
        assert r["n_data"] > 0 and r["n_views"] >= 1


def test_more_views_sample_more_points():
    cfg = Config()
    recs = run_coverage_sweep(
        cfg, conditions=[CoverageCondition("a4c", views=("a4c",)),
                         CoverageCondition("4-view",
                                           views=("a4c", "a2c", "plax", "psax"))],
        backbone="baseline", steps=10, lbfgs_iters=1, seeds=(0,),
        use_traction=False, n_fluid=1000, n_wall=200, n_frames=3, verbose=False)
    by = {r["name"]: r for r in recs}
    # Four complementary slabs insonify strictly more points than one plane.
    assert by["4-view"]["n_data"] > by["a4c"]["n_data"]


def test_point_windows_cover_whole_volume():
    cfg = Config()
    recs = run_coverage_sweep(
        cfg, conditions=[CoverageCondition("pw3", n_point_windows=3)],
        backbone="baseline", steps=10, lbfgs_iters=1, seeds=(0,),
        use_traction=False, n_fluid=900, n_wall=200, n_frames=3, verbose=False)
    r = recs[0]
    assert not r["planar"] and r["n_views"] == 3
    assert r["n_data"] == 3 * 900  # 3 windows * all fluid points (n_fluid rows)
