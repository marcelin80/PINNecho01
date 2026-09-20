"""Smoke test: the observability sweep runs one tiny condition end-to-end."""

import torch

from pinnecho.config import Config
from pinnecho.train.observability import (
    SweepCondition, run_sweep, plot_sweep, _snr_db,
)


def test_snr_db_monotonic():
    assert _snr_db(0.01) > _snr_db(0.1) > _snr_db(1.0)


def test_run_sweep_tiny(tmp_path):
    torch.set_default_dtype(torch.float32)
    cfg = Config()
    conds = [SweepCondition("windows=1", n_windows=1, noise_level=0.05,
                            n_points_per_frame=120),
             SweepCondition("windows=2", n_windows=2, noise_level=0.05,
                            n_points_per_frame=120)]
    recs = run_sweep(cfg, conditions=conds, backbone="fsi_informed",
                     steps=40, lbfgs_iters=0, seeds=(0,), use_traction=True,
                     verbose=False)
    assert len(recs) == 2
    for r in recs:
        assert "vel_relL2_speed_mean" in r and r["snr_db"] > 0
    p = plot_sweep(recs, tmp_path / "sweep.png", metric="vel_relL2_speed")
    assert p.exists() and p.stat().st_size > 0
