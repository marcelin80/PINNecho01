"""Smoke tests for the band-localized forcing oracle dry-run.

Covers the band-localization option on the synthetic generators, the band-coverage
reporter, and the A / exact / shuffle / band-mask sweep plumbing.
"""

import math

import torch

from pinnecho.config import Config
from pinnecho.data import band_coverage_fraction, forcing_band_mask
from pinnecho.data.ibfe_dataset import apply_forcing_control
from pinnecho.data.load_ibfe_output import (
    synthetic_ibfe_frames,
    synthetic_ibfe_frames_3d,
)
from pinnecho.train.band_oracle import (
    CONTRASTS,
    default_oracle_conditions,
    run_band_oracle_sweep,
)

torch.set_default_dtype(torch.float32)


def test_band_localized_forcing_zeros_interior():
    cfg = Config()
    frames = synthetic_ibfe_frames_3d(cfg, n_fluid=1500, n_wall=300, n_frames=4,
                                      forcing_band_frac=0.25)
    cov = band_coverage_fraction(frames.forcing_fluid)
    # A proper band: active over part of the cavity, zero elsewhere.
    assert 0.0 < cov < 1.0
    band = forcing_band_mask(frames.forcing_fluid)
    assert torch.count_nonzero(frames.forcing_fluid[~band]) == 0


def test_full_cavity_default_has_full_coverage():
    cfg = Config()
    frames = synthetic_ibfe_frames_3d(cfg, n_fluid=1200, n_wall=300, n_frames=4)
    # Default (no band_frac) is the manufactured full-cavity field.
    assert band_coverage_fraction(frames.forcing_fluid) > 0.95


def test_thicker_band_covers_more():
    cfg = Config()
    thin = synthetic_ibfe_frames(cfg, n_fluid=2000, n_wall=300, n_frames=4,
                                 forcing_band_frac=0.15)
    thick = synthetic_ibfe_frames(cfg, n_fluid=2000, n_wall=300, n_frames=4,
                                  forcing_band_frac=0.35)
    assert (band_coverage_fraction(thick.forcing_fluid)
            > band_coverage_fraction(thin.forcing_fluid))


def test_shuffle_on_banded_frames_preserves_support():
    cfg = Config()
    frames = synthetic_ibfe_frames_3d(cfg, n_fluid=1500, n_wall=300, n_frames=4,
                                      forcing_band_frac=0.25)
    band = forcing_band_mask(frames.forcing_fluid)
    out = apply_forcing_control(frames.forcing_fluid, frames.coords_fluid,
                                "shuffle", seed=0, dim=3)
    assert torch.count_nonzero(out[~band]) == 0        # interior stays zero
    assert torch.equal(forcing_band_mask(out), band)   # support unchanged


def test_default_conditions_are_baseline_plus_oracle_controls():
    conds = default_oracle_conditions()
    names = [c.name for c in conds]
    assert names[0] == "A (baseline)"
    assert conds[0].backbone == "baseline"
    fsi = [c for c in conds if c.backbone == "fsi_informed"]
    assert len(fsi) == 3
    controls = {c.forcing_control for c in fsi}
    assert controls == {None, "shuffle", "band_mask"}


def test_band_oracle_sweep_runs_and_reports():
    res = run_band_oracle_sweep(
        Config(), dim=2, band_frac=0.25, steps=12, lbfgs_iters=1, seeds=(0,),
        use_traction=False, n_fluid=900, n_wall=200, n_frames=3, n_windows=2,
        verbose=False)
    assert 0.0 < res["band_coverage"] < 1.0
    assert len(res["records"]) == len(default_oracle_conditions())
    for r in res["records"]:
        assert math.isfinite(r["pressure_relL2_mean"])
        assert math.isfinite(r["vel_relL2_speed_mean"])
    # Paired contrasts present with the sign-test fields.
    assert set(res["contrasts"]) == set(CONTRASTS)
    st = res["contrasts"]["exact_vs_baseline"]["pressure_relL2"]
    for k in ("mean_diff", "t_stat", "n_wins", "n", "sign_p_one_sided"):
        assert k in st
