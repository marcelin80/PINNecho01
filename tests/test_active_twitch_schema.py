"""Tests for the active_twitch schema-test IBFEFrames fixture.

This fixture mimics the first (format-only) IBFE export: band-localized forcing,
open base (no valves), traction on hold (zeros), a short partial segment. The
tests assert those contract properties and that the loader / validator / trainer
path is drop-in on it.
"""

import torch

from pinnecho.data import (
    band_coverage_fraction,
    forcing_band_mask,
    load_ibfe_output,
    synthetic_active_twitch_frames,
    train_ibfe,
    validate_ibfe_frames,
)
from pinnecho.data.ibfe_io import save_ibfe_npz, save_ibfe_manifest_csv

torch.set_default_dtype(torch.float32)


def _fixture(**kw):
    return synthetic_active_twitch_frames(n_fluid=1200, n_wall=400, n_frames=4, **kw)


def test_schema_shape_and_open_base():
    f = _fixture()
    assert f.spatial_dim == 3
    assert f.coords_fluid.shape[1] == 4 and f.velocity_fluid.shape[1] == 3
    assert f.forcing_fluid.shape[1] == 3
    # Open base: no valve arrays, but the (0, dim) shape contract is kept.
    assert f.coords_mitral.shape == (0, 4) and f.velocity_mitral.shape == (0, 3)
    assert f.coords_aortic.shape == (0, 4) and f.velocity_aortic.shape == (0, 3)


def test_traction_on_hold_is_zero():
    f = _fixture()
    assert f.traction_wall.shape == f.velocity_wall.shape
    assert torch.count_nonzero(f.traction_wall) == 0


def test_forcing_is_band_localized():
    f = _fixture(band_frac=0.25)
    cov = band_coverage_fraction(f.forcing_fluid)
    assert 0.0 < cov < 1.0                                   # a band, not full/empty
    band = forcing_band_mask(f.forcing_fluid)
    assert torch.count_nonzero(f.forcing_fluid[~band]) == 0   # interior exactly zero


def test_short_partial_segment():
    f = _fixture(segment_length=0.33)
    assert abs(f.cycle_period - 0.33) < 1e-6
    t = f.coords_fluid[:, 3]
    assert float(t.min()) >= -1e-6 and float(t.max()) <= 0.33 + 1e-6


def test_validator_accepts_schema_fixture():
    f = _fixture(band_frac=0.25)
    rep = validate_ibfe_frames(f)
    assert rep.ok
    # Band-localized => must not trip the all-zero or full-cavity warnings.
    assert 0.0 < rep.stats["forcing_band_fraction"] < 0.7
    assert not any("all zeros" in w for w in rep.warnings)
    assert not any("band-localized" in w for w in rep.warnings)


def test_npz_roundtrip_preserves_contract(tmp_path):
    f = _fixture()
    path = save_ibfe_npz(f, tmp_path / "active_twitch.npz")
    g = load_ibfe_output(str(path), spatial_dim=3)
    assert g.spatial_dim == 3
    assert g.coords_mitral.shape == (0, 4)                    # open base survives
    assert torch.count_nonzero(g.traction_wall) == 0          # zeros survive
    assert torch.allclose(g.forcing_fluid, f.forcing_fluid.to(g.forcing_fluid.dtype))
    assert validate_ibfe_frames(g).ok


def test_manifest_roundtrip_and_training_smoke(tmp_path):
    f = _fixture()
    manifest = save_ibfe_manifest_csv(f, tmp_path / "dir")
    g = load_ibfe_output(str(manifest), spatial_dim=3)
    assert validate_ibfe_frames(g).ok
    # Drop-in training (traction off, matching the on-hold contract) is finite.
    _, metrics = train_ibfe(g, backbone="fsi_informed", steps=10, lbfgs_iters=1,
                            use_traction=False, seed=0, verbose=False)
    assert all(metrics[k] == metrics[k] for k in metrics)     # no NaNs
