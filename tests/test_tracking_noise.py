"""Tests for named wall-tracking noise presets (STE literature calibration)."""

import torch

from pinnecho.config import Config, config_from_dict
from pinnecho.data import (
    TRACKING_PRESETS,
    apply_tracking_noise,
    relative_rms_error,
    resolve_tracking,
    synthetic_ibfe_frames,
)
from pinnecho.data.ibfe_dataset import make_ibfe_batch_builder
from pinnecho.train.train import build_toy_dataset

torch.set_default_dtype(torch.float32)


def test_exact_preset_is_identity():
    v = torch.randn(200, 2)
    out = apply_tracking_noise(v, "exact", seed=0)
    assert torch.equal(out, v)
    assert out.data_ptr() != v.data_ptr()          # cloned, not aliased


def test_placeholder_reproduces_legacy_formula():
    v = torch.linspace(-1.0, 1.0, 300).reshape(150, 2)
    # Legacy Stage-1: v * (1 - 0.08) + N(0, (0.10 * rms)^2), seed+7.
    rms = float(v.pow(2).mean().sqrt().clamp_min(1e-9))
    g = torch.Generator().manual_seed(0 + 7)
    noise = torch.randn(v.shape, generator=g, dtype=v.dtype) * (0.10 * rms)
    legacy = v * (1.0 - 0.08) + noise
    got = apply_tracking_noise(v, "placeholder", seed=0)
    assert torch.allclose(got, legacy, atol=1e-6)


def test_ste_preset_matches_literature_numbers():
    spec = resolve_tracking("ste")
    assert spec.bias == -0.05
    assert spec.noise == 0.09
    v = torch.ones(100, 2) * 0.4
    # Zero-noise override isolates the multiplicative bias.
    biased = apply_tracking_noise(v, "ste", seed=0, noise=0.0)
    assert torch.allclose(biased, v * 0.95, atol=1e-6)
    # Full STE noise: realized relative RMS is in the 5-15% band (bias + scatter).
    noisy = apply_tracking_noise(v, "ste", seed=1)
    err = relative_rms_error(noisy, v)
    assert 0.04 < err < 0.20


def test_unknown_preset_raises():
    import pytest
    with pytest.raises(KeyError):
        resolve_tracking("not-a-preset")


def test_config_wall_tracking_roundtrip():
    c = Config()
    assert c.wall_tracking.preset == "placeholder"
    c2 = config_from_dict({"wall_tracking": {"preset": "ste"}})
    assert c2.wall_tracking.preset == "ste"
    assert c2.wall_tracking.bias is None


def test_toy_dataset_exact_walls_coincide():
    ds, _ = build_toy_dataset(Config(), tracking="exact", seed=0)
    assert torch.equal(ds.extras["wall_velocity_fsi"],
                       ds.extras["wall_velocity_kin"])
    assert ds.extras["wall_tracking"].name == "exact"


def test_toy_dataset_placeholder_is_default_and_diverges():
    ds, _ = build_toy_dataset(Config(), seed=0)
    assert ds.extras["wall_tracking"].name == "placeholder"
    err = relative_rms_error(ds.extras["wall_velocity_kin"],
                             ds.extras["wall_velocity_fsi"])
    assert err > 0.05


def test_ibfe_builder_default_is_coincident():
    frames = synthetic_ibfe_frames(Config(), n_fluid=400, n_wall=80, n_frames=3)
    n_all = frames.velocity_wall.shape[0]
    b_exact = make_ibfe_batch_builder(frames, tracking=None, n_wall=n_all,
                                      n_data=50, n_col=50)(0)
    b_ste = make_ibfe_batch_builder(frames, tracking="ste", n_wall=n_all,
                                    n_data=50, n_col=50, seed=0)(0)
    # Default (None) uses the raw FSI wall; ste corrupts it.
    truth = frames.velocity_wall.to(b_exact["wall"]["u_wall"].dtype)
    assert torch.allclose(b_exact["wall"]["u_wall"], truth, atol=1e-6)
    assert not torch.allclose(b_ste["wall"]["u_wall"], truth, atol=1e-4)


def test_ste_is_milder_than_placeholder():
    """Literature STE (5%/9%) is a smaller perturbation than the 8%/10% placeholder."""
    v = torch.randn(500, 2)
    e_ph = relative_rms_error(apply_tracking_noise(v, "placeholder", seed=0), v)
    e_ste = relative_rms_error(apply_tracking_noise(v, "ste", seed=0), v)
    assert e_ste < e_ph
    assert set(TRACKING_PRESETS) == {"exact", "placeholder", "ste"}
