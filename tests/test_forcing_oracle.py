"""Tests for the band-localized / oracle forcing diagnostic and validator.

Real IBFE forcing is band-localized (nonzero only in a ~3-cell wall shell, zero
in the cavity interior) and is evaluation-only ground truth. These cover the
support-preserving shuffle, the geometric band-mask control, and the validator's
band-coverage reporting.
"""

import dataclasses

import torch

from pinnecho.config import Config
from pinnecho.data.ibfe_dataset import (
    apply_forcing_control,
    forcing_band_mask,
)
from pinnecho.data.ibfe_validate import validate_ibfe_frames
from pinnecho.data.load_ibfe_output import synthetic_ibfe_frames_3d

torch.set_default_dtype(torch.float64)


def _band_localized_forcing(n, dim, band_frac=0.3, seed=0):
    """A forcing field that is zero outside a random 'band' subset of points."""
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(n, dim, generator=g)
    keep = torch.zeros(n, dtype=torch.bool)
    nb = int(band_frac * n)
    keep[torch.randperm(n, generator=g)[:nb]] = True
    f[~keep] = 0.0
    return f, keep


def test_forcing_band_mask_matches_support():
    f, keep = _band_localized_forcing(500, 3, band_frac=0.25)
    mask = forcing_band_mask(f)
    assert torch.equal(mask, keep)
    assert 0 < int(mask.sum()) < 500


def test_support_preserving_shuffle_keeps_interior_zero():
    f, keep = _band_localized_forcing(400, 3, band_frac=0.3, seed=1)
    coords = torch.rand(400, 4)
    out = apply_forcing_control(f, coords, "shuffle", seed=0, dim=3)
    # Interior (outside band) must stay exactly zero -- support preserved.
    assert torch.count_nonzero(out[~keep]) == 0
    # Band support unchanged, and the multiset of band vectors is a permutation.
    assert torch.equal(forcing_band_mask(out), keep)
    a = torch.sort(f[keep].norm(dim=1)).values
    b = torch.sort(out[keep].norm(dim=1)).values
    assert torch.allclose(a, b)
    # It is an actual decorrelation (not identity) for a non-trivial band.
    assert not torch.allclose(out[keep], f[keep])


def test_band_mask_control_is_geometric_only():
    f, keep = _band_localized_forcing(600, 3, band_frac=0.3, seed=2)
    coords = torch.rand(600, 4)
    out = apply_forcing_control(f, coords, "band_mask", seed=0, dim=3)
    assert torch.count_nonzero(out[~keep]) == 0          # zero outside band
    # All band vectors share one magnitude (the mean band |f|): no per-point
    # trajectory content, only a geometric wall-position placeholder.
    norms = out[keep].norm(dim=1)
    assert torch.allclose(norms, norms[0] * torch.ones_like(norms), atol=1e-6)
    assert abs(float(norms[0]) - float(f[keep].norm(dim=1).mean())) < 1e-6
    # Points toward the cavity centroid (inward).
    centroid = coords[:, :3].mean(dim=0, keepdim=True)
    inward = (centroid - coords[:, :3])
    inward = inward / inward.norm(dim=1, keepdim=True)
    cos = (out[keep] / norms[:, None] * inward[keep]).sum(dim=1)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-6)


def test_apply_forcing_control_none_is_identity():
    f, _ = _band_localized_forcing(50, 2, seed=3)
    coords = torch.rand(50, 3)
    assert torch.equal(apply_forcing_control(f, coords, None, 0, 2), f)


def test_validator_reports_band_fraction_for_localized_forcing():
    cfg = Config()
    frames = synthetic_ibfe_frames_3d(cfg, n_fluid=600, n_wall=200, n_frames=3)
    f, keep = _band_localized_forcing(frames.coords_fluid.shape[0], 3,
                                      band_frac=0.2, seed=4)
    frames = dataclasses.replace(frames, forcing_fluid=f)
    rep = validate_ibfe_frames(frames)
    assert rep.ok
    assert abs(rep.stats["forcing_band_fraction"] - float(keep.float().mean())) < 1e-9
    # Band-localized forcing must NOT trip the all-zero or full-cavity warnings.
    assert not any("all zeros" in w for w in rep.warnings)
    assert not any("band-localized" in w for w in rep.warnings)


def test_validator_warns_on_full_cavity_forcing():
    # The manufactured synthetic forcing fills the whole cavity -> warn that this
    # is not the band-localized real IB forcing (FSI claims unsafe).
    cfg = Config()
    frames = synthetic_ibfe_frames_3d(cfg, n_fluid=600, n_wall=200, n_frames=3)
    rep = validate_ibfe_frames(frames)
    assert rep.stats["forcing_band_fraction"] > 0.7
    assert any("band-localized" in w for w in rep.warnings)
