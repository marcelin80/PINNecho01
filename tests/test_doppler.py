"""Doppler sampling: single-component projection and noise behaviour."""

import numpy as np
import torch

from pinnecho.config import Config
from pinnecho.data.synthetic_lv import SyntheticLVFSI
from pinnecho.data.doppler import DopplerSampler, project_velocity


def test_projection_matches_dot_product():
    u = torch.tensor([[1.0], [0.0], [3.0]])
    v = torch.tensor([[0.0], [2.0], [4.0]])
    beam = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    proj = project_velocity(u, v, beam)
    expected = torch.tensor([[1.0], [2.0], [3.0 * 0.6 + 4.0 * 0.8]])
    assert torch.allclose(proj, expected)


def test_beam_directions_are_unit():
    c = Config()
    lv = SyntheticLVFSI(c.geometry, c.flow)
    sampler = DopplerSampler(c.doppler)
    rng = np.random.default_rng(0)
    X = lv.sample_interior(200, np.array([0.3]), rng)
    beam = sampler.beam_directions(X)
    norms = beam.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-10)


def test_noise_free_doppler_equals_true_projection():
    c = Config()
    c.doppler.noise_level = 0.0
    c.doppler.n_points_per_frame = 50
    c.doppler.n_frames = 3
    lv = SyntheticLVFSI(c.geometry, c.flow)
    sampler = DopplerSampler(c.doppler)
    meas = sampler.sample(lv, np.random.default_rng(0))
    assert torch.allclose(meas["doppler"], meas["doppler_true"], atol=1e-12)
    # And the true projection equals u.beam.
    proj = project_velocity(meas["u_true"], meas["v_true"], meas["beam"])
    assert torch.allclose(proj, meas["doppler_true"], atol=1e-12)


def test_noise_is_added_when_requested():
    c = Config()
    c.doppler.noise_level = 0.1
    lv = SyntheticLVFSI(c.geometry, c.flow)
    sampler = DopplerSampler(c.doppler)
    meas = sampler.sample(lv, np.random.default_rng(0))
    assert not torch.allclose(meas["doppler"], meas["doppler_true"])
