"""Tests for the canonical Doppler synthesiser (sparsity/aliasing/SNR)."""

import torch

from pinnecho.data.synthesize_doppler import (
    beam_directions,
    synthesize_doppler,
    wrap_to_nyquist,
)

torch.set_default_dtype(torch.float64)


def test_projection_is_single_component():
    # Velocity purely along +x; a transducer far on -x sees beams ~ +x.
    n = 200
    coords = torch.rand(n, 3)
    coords[:, 0] += 1.0  # push points to +x so beam ~ +x
    vel = torch.zeros(n, 2)
    vel[:, 0] = 0.5  # 0.5 m/s along x
    tr = torch.tensor([[-100.0, 0.0]])  # far away => beams ~ (+1, 0)
    m = synthesize_doppler(coords, vel, tr, v_nyquist=10.0, snr_db=100.0,
                           sparsity=0.0, dealias=True)
    # Beam ~ +x so projection ~ 0.5, and only the x-component is seen.
    assert torch.allclose(m.v_beam_clean, torch.full_like(m.v_beam_clean, 0.5),
                          atol=1e-3)


def test_aliasing_wraps_and_dealias_recovers():
    v = torch.tensor([[0.2], [0.9], [1.5], [-1.2]])
    a = 0.8
    wrapped = wrap_to_nyquist(v, a)
    assert (wrapped.abs() <= a + 1e-9).all()
    # 0.9 -> 0.9 - 1.6 = -0.7 ; 1.5 -> 1.5 - 1.6 = -0.1
    assert torch.allclose(wrapped, torch.tensor([[0.2], [-0.7], [-0.1], [0.4]]),
                          atol=1e-9)

    # With aliasing ON (dealias False) the measured signal is wrapped; with
    # dealias ON it equals the clean projection.
    n = 300
    coords = torch.rand(n, 3)
    vel = torch.randn(n, 2) * 1.5  # large enough to alias at v_nyquist=0.8
    tr = torch.tensor([[0.0, -2.0]])
    g = torch.Generator().manual_seed(0)
    aliased = synthesize_doppler(coords, vel, tr, v_nyquist=0.8, snr_db=200.0,
                                 sparsity=0.0, dealias=False, rng=g)
    assert (aliased.v_beam.abs() <= 0.8 + 1e-2).all()
    g = torch.Generator().manual_seed(0)
    clean = synthesize_doppler(coords, vel, tr, v_nyquist=0.8, snr_db=200.0,
                               sparsity=0.0, dealias=True, rng=g)
    assert torch.allclose(clean.v_beam, clean.v_beam_clean, atol=1e-2)


def test_sparsity_fraction():
    n = 5000
    coords = torch.rand(n, 3)
    vel = torch.randn(n, 2)
    tr = torch.tensor([[0.0, -1.0]])
    g = torch.Generator().manual_seed(1)
    m = synthesize_doppler(coords, vel, tr, sparsity=0.6, snr_db=100.0, rng=g)
    kept_frac = m.v_beam.shape[0] / n
    assert 0.35 < kept_frac < 0.45  # ~0.4 kept


def test_snr_level_is_reasonable():
    n = 20000
    coords = torch.rand(n, 3)
    vel = torch.randn(n, 2)
    tr = torch.tensor([[0.0, -1.0]])
    g = torch.Generator().manual_seed(2)
    snr_db = 20.0
    m = synthesize_doppler(coords, vel, tr, v_nyquist=100.0, snr_db=snr_db,
                           sparsity=0.0, dealias=True, rng=g)
    noise = m.v_beam - m.v_beam_clean
    sig_p = float(m.v_beam_clean.pow(2).mean())
    noise_p = float(noise.pow(2).mean())
    measured_snr = 10.0 * torch.log10(torch.tensor(sig_p / noise_p))
    assert abs(float(measured_snr) - snr_db) < 1.5


def test_multi_window_stacks():
    n = 100
    coords = torch.rand(n, 3)
    vel = torch.randn(n, 2)
    tr = torch.tensor([[0.0, -1.0], [1.0, -1.0]])
    m = synthesize_doppler(coords, vel, tr, sparsity=0.0, snr_db=100.0)
    assert m.v_beam.shape[0] == 2 * n
    assert set(m.window_id.unique().tolist()) == {0, 1}
