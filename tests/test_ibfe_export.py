"""Tests for the synthetic IBFE exporter (real-data-ready IBFEFrames interface)."""

import torch

from pinnecho.config import Config, CollocationConfig
from pinnecho.data.load_ibfe_output import IBFEFrames, synthetic_ibfe_frames
from pinnecho.data.synthesize_doppler import synthesize_doppler

torch.set_default_dtype(torch.float64)


def test_synthetic_ibfe_frames_shapes():
    config = Config()
    frames = synthetic_ibfe_frames(config, n_fluid=300, n_wall=120, n_frames=4)
    assert isinstance(frames, IBFEFrames)

    n_f = frames.coords_fluid.shape[0]
    assert frames.coords_fluid.shape == (n_f, 3)
    assert frames.velocity_fluid.shape == (n_f, 2)
    assert frames.pressure_fluid.shape == (n_f, 1)
    assert frames.forcing_fluid.shape == (n_f, 2)

    n_w = frames.coords_wall.shape[0]
    assert frames.coords_wall.shape == (n_w, 3)
    assert frames.normals_wall.shape == (n_w, 2)
    assert frames.velocity_wall.shape == (n_w, 2)
    assert frames.traction_wall.shape == (n_w, 2)

    # Normals are unit vectors.
    norms = frames.normals_wall.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    # Metadata carried through.
    assert frames.cycle_period == config.flow.period
    assert frames.rho == config.flow.density
    assert frames.mu == config.flow.viscosity
    # Valves empty in the closed synthetic cavity.
    assert frames.coords_mitral.shape == (0, 3)


def test_ibfe_frames_feed_doppler_synthesis():
    """IBFEFrames fluid field flows through the canonical Doppler synthesiser."""
    config = Config()
    frames = synthetic_ibfe_frames(config, n_fluid=500, n_wall=50, n_frames=4)
    transducers = torch.tensor([[0.0, -0.075]], dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    m = synthesize_doppler(
        frames.coords_fluid, frames.velocity_fluid, transducers,
        v_nyquist=0.8, snr_db=20.0, sparsity=0.5, dealias=False, rng=g,
    )
    assert m.v_beam.shape[1] == 1
    assert m.coords_meas.shape[0] == m.v_beam.shape[0]
    # Sparsity ~0.5 keeps roughly half.
    assert 0.35 * 500 < m.v_beam.shape[0] < 0.65 * 500
