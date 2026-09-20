"""3D synthetic LV ground-truth invariants (analytic self-consistency)."""

import numpy as np
import torch

from pinnecho.config import Config
from pinnecho.data.synthetic_lv_3d import SyntheticLV3D
from pinnecho.physics.ns_residual import (
    continuity_residual_nd, momentum_residual_nd,
)


def _lv():
    torch.set_default_dtype(torch.float64)
    cfg = Config()
    return SyntheticLV3D(cfg.geometry, cfg.flow), cfg


def test_volume_is_preserved():
    lv, cfg = _lv()
    ts = torch.linspace(0.0, cfg.flow.period, 25).reshape(-1, 1)
    rx, ry, rz = lv.semi_axes(ts)
    vol = rx * ry * rz
    assert torch.allclose(vol, vol[0].expand_as(vol), rtol=1e-10, atol=1e-14)


def test_interior_divergence_free():
    lv, cfg = _lv()
    rng = np.random.default_rng(0)
    tv = np.linspace(0.0, cfg.flow.period, 8)
    X = lv.sample_interior(3000, tv, rng).requires_grad_(True)
    uvw = lv._velocity_from_graph(X)
    div = continuity_residual_nd([uvw[:, i:i + 1] for i in range(3)], X)
    assert float(div.detach().abs().max()) < 1e-8


def test_exact_no_slip_at_wall():
    lv, cfg = _lv()
    rng = np.random.default_rng(1)
    tv = np.linspace(0.0, cfg.flow.period, 8)
    wall = lv.sample_wall(2000, tv, rng)
    Xw = wall["X"].requires_grad_(True)
    # The vortex part must vanish on the wall, so u == u_wall there.
    u_vortex = lv.vortex_velocity(Xw)
    assert float(u_vortex.detach().abs().max()) < 1e-8
    u_tot = lv._velocity_from_graph(Xw)
    assert float((u_tot - wall["wall_velocity"]).detach().abs().max()) < 1e-8


def test_navier_stokes_exact_with_forcing():
    lv, cfg = _lv()
    rng = np.random.default_rng(2)
    tv = np.linspace(0.0, cfg.flow.period, 8)
    X = lv.sample_interior(2000, tv, rng).requires_grad_(True)
    uvw = lv._velocity_from_graph(X)
    vel = [uvw[:, i:i + 1] for i in range(3)]
    p = lv.pressure(X)
    f = lv.forcing(X)
    res = momentum_residual_nd(vel, p, X, rho=cfg.flow.density,
                              mu=cfg.flow.viscosity, forcing=f)
    assert max(float(r.detach().abs().max()) for r in res) < 1e-6


def test_shapes():
    lv, cfg = _lv()
    rng = np.random.default_rng(3)
    tv = np.linspace(0.0, cfg.flow.period, 4)
    X = lv.sample_interior(64, tv, rng)
    fields = lv.all_fields(X)
    for k in ("u", "v", "w", "p"):
        assert fields[k].shape == (64, 1)
    assert fields["vorticity"].shape == (64, 3)
    assert fields["forcing"].shape == (64, 3)
