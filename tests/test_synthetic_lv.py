"""The manufactured LV FSI field must be self-consistent."""

import numpy as np
import torch

from pinnecho.config import Config
from pinnecho.data.synthetic_lv import SyntheticLVFSI
from pinnecho.physics import operators as ops
from pinnecho.physics.navier_stokes import navier_stokes_residual


def _lv():
    c = Config()
    return SyntheticLVFSI(c.geometry, c.flow), c


def test_velocity_is_divergence_free():
    lv, c = _lv()
    rng = np.random.default_rng(0)
    tvals = np.linspace(0, c.flow.period, 12)
    X = lv.sample_interior(1500, tvals, rng).clone().requires_grad_(True)
    uv = lv.velocity(X)
    div = ops.divergence(uv[:, 0:1], uv[:, 1:2], X)
    assert float(div.detach().abs().max()) < 1e-9


def test_exact_no_slip_at_wall():
    lv, c = _lv()
    rng = np.random.default_rng(1)
    tvals = np.linspace(0, c.flow.period, 12)
    wall = lv.sample_wall(500, tvals, rng)
    Xw = wall["X"].clone().requires_grad_(True)
    uv = lv.velocity(Xw)
    err = (uv - wall["wall_velocity"]).detach().abs().max()
    assert float(err) < 1e-9


def test_forcing_makes_navier_stokes_exact():
    """(u, p, f) must satisfy incompressible NS to machine precision."""
    lv, c = _lv()
    rng = np.random.default_rng(2)
    tvals = np.linspace(0, c.flow.period, 12)
    X = lv.sample_interior(1200, tvals, rng).clone().requires_grad_(True)
    uv = lv.velocity(X)
    p = lv.pressure(X)
    f = lv.forcing(X)  # f defined as the NS residual of (u, p)
    cont, rx, ry = navier_stokes_residual(
        uv[:, 0:1], uv[:, 1:2], p, X,
        rho=c.flow.density, mu=c.flow.viscosity, forcing=f,
    )
    # With the correct forcing the momentum residual vanishes.
    assert float(rx.abs().max()) < 1e-6
    assert float(ry.abs().max()) < 1e-6
    assert float(cont.abs().max()) < 1e-9


def test_field_statistics_physiological():
    lv, _ = _lv()
    stats = lv.field_statistics(seed=0)
    # Peak intracavitary speed should land in a plausible LV range.
    assert 0.1 < stats["speed_max"] < 1.5
    assert stats["forcing_absmax"] > 0.0
