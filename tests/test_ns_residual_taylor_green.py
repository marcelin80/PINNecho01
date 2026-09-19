"""Analytical unit tests for the incompressible Navier-Stokes residual.

These are the "catch autograd/implementation bugs early" tests required before
touching real IBFE data: for a known exact solution of the incompressible NS
equations, the residual computed by :mod:`pinnecho.physics.ns_residual` must be
~0 (to floating-point precision).

Two classic exact solutions are used:

* the 2D Taylor-Green vortex (unsteady, viscous-decaying), and
* 2D plane Poiseuille flow (steady, pressure-driven).
"""

import math

import torch

from pinnecho.physics.ns_residual import (
    continuity_residual,
    momentum_residual,
    navier_stokes_residual_nd,
)

torch.set_default_dtype(torch.float64)  # tight tolerances need float64


def _make_X(coords):
    """Stack a list of 1D coordinate tensors into a leaf tensor with grad."""
    X = torch.stack(coords, dim=1).clone().requires_grad_(True)
    return X


def test_taylor_green_vortex_residual_is_zero():
    """2D Taylor-Green vortex is an exact solution -> residual ~ 0.

        u =  cos(x) sin(y) F(t)
        v = -sin(x) cos(y) F(t)
        p = -(rho/4) (cos 2x + cos 2y) F(t)^2 ,   F(t) = exp(-2 nu t),  nu = mu/rho
    """
    rho, mu = 1.0, 0.1
    nu = mu / rho

    n = 512
    g = torch.Generator().manual_seed(0)
    x = (torch.rand(n, generator=g) * 2 - 1) * math.pi
    y = (torch.rand(n, generator=g) * 2 - 1) * math.pi
    t = torch.rand(n, generator=g) * 0.5
    X = _make_X([x, y, t])

    xx, yy, tt = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    F = torch.exp(-2.0 * nu * tt)
    u = torch.cos(xx) * torch.sin(yy) * F
    v = -torch.sin(xx) * torch.cos(yy) * F
    p = -(rho / 4.0) * (torch.cos(2 * xx) + torch.cos(2 * yy)) * F**2

    cont = continuity_residual(u, v, X)
    res_x, res_y = momentum_residual(u, v, p, X, rho, mu)

    assert torch.allclose(cont, torch.zeros_like(cont), atol=1e-8)
    assert torch.allclose(res_x, torch.zeros_like(res_x), atol=1e-8)
    assert torch.allclose(res_y, torch.zeros_like(res_y), atol=1e-8)


def test_poiseuille_flow_residual_is_zero():
    """Steady 2D plane Poiseuille flow is an exact solution -> residual ~ 0.

        u = U (1 - (y/H)^2),  v = 0,  dp/dx = -2 mu U / H^2  (=> p linear in x)
    """
    rho, mu = 1.05, 0.02
    U, H = 1.3, 0.75
    dpdx = -2.0 * mu * U / H**2

    n = 400
    g = torch.Generator().manual_seed(1)
    x = torch.rand(n, generator=g) * 3.0
    y = (torch.rand(n, generator=g) * 2 - 1) * H
    t = torch.rand(n, generator=g)
    X = _make_X([x, y, t])

    xx, yy = X[:, 0:1], X[:, 1:2]
    u = U * (1 - (yy / H) ** 2)
    v = torch.zeros_like(u)
    p = dpdx * xx

    cont = continuity_residual(u, v, X)
    res_x, res_y = momentum_residual(u, v, p, X, rho, mu)

    assert torch.allclose(cont, torch.zeros_like(cont), atol=1e-10)
    assert torch.allclose(res_x, torch.zeros_like(res_x), atol=1e-8)
    assert torch.allclose(res_y, torch.zeros_like(res_y), atol=1e-10)


def test_forcing_recovers_residual():
    """Supplying f equal to the residual of an arbitrary field zeroes it out.

    This checks the FSI-forcing plumbing: for any smooth (u, v, p), computing the
    unforced residual and feeding it back as ``forcing`` must give ~0.
    """
    rho, mu = 1060.0, 3.5e-3
    n = 256
    g = torch.Generator().manual_seed(2)
    x = torch.rand(n, generator=g)
    y = torch.rand(n, generator=g)
    t = torch.rand(n, generator=g)
    X = _make_X([x, y, t])

    xx, yy, tt = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    # An arbitrary (not divergence-free) field; forcing must still cancel momentum.
    u = torch.sin(3 * xx) * torch.cos(2 * yy) * (1 + tt)
    v = torch.cos(xx) * torch.sin(yy) * (1 + 0.5 * tt)
    p = torch.sin(xx + yy) * tt

    res_x0, res_y0 = momentum_residual(u, v, p, X, rho, mu)
    forcing = torch.cat([res_x0.detach(), res_y0.detach()], dim=1)
    res_x, res_y = momentum_residual(u, v, p, X, rho, mu, forcing=forcing)

    assert torch.allclose(res_x, torch.zeros_like(res_x), atol=1e-6)
    assert torch.allclose(res_y, torch.zeros_like(res_y), atol=1e-6)


def test_nd_wrapper_matches_2d():
    """The dimension-general wrapper must equal the 2D convenience functions."""
    rho, mu = 1.0, 0.05
    n = 128
    g = torch.Generator().manual_seed(3)
    x = torch.rand(n, generator=g)
    y = torch.rand(n, generator=g)
    t = torch.rand(n, generator=g)
    X = _make_X([x, y, t])
    xx, yy, tt = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    u = torch.sin(xx) * torch.cos(yy) * tt
    v = -torch.cos(xx) * torch.sin(yy) * tt
    p = torch.cos(xx) * torch.cos(yy)

    cont2, rx2, ry2 = (
        continuity_residual(u, v, X),
        *momentum_residual(u, v, p, X, rho, mu),
    )
    cont_nd, mom_nd = navier_stokes_residual_nd((u, v), p, X, rho, mu)
    assert torch.allclose(cont2, cont_nd, atol=1e-10)
    assert torch.allclose(rx2, mom_nd[0], atol=1e-10)
    assert torch.allclose(ry2, mom_nd[1], atol=1e-10)
