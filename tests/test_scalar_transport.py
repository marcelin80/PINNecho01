"""Analytical unit tests for the residence-time scalar-transport residual.

For known exact solutions of ``dc/dt + (u.grad)c = D lap(c) + source`` the
residual must be ~0, exercising the time, advection, diffusion and source terms
independently.
"""

import torch

from pinnecho.physics.scalar_transport import scalar_transport_residual

torch.set_default_dtype(torch.float64)


def _make_X(coords):
    return torch.stack(coords, dim=1).clone().requires_grad_(True)


def test_time_and_source_term():
    """No flow, c = t satisfies dc/dt = 1 (residence clock)."""
    n = 200
    g = torch.Generator().manual_seed(0)
    X = _make_X([torch.rand(n, generator=g), torch.rand(n, generator=g),
                 torch.rand(n, generator=g)])
    tt = X[:, 2:3]
    c = tt.clone()
    zero = torch.zeros_like(tt)
    res = scalar_transport_residual(c, (zero, zero), X, diffusivity=1e-3, source=1.0)
    assert torch.allclose(res, torch.zeros_like(res), atol=1e-9)


def test_advection_term():
    """u = (1, 0), c = x balances advection against unit source."""
    n = 200
    g = torch.Generator().manual_seed(1)
    X = _make_X([torch.rand(n, generator=g), torch.rand(n, generator=g),
                 torch.rand(n, generator=g)])
    xx = X[:, 0:1]
    c = xx.clone()
    u = torch.ones_like(xx)
    v = torch.zeros_like(xx)
    res = scalar_transport_residual(c, (u, v), X, diffusivity=1e-6, source=1.0)
    assert torch.allclose(res, torch.zeros_like(res), atol=1e-9)


def test_diffusion_term():
    """No flow, c = x^2 -> lap(c) = 2; source = -2 D cancels the residual."""
    n = 200
    D = 0.5
    g = torch.Generator().manual_seed(2)
    X = _make_X([torch.rand(n, generator=g), torch.rand(n, generator=g),
                 torch.rand(n, generator=g)])
    xx = X[:, 0:1]
    c = xx**2
    zero = torch.zeros_like(xx)
    res = scalar_transport_residual(c, (zero, zero), X, diffusivity=D, source=-2 * D)
    assert torch.allclose(res, torch.zeros_like(res), atol=1e-9)
