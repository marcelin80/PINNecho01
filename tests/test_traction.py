"""Tests for the fluid Cauchy traction and traction-continuity BC (Model B ii)."""

import torch

from pinnecho.bc.boundary_conditions import fluid_traction_2d, traction_continuity_loss

torch.set_default_dtype(torch.float64)


def _leaf(coords):
    return torch.stack(coords, dim=1).clone().requires_grad_(True)


def test_pressure_only_traction_is_minus_p_n():
    """u = v = 0, p = const  ->  t = sigma.n = -p n."""
    n = 50
    g = torch.Generator().manual_seed(0)
    X = _leaf([torch.rand(n, generator=g), torch.rand(n, generator=g),
               torch.rand(n, generator=g)])
    u = torch.zeros(n, 1)
    v = torch.zeros(n, 1)
    p = torch.full((n, 1), 3.0)
    normals = torch.randn(n, 2)
    normals = normals / normals.norm(dim=1, keepdim=True)
    t = fluid_traction_2d(u, v, p, X, normals, mu=0.1)
    assert torch.allclose(t, -3.0 * normals, atol=1e-9)


def test_simple_shear_traction():
    """u = gamma*y, v = 0, p = 0, n = (0,1)  ->  t = (mu*gamma, 0)."""
    n = 40
    gamma, mu = 2.5, 0.03
    g = torch.Generator().manual_seed(1)
    X = _leaf([torch.rand(n, generator=g), torch.rand(n, generator=g),
               torch.rand(n, generator=g)])
    y = X[:, 1:2]
    u = gamma * y
    v = torch.zeros(n, 1)
    p = torch.zeros(n, 1)
    normals = torch.tensor([[0.0, 1.0]]).repeat(n, 1)
    t = fluid_traction_2d(u, v, p, X, normals, mu=mu)
    expected = torch.tensor([[mu * gamma, 0.0]]).repeat(n, 1)
    assert torch.allclose(t, expected, atol=1e-9)


def test_traction_continuity_zero_when_matched():
    """Residual is ~0 when the structure traction equals the fluid traction."""
    n = 30
    mu = 0.05
    g = torch.Generator().manual_seed(2)
    X = _leaf([torch.rand(n, generator=g), torch.rand(n, generator=g),
               torch.rand(n, generator=g)])
    xx, yy = X[:, 0:1], X[:, 1:2]
    u = torch.sin(xx) * yy
    v = -torch.cos(yy) * xx
    p = xx + yy
    normals = torch.randn(n, 2)
    normals = normals / normals.norm(dim=1, keepdim=True)
    target = fluid_traction_2d(u, v, p, X, normals, mu).detach()
    loss = traction_continuity_loss([u, v], p, X, normals, target, mu)
    assert float(loss.detach()) < 1e-12
