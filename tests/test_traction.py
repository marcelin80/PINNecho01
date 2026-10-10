"""Tests for the fluid Cauchy traction and traction-continuity BC (Model B ii)."""

import torch

from pinnecho.bc.boundary_conditions import (
    fluid_traction, fluid_traction_2d, traction_continuity_loss,
)

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


def _leaf4(cols):
    return torch.stack(cols, dim=1).clone().requires_grad_(True)


def test_fluid_traction_3d_pressure_only():
    """3D: u=v=w=0, p=const  ->  t = -p n (coords are (x,y,z,t))."""
    n = 40
    g = torch.Generator().manual_seed(3)
    X = _leaf4([torch.rand(n, generator=g) for _ in range(4)])  # x,y,z,t
    z = [torch.zeros(n, 1) for _ in range(3)]
    p = torch.full((n, 1), 2.0)
    nrm = torch.randn(n, 3)
    nrm = nrm / nrm.norm(dim=1, keepdim=True)
    t = fluid_traction(z, p, X, nrm, mu=0.1)
    assert t.shape == (n, 3)
    assert torch.allclose(t, -2.0 * nrm, atol=1e-9)


def test_fluid_traction_3d_simple_shear():
    """3D: u = gamma*y, v=w=0, p=0, n=(0,1,0)  ->  t = (mu*gamma, 0, 0)."""
    n = 32
    gamma, mu = 1.7, 0.02
    g = torch.Generator().manual_seed(4)
    X = _leaf4([torch.rand(n, generator=g) for _ in range(4)])
    y = X[:, 1:2]
    vel = [gamma * y, torch.zeros(n, 1), torch.zeros(n, 1)]
    p = torch.zeros(n, 1)
    nrm = torch.tensor([[0.0, 1.0, 0.0]]).repeat(n, 1)
    t = fluid_traction(vel, p, X, nrm, mu=mu)
    expected = torch.tensor([[mu * gamma, 0.0, 0.0]]).repeat(n, 1)
    assert torch.allclose(t, expected, atol=1e-9)


def test_traction_continuity_3d_zero_when_matched():
    n = 24
    mu = 0.04
    g = torch.Generator().manual_seed(5)
    X = _leaf4([torch.rand(n, generator=g) for _ in range(4)])
    xx, yy, zz = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    vel = [torch.sin(xx) * yy, -torch.cos(yy) * zz, xx * zz]
    p = xx + yy - zz
    nrm = torch.randn(n, 3)
    nrm = nrm / nrm.norm(dim=1, keepdim=True)
    target = fluid_traction(vel, p, X, nrm, mu).detach()
    loss = traction_continuity_loss(vel, p, X, nrm, target, mu)
    assert float(loss.detach()) < 1e-12
