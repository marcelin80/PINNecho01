"""Autograd differential operators vs. closed-form derivatives."""

import math

import torch

from pinnecho.physics import operators as ops


def _coords(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 3, generator=g, dtype=torch.float64) * 2 - 1
    return X.clone().requires_grad_(True)


def test_gradient_matches_analytic():
    X = _coords()
    x, y, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    f = torch.sin(x) * torch.cos(y) + t ** 2
    g = ops.grad(f, X)
    assert torch.allclose(g[:, 0:1], torch.cos(x) * torch.cos(y), atol=1e-10)
    assert torch.allclose(g[:, 1:2], -torch.sin(x) * torch.sin(y), atol=1e-10)
    assert torch.allclose(g[:, 2:3], 2 * t, atol=1e-10)


def test_laplacian_matches_analytic():
    X = _coords()
    x, y = X[:, 0:1], X[:, 1:2]
    f = torch.sin(x) * torch.cos(y)
    lap = ops.laplacian(f, X)
    # d2/dx2 + d2/dy2 of sin(x)cos(y) = -2 sin(x) cos(y)
    assert torch.allclose(lap, -2.0 * torch.sin(x) * torch.cos(y), atol=1e-9)


def test_divergence_and_curl():
    X = _coords()
    x, y = X[:, 0:1], X[:, 1:2]
    # u = (y, -x): divergence 0, vorticity (dv/dx - du/dy) = -1 - 1 = -2
    u = y
    v = -x
    div = ops.divergence(u, v, X)
    curl = ops.curl_z(u, v, X)
    assert torch.allclose(div, torch.zeros_like(div), atol=1e-10)
    assert torch.allclose(curl, -2.0 * torch.ones_like(curl), atol=1e-10)


def test_material_derivative():
    X = _coords()
    x, y, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    s = x * y + torch.sin(t)
    u = torch.ones_like(x)
    v = 2.0 * torch.ones_like(x)
    md = ops.material_derivative(s, u, v, X)
    # ds/dt + u ds/dx + v ds/dy = cos(t) + y + 2 x
    assert torch.allclose(md, torch.cos(t) + y + 2.0 * x, atol=1e-10)
