"""Differential operators built on ``torch.autograd``.

Throughout PINNecho the spatio-temporal coordinate is a single tensor
``X`` of shape ``(N, 3)`` whose columns are ``(x, y, t)``. Keeping every
coordinate in one leaf tensor makes reverse-mode differentiation w.r.t. any
combination of space/time trivial and lets us build higher-order operators
(Laplacian, material derivative) by composition.

All helpers assume ``X.requires_grad_(True)`` has been called by the caller and
use ``create_graph=True`` so the resulting quantities remain differentiable
(required for second derivatives and for backprop through the PDE residual).
"""

import torch

X_INDEX = {"x": 0, "y": 1, "t": 2}


def grad(scalar: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Gradient of a scalar field ``scalar`` (shape ``(N, 1)``) w.r.t. ``X``.

    Returns a tensor of shape ``(N, 3)`` holding ``[d/dx, d/dy, d/dt]``.
    """
    if scalar.dim() == 1:
        scalar = scalar.unsqueeze(-1)
    (g,) = torch.autograd.grad(
        scalar,
        X,
        grad_outputs=torch.ones_like(scalar),
        create_graph=True,
        retain_graph=True,
    )
    return g


def d(scalar: torch.Tensor, X: torch.Tensor, wrt: str) -> torch.Tensor:
    """Single partial derivative ``d(scalar)/d(wrt)`` as an ``(N, 1)`` tensor."""
    return grad(scalar, X)[:, X_INDEX[wrt]: X_INDEX[wrt] + 1]


def gradient_xy(scalar: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Spatial gradient ``[d/dx, d/dy]`` of a scalar, shape ``(N, 2)``."""
    return grad(scalar, X)[:, 0:2]


def divergence(u: torch.Tensor, v: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """2D divergence ``du/dx + dv/dy`` of a velocity field, shape ``(N, 1)``."""
    return d(u, X, "x") + d(v, X, "y")


def curl_z(u: torch.Tensor, v: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Out-of-plane vorticity ``omega_z = dv/dx - du/dy``, shape ``(N, 1)``."""
    return d(v, X, "x") - d(u, X, "y")


def laplacian(scalar: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Spatial Laplacian ``d2/dx2 + d2/dy2`` of a scalar, shape ``(N, 1)``."""
    g = grad(scalar, X)
    d2x = d(g[:, 0:1], X, "x")
    d2y = d(g[:, 1:2], X, "y")
    return d2x + d2y


def material_derivative(
    scalar: torch.Tensor, u: torch.Tensor, v: torch.Tensor, X: torch.Tensor
) -> torch.Tensor:
    """Material derivative ``Ds/Dt = ds/dt + u ds/dx + v ds/dy`` (shape ``(N, 1)``)."""
    g = grad(scalar, X)
    return g[:, 2:3] + u * g[:, 0:1] + v * g[:, 1:2]
