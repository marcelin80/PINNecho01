"""Synthetic IBAMR/IBFE-like left-ventricle FSI ground truth.

Stage 1 has no access to a real IBAMR/IBFE run, so this module *manufactures* a
ground-truth flow field that has the salient properties we need to test a PINN
reconstruction, while being cheap, analytic and fully self-consistent:

Geometry
    A deforming elliptical LV cavity. The long/short semi-axes vary over the
    cardiac cycle while the 2D cavity area is held constant, so the wall motion
    is genuinely divergence-free (2D incompressibility) -- no phantom sources.

Velocity  ``u = u_wall + u_vortex``
    * ``u_wall``  -- the affine "wall-following" velocity of the deforming
      ellipse. It is divergence-free (area preservation) and, at the wall,
      equals the true endocardial velocity.
    * ``u_vortex = curl(psi)`` with a stream function ``psi = (1 - rho^2)^2 * (...)``
      that vanishes together with its gradient on the wall (``rho`` is the
      reference/normalised radius, ``rho = 1`` on the wall). Hence the vortex
      contributes nothing at the wall and the total field satisfies *exact
      no-slip*: ``u = u_wall`` on the endocardium.

Pressure
    A smooth manufactured pressure field ``p`` (defined only up to a constant,
    as usual for incompressible flow).

FSI body forcing ``f``
    Defined as *exactly* the Navier-Stokes momentum residual of the manufactured
    ``(u, p)``. By construction ``(u, p, f)`` satisfy incompressible NS exactly.
    Physically ``f`` stands in for the net effect of myocardial active
    contraction / structural coupling. The baseline backbone ignores it
    (``f = 0``); the FSI-informed backbone is handed it.

Every derivative (velocity from the stream function, vorticity, forcing) is
taken with ``torch.autograd`` so there is no hand-derived algebra to get wrong.
"""

import math
from typing import Dict, Tuple

import numpy as np
import torch

from ..config import FlowConfig, GeometryConfig
from ..physics import operators as ops
from ..physics.navier_stokes import momentum_residual


class SyntheticLVFSI:
    """Analytic, autograd-differentiable synthetic LV FSI dataset."""

    def __init__(self, geometry: GeometryConfig, flow: FlowConfig,
                 velocity_scale: float = 15.0, dtype: torch.dtype = torch.float64):
        self.geom = geometry
        self.flow = flow
        self.dtype = dtype
        # Characteristic length and stream-function amplitude. ``velocity_scale``
        # is a dimensionless multiplier chosen so peak intracavitary speeds land
        # in a physiological ~0.3-0.7 m/s range for the default geometry.
        self.l_mean = 0.5 * (geometry.r0_x + geometry.r0_y)
        self.u_ref = geometry.r0_y / flow.period
        self.psi0 = velocity_scale * self.l_mean * self.u_ref

    # ------------------------------------------------------------------ #
    # Geometry / temporal shape
    # ------------------------------------------------------------------ #
    def _omega(self) -> float:
        return 2.0 * math.pi / self.flow.period

    def s_of_t(self, t):
        """Area-preserving shape modulation ``s(t)`` (dimensionless)."""
        return 1.0 + self.geom.strain_amplitude * self._sin(self._omega() * t)

    def semi_axes(self, t) -> Tuple:
        s = self.s_of_t(t)
        rx = self.geom.r0_x / s
        ry = self.geom.r0_y * s
        return rx, ry

    @staticmethod
    def _sin(x):
        return torch.sin(x) if isinstance(x, torch.Tensor) else math.sin(x)

    @staticmethod
    def _cos(x):
        return torch.cos(x) if isinstance(x, torch.Tensor) else math.cos(x)

    # ------------------------------------------------------------------ #
    # Closed-form primitive fields (differentiable in X)
    # ------------------------------------------------------------------ #
    def _normalised(self, X: torch.Tensor):
        """Return ``xi, eta, rho2`` in the moving reference frame."""
        x = X[:, 0:1]
        y = X[:, 1:2]
        t = X[:, 2:3]
        rx, ry = self.semi_axes(t)
        xi = (x - self.geom.center_x) / rx
        eta = (y - self.geom.center_y) / ry
        rho2 = xi * xi + eta * eta
        return xi, eta, rho2

    def stream_function(self, X: torch.Tensor) -> torch.Tensor:
        xi, eta, rho2 = self._normalised(X)
        t = X[:, 2:3]
        w = self._omega()
        env = (1.0 - rho2) ** 2  # zero value & gradient at wall (rho=1)
        swirl = self.flow.vortex_strength * self._cos(w * t)
        jet = self.flow.inflow_strength * eta * self._sin(w * t)
        return self.psi0 * env * (swirl + jet)

    def wall_following_velocity(self, X: torch.Tensor) -> torch.Tensor:
        """Affine, divergence-free velocity of the deforming ellipse."""
        x = X[:, 0:1]
        y = X[:, 1:2]
        t = X[:, 2:3]
        s = self.s_of_t(t)
        w = self._omega()
        s_dot = self.geom.strain_amplitude * w * self._cos(w * t)
        rate = s_dot / s  # d/dt ln s
        vwx = -rate * (x - self.geom.center_x)
        vwy = rate * (y - self.geom.center_y)
        return torch.cat([vwx, vwy], dim=1)

    def pressure(self, X: torch.Tensor) -> torch.Tensor:
        xi, eta, rho2 = self._normalised(X)
        t = X[:, 2:3]
        w = self._omega()
        return self.flow.pressure_amplitude * (
            self._cos(w * t) * (0.5 - 0.5 * rho2) + 0.3 * eta
        )

    # ------------------------------------------------------------------ #
    # Derived fields via autograd
    # ------------------------------------------------------------------ #
    def _velocity_from_graph(self, X: torch.Tensor) -> torch.Tensor:
        """``u = curl(psi) + u_wall`` keeping the autograd graph alive."""
        psi = self.stream_function(X)
        g = ops.grad(psi, X)
        u_vortex = g[:, 1:2]        # + d psi / d y
        v_vortex = -g[:, 0:1]       # - d psi / d x
        wall = self.wall_following_velocity(X)
        u = u_vortex + wall[:, 0:1]
        v = v_vortex + wall[:, 1:2]
        return torch.cat([u, v], dim=1)

    def velocity(self, X: torch.Tensor) -> torch.Tensor:
        return self._velocity_from_graph(X)

    def vorticity(self, X: torch.Tensor) -> torch.Tensor:
        uv = self._velocity_from_graph(X)
        return ops.curl_z(uv[:, 0:1], uv[:, 1:2], X)

    def forcing(self, X: torch.Tensor) -> torch.Tensor:
        """FSI body forcing ``f = rho Du/Dt + grad p - mu lap u`` (``(N, 2)``)."""
        uv = self._velocity_from_graph(X)
        p = self.pressure(X)
        fx, fy = momentum_residual(
            uv[:, 0:1], uv[:, 1:2], p, X,
            rho=self.flow.density, mu=self.flow.viscosity, forcing=None,
        )
        return torch.cat([fx, fy], dim=1)

    def all_fields(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Evaluate every ground-truth field at ``X`` (numpy-ready, detached).

        ``X`` may be a plain tensor; a differentiable clone is made internally.
        """
        Xg = X.detach().to(self.dtype).clone().requires_grad_(True)
        uv = self._velocity_from_graph(Xg)
        p = self.pressure(Xg)
        omega = ops.curl_z(uv[:, 0:1], uv[:, 1:2], Xg)
        fx, fy = momentum_residual(
            uv[:, 0:1], uv[:, 1:2], p, Xg,
            rho=self.flow.density, mu=self.flow.viscosity, forcing=None,
        )
        speed = torch.sqrt(uv[:, 0:1] ** 2 + uv[:, 1:2] ** 2)
        return {
            "u": uv[:, 0:1].detach(),
            "v": uv[:, 1:2].detach(),
            "speed": speed.detach(),
            "p": p.detach(),
            "vorticity": omega.detach(),
            "forcing": torch.cat([fx, fy], dim=1).detach(),
        }

    # ------------------------------------------------------------------ #
    # Sampling helpers (numpy driven, returns torch coordinates)
    # ------------------------------------------------------------------ #
    def _reference_disk(self, n: int, rng: np.random.Generator) -> np.ndarray:
        r = np.sqrt(rng.uniform(0.0, 1.0, size=n))  # uniform in area
        theta = rng.uniform(0.0, 2.0 * math.pi, size=n)
        return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    def _map_to_physical(self, ref_xy: np.ndarray, t: np.ndarray) -> np.ndarray:
        t_tensor = torch.as_tensor(t, dtype=self.dtype).reshape(-1, 1)
        rx, ry = self.semi_axes(t_tensor)
        rx = rx.numpy().reshape(-1)
        ry = ry.numpy().reshape(-1)
        x = self.geom.center_x + rx * ref_xy[:, 0]
        y = self.geom.center_y + ry * ref_xy[:, 1]
        return np.stack([x, y], axis=1)

    def sample_interior(self, n: int, t_values: np.ndarray,
                        rng: np.random.Generator) -> torch.Tensor:
        """Sample ``n`` interior points, times drawn from ``t_values``."""
        ref = self._reference_disk(n, rng)
        t = rng.choice(np.asarray(t_values, dtype=float), size=n)
        xy = self._map_to_physical(ref, t)
        X = np.concatenate([xy, t.reshape(-1, 1)], axis=1)
        return torch.as_tensor(X, dtype=self.dtype)

    def sample_wall(self, n: int, t_values: np.ndarray,
                    rng: np.random.Generator) -> Dict[str, torch.Tensor]:
        """Sample ``n`` wall points with their velocity and outward normal."""
        theta = rng.uniform(0.0, 2.0 * math.pi, size=n)
        ref = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        t = rng.choice(np.asarray(t_values, dtype=float), size=n)
        xy = self._map_to_physical(ref, t)
        X = torch.as_tensor(np.concatenate([xy, t.reshape(-1, 1)], axis=1),
                            dtype=self.dtype)
        wall_vel = self.wall_following_velocity(X).detach()
        # Outward normal from grad(rho^2) = (2 xi / rx, 2 eta / ry).
        t_tensor = X[:, 2:3]
        rx, ry = self.semi_axes(t_tensor)
        nx = (X[:, 0:1] - self.geom.center_x) / (rx * rx)
        ny = (X[:, 1:2] - self.geom.center_y) / (ry * ry)
        normal = torch.cat([nx, ny], dim=1)
        normal = normal / normal.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return {"X": X, "wall_velocity": wall_vel, "normal": normal}

    def grid(self, t: float, n: int = 80, pad: float = 1.0):
        """A masked evaluation grid at time ``t`` (for plotting / dense eval).

        Returns ``(X, inside_mask)`` where ``inside_mask`` marks grid points
        lying inside the cavity at time ``t``.
        """
        t_tensor = torch.tensor([[t]], dtype=self.dtype)
        rx, ry = self.semi_axes(t_tensor)
        rx = float(rx); ry = float(ry)
        xs = np.linspace(self.geom.center_x - pad * rx, self.geom.center_x + pad * rx, n)
        ys = np.linspace(self.geom.center_y - pad * ry, self.geom.center_y + pad * ry, n)
        gx, gy = np.meshgrid(xs, ys)
        xi = (gx - self.geom.center_x) / rx
        eta = (gy - self.geom.center_y) / ry
        inside = (xi ** 2 + eta ** 2) <= 1.0
        tt = np.full(gx.size, float(t))
        X = np.stack([gx.reshape(-1), gy.reshape(-1), tt], axis=1)
        return (torch.as_tensor(X, dtype=self.dtype),
                inside.reshape(-1))

    def field_statistics(self, n: int = 4000, seed: int = 0) -> Dict[str, float]:
        """Quick magnitude stats, handy for sanity checks and tuning."""
        rng = np.random.default_rng(seed)
        t_values = np.linspace(0.0, self.flow.period, 32)
        X = self.sample_interior(n, t_values, rng)
        f = self.all_fields(X)
        return {
            "speed_mean": float(f["speed"].mean()),
            "speed_max": float(f["speed"].max()),
            "pressure_std": float(f["p"].std()),
            "vorticity_absmax": float(f["vorticity"].abs().max()),
            "forcing_absmax": float(f["forcing"].abs().max()),
        }
