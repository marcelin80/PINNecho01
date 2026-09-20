"""Synthetic 3D left-ventricle FSI ground truth (analytic, autograd-consistent).

The 3D analogue of :class:`~pinnecho.data.synthetic_lv.SyntheticLVFSI`. A
deforming *ellipsoid* cavity whose three semi-axes vary over the cardiac cycle
while the cavity **volume is preserved** (3D incompressibility of the wall
motion -- no phantom sources/sinks).

Velocity ``u = u_wall + u_vortex``
    * ``u_wall``   -- affine wall-following velocity ``u_wall,i = (d/dt ln a_i)(x_i - c_i)``.
      Its divergence is ``sum_i d/dt ln a_i = d/dt ln(prod a_i) = 0`` because the
      volume ``prod a_i`` is constant, so it is exactly divergence-free and equals
      the true endocardial velocity on the wall.
    * ``u_vortex = curl(A)`` with vector potential ``A = env(x,t) * a(t)`` where
      ``env = (1 - rho^2)^2`` vanishes together with its gradient at the wall
      (``rho = 1``) and ``a(t)`` is spatially uniform. Then
      ``u_vortex = grad(env) x a`` is divergence-free (a curl) and *zero at the
      wall* (both ``env`` and ``grad(env)`` vanish there). Hence the total field
      satisfies exact no-slip ``u = u_wall`` on the endocardium.

Pressure ``p`` is a smooth manufactured field; the FSI body forcing ``f`` is
defined as the exact 3D Navier-Stokes momentum residual of ``(u, p)`` so that
``(u, p, f)`` satisfy incompressible NS exactly. Every derivative is taken with
``torch.autograd``.
"""

import math
from typing import Dict, Tuple

import numpy as np
import torch

from ..config import FlowConfig, GeometryConfig
from ..physics import operators as ops
from ..physics.ns_residual import momentum_residual_nd


class SyntheticLV3D:
    """Analytic, autograd-differentiable synthetic 3D LV FSI dataset."""

    spatial_dim = 3

    def __init__(self, geometry: GeometryConfig, flow: FlowConfig,
                 velocity_scale: float = 8.0, dtype: torch.dtype = torch.float64):
        self.geom = geometry
        self.flow = flow
        self.dtype = dtype
        self.l_mean = (geometry.r0_x + geometry.r0_y + geometry.r0_z) / 3.0
        self.u_ref = geometry.r0_z / flow.period
        # Vector-potential amplitude: u ~ grad(env) x a ~ a / L, so a ~ U * L.
        self.a0 = velocity_scale * self.l_mean * self.u_ref

    # ------------------------------------------------------------------ #
    # Geometry / temporal shape
    # ------------------------------------------------------------------ #
    def _omega(self) -> float:
        return 2.0 * math.pi / self.flow.period

    def s_of_t(self, t):
        return 1.0 + self.geom.strain_amplitude * self._sin(self._omega() * t)

    def semi_axes(self, t) -> Tuple:
        """Volume-preserving semi-axes: ``rx*ry*rz = r0x*r0y*r0z`` for all ``t``.

        The long (apex-base, z) axis lengthens by ``s``; the two short axes each
        shorten by ``s^{-1/2}`` so the product is invariant.
        """
        s = self.s_of_t(t)
        inv_sqrt = 1.0 / self._sqrt(s)
        rx = self.geom.r0_x * inv_sqrt
        ry = self.geom.r0_y * inv_sqrt
        rz = self.geom.r0_z * s
        return rx, ry, rz

    @staticmethod
    def _sin(x):
        return torch.sin(x) if isinstance(x, torch.Tensor) else math.sin(x)

    @staticmethod
    def _cos(x):
        return torch.cos(x) if isinstance(x, torch.Tensor) else math.cos(x)

    @staticmethod
    def _sqrt(x):
        return torch.sqrt(x) if isinstance(x, torch.Tensor) else math.sqrt(x)

    # ------------------------------------------------------------------ #
    # Primitive fields (differentiable in X = (x, y, z, t))
    # ------------------------------------------------------------------ #
    def _normalised(self, X: torch.Tensor):
        x, y, z, t = X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 3:4]
        rx, ry, rz = self.semi_axes(t)
        xi = (x - self.geom.center_x) / rx
        eta = (y - self.geom.center_y) / ry
        zeta = (z - self.geom.center_z) / rz
        rho2 = xi * xi + eta * eta + zeta * zeta
        return xi, eta, zeta, rho2

    def _amplitude_vector(self, t):
        """Spatially uniform vector-potential amplitude ``a(t)`` (per column)."""
        w = self._omega()
        ax = self.a0 * self.flow.vortex_strength * self._cos(w * t)
        ay = self.a0 * self.flow.inflow_strength * self._sin(w * t)
        az = self.a0 * 0.5 * self.flow.vortex_strength * self._sin(w * t)
        return ax, ay, az

    def wall_following_velocity(self, X: torch.Tensor) -> torch.Tensor:
        x, y, z, t = X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 3:4]
        s = self.s_of_t(t)
        w = self._omega()
        s_dot = self.geom.strain_amplitude * w * self._cos(w * t)
        rate = s_dot / s  # d/dt ln s
        # ln rx = ln r0x - 0.5 ln s -> rate_x = -0.5 rate; rate_z = +rate.
        vx = -0.5 * rate * (x - self.geom.center_x)
        vy = -0.5 * rate * (y - self.geom.center_y)
        vz = rate * (z - self.geom.center_z)
        return torch.cat([vx, vy, vz], dim=1)

    def _env(self, X: torch.Tensor) -> torch.Tensor:
        _, _, _, rho2 = self._normalised(X)
        return (1.0 - rho2) ** 2

    def vortex_velocity(self, X: torch.Tensor) -> torch.Tensor:
        """``u_vortex = grad(env) x a(t)`` -- divergence-free, zero at the wall."""
        env = self._env(X)
        genv = ops.grad(env, X)  # (N, 4): d/dx, d/dy, d/dz, d/dt
        ex, ey, ez = genv[:, 0:1], genv[:, 1:2], genv[:, 2:3]
        ax, ay, az = self._amplitude_vector(X[:, 3:4])
        # cross product grad(env) x a
        ux = ey * az - ez * ay
        uy = ez * ax - ex * az
        uz = ex * ay - ey * ax
        return torch.cat([ux, uy, uz], dim=1)

    def pressure(self, X: torch.Tensor) -> torch.Tensor:
        xi, eta, zeta, rho2 = self._normalised(X)
        t = X[:, 3:4]
        w = self._omega()
        return self.flow.pressure_amplitude * (
            self._cos(w * t) * (0.5 - 0.5 * rho2) + 0.3 * zeta
        )

    # ------------------------------------------------------------------ #
    # Derived fields via autograd
    # ------------------------------------------------------------------ #
    def _velocity_from_graph(self, X: torch.Tensor) -> torch.Tensor:
        return self.vortex_velocity(X) + self.wall_following_velocity(X)

    def velocity(self, X: torch.Tensor) -> torch.Tensor:
        return self._velocity_from_graph(X)

    def divergence(self, X: torch.Tensor) -> torch.Tensor:
        uvw = self._velocity_from_graph(X)
        d = torch.zeros_like(uvw[:, 0:1])
        for i in range(3):
            d = d + ops.grad(uvw[:, i:i + 1], X)[:, i:i + 1]
        return d

    def vorticity(self, X: torch.Tensor) -> torch.Tensor:
        """3D vorticity vector ``omega = curl(u)`` (shape ``(N, 3)``)."""
        uvw = self._velocity_from_graph(X)
        u, v, wv = uvw[:, 0:1], uvw[:, 1:2], uvw[:, 2:3]
        gu, gv, gw = ops.grad(u, X), ops.grad(v, X), ops.grad(wv, X)
        wx = gw[:, 1:2] - gv[:, 2:3]
        wy = gu[:, 2:3] - gw[:, 0:1]
        wz = gv[:, 0:1] - gu[:, 1:2]
        return torch.cat([wx, wy, wz], dim=1)

    def forcing(self, X: torch.Tensor) -> torch.Tensor:
        """FSI body forcing ``f = rho Du/Dt + grad p - mu lap u`` (``(N, 3)``)."""
        uvw = self._velocity_from_graph(X)
        vel = [uvw[:, i:i + 1] for i in range(3)]
        p = self.pressure(X)
        res = momentum_residual_nd(vel, p, X, rho=self.flow.density,
                                   mu=self.flow.viscosity, forcing=None)
        return torch.cat(res, dim=1)

    def all_fields(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        Xg = X.detach().to(self.dtype).clone().requires_grad_(True)
        uvw = self._velocity_from_graph(Xg)
        p = self.pressure(Xg)
        omega = self.vorticity(Xg)
        f = self.forcing(Xg)
        speed = torch.linalg.norm(uvw, dim=1, keepdim=True)
        return {
            "u": uvw[:, 0:1].detach(), "v": uvw[:, 1:2].detach(),
            "w": uvw[:, 2:3].detach(), "speed": speed.detach(),
            "p": p.detach(), "vorticity": omega.detach(),
            "forcing": f.detach(),
        }

    # ------------------------------------------------------------------ #
    # Sampling helpers
    # ------------------------------------------------------------------ #
    def _reference_ball(self, n: int, rng: np.random.Generator) -> np.ndarray:
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        r = rng.uniform(0.0, 1.0, size=(n, 1)) ** (1.0 / 3.0)  # uniform in volume
        return v * r

    def _reference_sphere(self, n: int, rng: np.random.Generator) -> np.ndarray:
        v = rng.normal(size=(n, 3))
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    def _map_to_physical(self, ref: np.ndarray, t: np.ndarray) -> np.ndarray:
        t_tensor = torch.as_tensor(t, dtype=self.dtype).reshape(-1, 1)
        rx, ry, rz = self.semi_axes(t_tensor)
        rx = rx.numpy().reshape(-1); ry = ry.numpy().reshape(-1); rz = rz.numpy().reshape(-1)
        x = self.geom.center_x + rx * ref[:, 0]
        y = self.geom.center_y + ry * ref[:, 1]
        z = self.geom.center_z + rz * ref[:, 2]
        return np.stack([x, y, z], axis=1)

    def sample_interior(self, n: int, t_values: np.ndarray,
                        rng: np.random.Generator) -> torch.Tensor:
        ref = self._reference_ball(n, rng)
        t = rng.choice(np.asarray(t_values, dtype=float), size=n)
        xyz = self._map_to_physical(ref, t)
        X = np.concatenate([xyz, t.reshape(-1, 1)], axis=1)
        return torch.as_tensor(X, dtype=self.dtype)

    def sample_wall(self, n: int, t_values: np.ndarray,
                    rng: np.random.Generator) -> Dict[str, torch.Tensor]:
        ref = self._reference_sphere(n, rng)
        t = rng.choice(np.asarray(t_values, dtype=float), size=n)
        xyz = self._map_to_physical(ref, t)
        X = torch.as_tensor(np.concatenate([xyz, t.reshape(-1, 1)], axis=1),
                            dtype=self.dtype)
        wall_vel = self.wall_following_velocity(X).detach()
        t_tensor = X[:, 3:4]
        rx, ry, rz = self.semi_axes(t_tensor)
        nx = (X[:, 0:1] - self.geom.center_x) / (rx * rx)
        ny = (X[:, 1:2] - self.geom.center_y) / (ry * ry)
        nz = (X[:, 2:3] - self.geom.center_z) / (rz * rz)
        normal = torch.cat([nx, ny, nz], dim=1)
        normal = normal / normal.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return {"X": X, "wall_velocity": wall_vel, "normal": normal}

    def field_statistics(self, n: int = 4000, seed: int = 0) -> Dict[str, float]:
        rng = np.random.default_rng(seed)
        t_values = np.linspace(0.0, self.flow.period, 16)
        X = self.sample_interior(n, t_values, rng)
        f = self.all_fields(X)
        speed = f["speed"].reshape(-1)
        return {
            "speed_mean": float(speed.mean()),
            "speed_max": float(speed.max()),
            "pressure_std": float(f["p"].std()),
            "vorticity_max": float(torch.linalg.norm(f["vorticity"], dim=1).max()),
        }
