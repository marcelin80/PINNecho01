"""Backward-compatible alias for the canonical NS residual module.

The full, dimension-general implementation now lives in
:mod:`pinnecho.physics.ns_residual` (named per the project spec). This module
re-exports the 2D convenience wrappers so existing imports keep working.
"""

from .ns_residual import (
    continuity_residual,
    momentum_residual,
    navier_stokes_residual,
)

__all__ = [
    "continuity_residual",
    "momentum_residual",
    "navier_stokes_residual",
]
