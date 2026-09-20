"""Validate an :class:`~pinnecho.data.load_ibfe_output.IBFEFrames` bundle.

Checks the *contract* a real IBAMR/IBFE export must satisfy before it can be fed
to training: array shapes and row-count consistency, finiteness, wall-normal
unit length, valve/inflow presence, time coverage, and physical-magnitude sanity
(velocity/pressure/forcing ranges). Returns a structured report; with
``strict=True`` it raises on any hard error.

Note: full PDE self-consistency (divergence-free, NS-exact forcing) requires a
*differentiable field*, not scattered samples, so it is verified in the synthetic
generator's own unit tests rather than here. This validator is what you run
against a real, sampled export.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class ValidationReport:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"IBFEFrames validation: {'OK' if self.ok else 'FAILED'}"]
        for e in self.errors:
            lines.append(f"  [error]   {e}")
        for w in self.warnings:
            lines.append(f"  [warning] {w}")
        if self.stats:
            lines.append("  stats:")
            for k, v in self.stats.items():
                lines.append(f"    {k:24s} {v}")
        return "\n".join(lines)


def _finite(name: str, t: torch.Tensor, errors: List[str]) -> None:
    if t.numel() and not torch.isfinite(t).all():
        errors.append(f"{name}: contains non-finite values (NaN/Inf)")


def validate_ibfe_frames(frames, strict: bool = False,
                         max_speed: float = 5.0) -> ValidationReport:
    """Validate ``frames`` against the IBFE contract.

    Parameters
    ----------
    frames
        The :class:`IBFEFrames` bundle to check.
    strict
        If True, raise ``ValueError`` when any hard error is found.
    max_speed
        Soft upper bound (m/s) for a physiological intracardiac speed; speeds
        above this raise a warning (not an error).
    """
    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, float] = {}

    dim = int(frames.spatial_dim)
    if dim not in (2, 3):
        errors.append(f"spatial_dim must be 2 or 3, got {dim}")
        report = ValidationReport(ok=False, errors=errors, warnings=warnings, stats=stats)
        if strict:
            raise ValueError(report.summary())
        return report

    # Expected column counts per array.
    expect = {
        "coords_fluid": dim + 1, "velocity_fluid": dim, "pressure_fluid": 1,
        "forcing_fluid": dim, "coords_wall": dim + 1, "normals_wall": dim,
        "velocity_wall": dim, "traction_wall": dim, "coords_mitral": dim + 1,
        "velocity_mitral": dim, "coords_aortic": dim + 1, "velocity_aortic": dim,
    }
    for name, ncol in expect.items():
        arr = getattr(frames, name)
        if arr.dim() != 2 or arr.shape[1] != ncol:
            errors.append(f"{name}: expected shape (*, {ncol}), got {tuple(arr.shape)}")
        _finite(name, arr, errors)

    # Row-count consistency within each group.
    groups = {
        "fluid": ["coords_fluid", "velocity_fluid", "pressure_fluid", "forcing_fluid"],
        "wall": ["coords_wall", "normals_wall", "velocity_wall", "traction_wall"],
        "mitral": ["coords_mitral", "velocity_mitral"],
        "aortic": ["coords_aortic", "velocity_aortic"],
    }
    for g, names in groups.items():
        rows = {getattr(frames, n).shape[0] for n in names}
        if len(rows) > 1:
            errors.append(f"{g}: inconsistent row counts across {names}: {rows}")

    n_fluid = frames.coords_fluid.shape[0]
    n_wall = frames.coords_wall.shape[0]
    stats["n_fluid"] = n_fluid
    stats["n_wall"] = n_wall
    stats["n_mitral"] = frames.coords_mitral.shape[0]
    stats["n_aortic"] = frames.coords_aortic.shape[0]

    if n_fluid == 0:
        errors.append("coords_fluid is empty; no interior ground truth to train/evaluate")
    if n_wall == 0:
        errors.append("coords_wall is empty; no boundary condition to enforce")

    # Wall-normal unit length.
    if n_wall and frames.normals_wall.shape[1] == dim:
        nn = frames.normals_wall.norm(dim=1)
        stats["normal_norm_mean"] = float(nn.mean())
        if float((nn - 1.0).abs().max()) > 1e-3:
            warnings.append("normals_wall are not unit length (max |‖n‖-1| > 1e-3)")

    # Time coverage.
    if n_fluid:
        t = frames.coords_fluid[:, dim]
        stats["t_min"] = float(t.min())
        stats["t_max"] = float(t.max())
        stats["n_time_frames"] = int(torch.unique(torch.round(t * 1e6)).numel())
        if stats["t_max"] > frames.cycle_period + 1e-6:
            warnings.append(
                f"fluid times exceed cycle_period ({stats['t_max']:.3f} > "
                f"{frames.cycle_period:.3f}); pass time_range to restrict to one cycle")

    # Magnitude sanity.
    if n_fluid:
        speed = frames.velocity_fluid.norm(dim=1)
        stats["speed_max"] = float(speed.max())
        stats["speed_mean"] = float(speed.mean())
        stats["pressure_min"] = float(frames.pressure_fluid.min())
        stats["pressure_max"] = float(frames.pressure_fluid.max())
        stats["forcing_absmax"] = float(frames.forcing_fluid.abs().max())
        if stats["speed_max"] > max_speed:
            warnings.append(
                f"peak speed {stats['speed_max']:.2f} m/s exceeds {max_speed} m/s "
                "-- check units (expected SI m/s)")
        if stats["forcing_absmax"] == 0.0:
            warnings.append("forcing_fluid is all zeros -- Model B reduces to Model A")

    if frames.rho <= 0 or frames.mu <= 0:
        errors.append(f"non-physical rho={frames.rho} / mu={frames.mu}")
    if frames.cycle_period <= 0:
        errors.append(f"non-physical cycle_period={frames.cycle_period}")

    report = ValidationReport(ok=(len(errors) == 0), errors=errors,
                              warnings=warnings, stats=stats)
    if strict and errors:
        raise ValueError(report.summary())
    return report
