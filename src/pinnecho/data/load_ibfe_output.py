"""TODO STUB: loader for IBAMR/IBFE cardiac FSI simulation output.

This is intentionally *not implemented*. The real IBFE output format (VTK / Silo
/ HDF5 exodus, restart dumps, Lagrangian mesh files, etc.) will be supplied
separately; this module documents the exact tensor shapes the rest of PINNecho
expects so that wiring the real loader is a drop-in.

Everything is returned as ``torch.Tensor`` (float64), non-dimensionalised later
by the dataset layer. Coordinates follow the project convention: spatial columns
first, time last.

Expected outputs (2D; 3D adds a ``z`` column and a ``w`` velocity component)
-----------------------------------------------------------------------------
An :class:`IBFEFrames` bundle with, for ``T`` time frames sampled over one
cardiac cycle:

Fluid volume (ground-truth interior field, held out from training as GT):
    coords_fluid   : (N_f, 3)   columns (x, y, t)          [m, m, s]
    velocity_fluid : (N_f, 2)   columns (u, v)             [m/s]
    pressure_fluid : (N_f, 1)                              [Pa]
    forcing_fluid  : (N_f, 2)   columns (f_x, f_y)         [N/m^3]
        Net body force the structure exerts on the fluid (active contraction +
        elastic coupling) -- this is the term that separates Model B from A.

Fluid-structure interface (endocardial wall):
    coords_wall    : (N_w, 3)   columns (x, y, t)          [m, m, s]
    normals_wall   : (N_w, 2)   outward unit normals
    velocity_wall  : (N_w, 2)   FSI structural velocity at interface  [m/s]
        (Model B Dirichlet BC; differs subtly from the kinematic dc/dt used by A)
    traction_wall  : (N_w, 2)   structural traction vector t = sigma.n  [Pa]
        (Model B traction-continuity penalty target -- FUTURE stage)

Valves (mitral inflow / aortic outflow, Dirichlet):
    coords_mitral  : (N_m, 3);  velocity_mitral : (N_m, 2)  [m/s]
    coords_aortic  : (N_a, 3);  velocity_aortic : (N_a, 2)  [m/s]

Metadata:
    cycle_period   : float  [s]      (for periodicity loss)
    rho, mu        : float           (blood properties actually used by the FSI)

Shapes for 3D: append a ``z`` column to every ``coords_*`` (making them
``(*, 4)``) and a ``w`` component to every ``velocity_*`` / ``forcing_*`` /
``normals_*`` / ``traction_*`` (making them ``(*, 3)``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class IBFEFrames:
    """Container for one cardiac cycle of IBFE fields (see module docstring)."""

    coords_fluid: torch.Tensor
    velocity_fluid: torch.Tensor
    pressure_fluid: torch.Tensor
    forcing_fluid: torch.Tensor

    coords_wall: torch.Tensor
    normals_wall: torch.Tensor
    velocity_wall: torch.Tensor
    traction_wall: torch.Tensor

    coords_mitral: torch.Tensor
    velocity_mitral: torch.Tensor
    coords_aortic: torch.Tensor
    velocity_aortic: torch.Tensor

    cycle_period: float
    rho: float
    mu: float
    spatial_dim: int = 2


def load_ibfe_output(
    path: str,
    time_range: Optional[tuple] = None,
    subsample: Optional[int] = None,
    spatial_dim: int = 2,
    dtype: torch.dtype = torch.float64,
) -> IBFEFrames:
    """Load one cardiac cycle of IBAMR/IBFE output into :class:`IBFEFrames`.

    Dispatches by ``path``:

    * ``*.npz``                     -> :func:`~pinnecho.data.ibfe_io.load_ibfe_npz`
    * a directory / ``*.yaml|*.yml``-> :func:`~pinnecho.data.ibfe_io.load_ibfe_manifest`
      (a directory is expected to contain ``manifest.yaml``)

    See :mod:`pinnecho.data.ibfe_io` for the exact on-disk contract (NPZ keys,
    manifest schema, CSV column names). ``time_range`` (seconds) restricts to one
    cycle; ``subsample`` thins each dense point set to at most that many points
    per set (for tractable training on very large meshes).

    Parameters
    ----------
    path
        ``.npz`` bundle, ``manifest.yaml``, or a directory containing one.
    time_range
        Optional ``(t0, t1)`` window in seconds to restrict to one cycle.
    subsample
        Optional max number of points to keep per point set (uniform random).
    spatial_dim
        Sanity-checked against the loaded bundle's ``spatial_dim``.
    dtype
        Torch dtype for the returned tensors.
    """
    from .ibfe_io import load_ibfe_npz, load_ibfe_manifest

    p = Path(path)
    if p.suffix == ".npz":
        frames = load_ibfe_npz(p, dtype=dtype)
    elif p.suffix in (".yaml", ".yml"):
        frames = load_ibfe_manifest(p, dtype=dtype)
    elif p.is_dir():
        manifest = p / "manifest.yaml"
        if not manifest.exists():
            raise FileNotFoundError(
                f"{p} is a directory but contains no manifest.yaml. Provide a "
                "manifest (see pinnecho.data.ibfe_io.load_ibfe_manifest) or an "
                "*.npz bundle.")
        frames = load_ibfe_manifest(manifest, dtype=dtype)
    else:
        raise ValueError(
            f"Unrecognised IBFE export path '{path}'. Expected a .npz bundle, a "
            ".yaml manifest, or a directory containing manifest.yaml. For Stage-1 "
            "development use synthetic_ibfe_frames(...) / synthetic_ibfe_frames_3d(...).")

    if frames.spatial_dim != spatial_dim:
        raise ValueError(
            f"loaded spatial_dim={frames.spatial_dim} but caller asked for "
            f"spatial_dim={spatial_dim}")

    if time_range is not None:
        frames = _restrict_time(frames, time_range)
    if subsample is not None:
        frames = _subsample(frames, subsample)
    return frames


def _restrict_time(frames: IBFEFrames, time_range: tuple) -> IBFEFrames:
    """Keep only points whose time column lies in ``[t0, t1]``."""
    import dataclasses

    t0, t1 = float(time_range[0]), float(time_range[1])
    dim = frames.spatial_dim

    def _mask(coords):
        if coords.shape[0] == 0:
            return coords.new_ones(0, dtype=torch.bool)
        t = coords[:, dim]
        return (t >= t0) & (t <= t1)

    pairs = [("coords_fluid", ["velocity_fluid", "pressure_fluid", "forcing_fluid"]),
             ("coords_wall", ["normals_wall", "velocity_wall", "traction_wall"]),
             ("coords_mitral", ["velocity_mitral"]),
             ("coords_aortic", ["velocity_aortic"])]
    updates = {}
    for coord_key, deps in pairs:
        coords = getattr(frames, coord_key)
        m = _mask(coords)
        updates[coord_key] = coords[m]
        for d in deps:
            updates[d] = getattr(frames, d)[m]
    return dataclasses.replace(frames, **updates)


def _subsample(frames: IBFEFrames, max_points: int) -> IBFEFrames:
    """Uniformly thin every point set to at most ``max_points`` points."""
    import dataclasses
    import numpy as np

    rng = np.random.default_rng(0)

    def _idx(n):
        if n <= max_points:
            return torch.arange(n)
        return torch.as_tensor(rng.choice(n, size=max_points, replace=False))

    groups = [("coords_fluid", ["velocity_fluid", "pressure_fluid", "forcing_fluid"]),
              ("coords_wall", ["normals_wall", "velocity_wall", "traction_wall"]),
              ("coords_mitral", ["velocity_mitral"]),
              ("coords_aortic", ["velocity_aortic"])]
    updates = {}
    for coord_key, deps in groups:
        coords = getattr(frames, coord_key)
        idx = _idx(coords.shape[0])
        updates[coord_key] = coords[idx]
        for d in deps:
            updates[d] = getattr(frames, d)[idx]
    return dataclasses.replace(frames, **updates)


def synthetic_ibfe_frames(
    config,
    n_fluid: int = 4000,
    n_wall: int = 800,
    n_frames: int = 16,
    seed: int = 0,
    with_valve: bool = False,
    mitral_fraction: float = 0.35,
    dtype: torch.dtype = torch.float64,
) -> IBFEFrames:
    """Produce a 2D :class:`IBFEFrames` bundle from the synthetic-LV stand-in.

    This exercises the exact interface a real IBAMR/IBFE loader will return, so
    downstream code can be developed and tested against ``IBFEFrames`` now and
    the real ``load_ibfe_output`` can be dropped in later. The fluid volume,
    interface velocity/traction and body forcing come from the manufactured,
    NS-exact synthetic field.

    ``with_valve`` populates the mitral-valve arrays with the upper
    ``mitral_fraction`` of the wall as an *inflow-boundary plumbing stand-in*
    (coords + wall velocity), so the valve Dirichlet term and the residence-time
    ``c = 0`` inflow reinitialisation can be developed/tested before real valve
    data is available. The closed area-preserving synthetic cavity has no true
    net inflow, so this is a plumbing placeholder, not a physical mitral jet.
    """
    import numpy as np

    from .synthetic_lv import SyntheticLVFSI
    from ..bc.boundary_conditions import fluid_traction_2d

    lv = SyntheticLVFSI(config.geometry, config.flow, dtype=dtype)
    rng = np.random.default_rng(seed)
    t_values = np.linspace(0.0, config.flow.period, n_frames)

    # Fluid volume (held-out ground truth).
    coords_fluid = lv.sample_interior(n_fluid, t_values, rng).to(dtype)
    fields = lv.all_fields(coords_fluid)
    velocity_fluid = torch.cat([fields["u"], fields["v"]], dim=1)
    pressure_fluid = fields["p"]
    forcing_fluid = fields["forcing"]

    # Interface (endocardial wall).
    wall = lv.sample_wall(n_wall, t_values, rng)
    coords_wall = wall["X"].to(dtype)
    normals_wall = wall["normal"].to(dtype)
    velocity_wall = wall["wall_velocity"].to(dtype)  # FSI structural velocity
    Xg = coords_wall.clone().requires_grad_(True)
    uv = lv._velocity_from_graph(Xg)
    traction_wall = fluid_traction_2d(
        uv[:, 0:1], uv[:, 1:2], lv.pressure(Xg), Xg, normals_wall,
        config.flow.viscosity,
    ).detach()

    coords_mitral = torch.zeros(0, 3, dtype=dtype)
    velocity_mitral = torch.zeros(0, 2, dtype=dtype)
    if with_valve:
        coords_mitral, velocity_mitral = _mitral_patch(
            coords_wall, velocity_wall, spatial_dim=2, fraction=mitral_fraction)

    return IBFEFrames(
        coords_fluid=coords_fluid, velocity_fluid=velocity_fluid,
        pressure_fluid=pressure_fluid, forcing_fluid=forcing_fluid,
        coords_wall=coords_wall, normals_wall=normals_wall,
        velocity_wall=velocity_wall, traction_wall=traction_wall,
        coords_mitral=coords_mitral, velocity_mitral=velocity_mitral,
        coords_aortic=torch.zeros(0, 3, dtype=dtype),
        velocity_aortic=torch.zeros(0, 2, dtype=dtype),
        cycle_period=float(config.flow.period),
        rho=float(config.flow.density), mu=float(config.flow.viscosity),
        spatial_dim=2,
    )


def _mitral_patch(coords_wall: torch.Tensor, velocity_wall: torch.Tensor,
                  spatial_dim: int, fraction: float = 0.35):
    """Designate the upper ``fraction`` of wall points (max y) as a mitral patch.

    Returns ``(coords, velocity)`` for the selected wall subset -- an inflow
    plumbing stand-in (see :func:`synthetic_ibfe_frames`).
    """
    y = coords_wall[:, 1]
    if y.numel() == 0:
        return coords_wall.new_zeros(0, spatial_dim + 1), velocity_wall.new_zeros(0, spatial_dim)
    thresh = torch.quantile(y, 1.0 - float(fraction))
    mask = y >= thresh
    return coords_wall[mask].clone(), velocity_wall[mask].clone()


def synthetic_ibfe_frames_3d(
    config,
    n_fluid: int = 6000,
    n_wall: int = 1200,
    n_frames: int = 16,
    seed: int = 0,
    with_valve: bool = False,
    mitral_fraction: float = 0.35,
    dtype: torch.dtype = torch.float64,
) -> IBFEFrames:
    """Produce a 3D :class:`IBFEFrames` bundle from the volume-preserving
    ellipsoid ground truth (:class:`~pinnecho.data.synthetic_lv_3d.SyntheticLV3D`).

    Fields, wall velocity, wall traction (``sigma . n`` from the exact ``(u, p)``)
    and body forcing all come from the NS-exact 3D synthetic field, so the real
    3D ``load_ibfe_output`` path can be developed against the identical interface.
    ``with_valve`` behaves as in :func:`synthetic_ibfe_frames`.
    """
    import numpy as np

    from .synthetic_lv_3d import SyntheticLV3D
    from ..physics import operators as ops

    lv = SyntheticLV3D(config.geometry, config.flow, dtype=dtype)
    rng = np.random.default_rng(seed)
    t_values = np.linspace(0.0, config.flow.period, n_frames)

    coords_fluid = lv.sample_interior(n_fluid, t_values, rng).to(dtype)
    fields = lv.all_fields(coords_fluid)
    velocity_fluid = torch.cat([fields["u"], fields["v"], fields["w"]], dim=1)
    pressure_fluid = fields["p"]
    forcing_fluid = fields["forcing"]

    wall = lv.sample_wall(n_wall, t_values, rng)
    coords_wall = wall["X"].to(dtype)
    normals_wall = wall["normal"].to(dtype)
    velocity_wall = wall["wall_velocity"].to(dtype)

    # Fluid Cauchy traction t = (-p I + mu (grad u + grad u^T)) . n at the wall.
    Xg = coords_wall.clone().requires_grad_(True)
    uvw = lv._velocity_from_graph(Xg)
    p = lv.pressure(Xg)
    grads = [ops.grad(uvw[:, i:i + 1], Xg) for i in range(3)]
    traction_cols = []
    for i in range(3):
        # row i of sigma dotted with n
        ti = -p * normals_wall[:, i:i + 1]
        for j in range(3):
            sij = grads[i][:, j:j + 1] + grads[j][:, i:i + 1]
            ti = ti + config.flow.viscosity * sij * normals_wall[:, j:j + 1]
        traction_cols.append(ti)
    traction_wall = torch.cat(traction_cols, dim=1).detach()

    coords_mitral = torch.zeros(0, 4, dtype=dtype)
    velocity_mitral = torch.zeros(0, 3, dtype=dtype)
    if with_valve:
        coords_mitral, velocity_mitral = _mitral_patch(
            coords_wall, velocity_wall, spatial_dim=3, fraction=mitral_fraction)

    return IBFEFrames(
        coords_fluid=coords_fluid, velocity_fluid=velocity_fluid,
        pressure_fluid=pressure_fluid, forcing_fluid=forcing_fluid,
        coords_wall=coords_wall, normals_wall=normals_wall,
        velocity_wall=velocity_wall, traction_wall=traction_wall,
        coords_mitral=coords_mitral, velocity_mitral=velocity_mitral,
        coords_aortic=torch.zeros(0, 4, dtype=dtype),
        velocity_aortic=torch.zeros(0, 3, dtype=dtype),
        cycle_period=float(config.flow.period),
        rho=float(config.flow.density), mu=float(config.flow.viscosity),
        spatial_dim=3,
    )
