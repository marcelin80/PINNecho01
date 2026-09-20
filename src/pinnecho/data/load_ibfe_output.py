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
) -> IBFEFrames:
    """Load one cardiac cycle of IBAMR/IBFE output into :class:`IBFEFrames`.

    Parameters
    ----------
    path
        Path to the IBFE output (directory or file); format TBD by the user.
    time_range
        Optional ``(t0, t1)`` window in seconds to restrict to one cycle.
    subsample
        Optional stride/count to thin dense meshes for tractable training.
    spatial_dim
        2 or 3.

    Returns
    -------
    IBFEFrames
        Tensors with the shapes documented at module level.

    Notes
    -----
    TODO: implement once the concrete IBFE export format is provided. The
    implementation must (1) read the Eulerian fluid grid and Lagrangian
    structure mesh, (2) interpolate structure velocity/traction onto the
    fluid-structure interface, (3) compute or read the net fluid body force,
    and (4) non-dimensionalise consistently with :mod:`pinnecho.data.dataset`.
    """
    raise NotImplementedError(
        "load_ibfe_output is a documented stub. Supply the IBFE export format "
        "and implement per the shapes in this module's docstring. For Stage-1 "
        "development use synthetic_ibfe_frames(...) below, which produces the "
        "same IBFEFrames bundle from the manufactured synthetic-LV field."
    )


def synthetic_ibfe_frames(
    config,
    n_fluid: int = 4000,
    n_wall: int = 800,
    n_frames: int = 16,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> IBFEFrames:
    """Produce an :class:`IBFEFrames` bundle from the synthetic-LV stand-in.

    This exercises the exact interface a real IBAMR/IBFE loader will return, so
    downstream code can be developed and tested against ``IBFEFrames`` now and
    the real ``load_ibfe_output`` can be dropped in later. The fluid volume,
    interface velocity/traction and body forcing come from the manufactured,
    NS-exact synthetic field.

    Valve arrays are returned empty (the closed area-preserving synthetic cavity
    has no mitral/aortic openings); real IBFE output will populate them.
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

    empty3 = torch.zeros(0, 3, dtype=dtype)
    empty2 = torch.zeros(0, 2, dtype=dtype)
    return IBFEFrames(
        coords_fluid=coords_fluid, velocity_fluid=velocity_fluid,
        pressure_fluid=pressure_fluid, forcing_fluid=forcing_fluid,
        coords_wall=coords_wall, normals_wall=normals_wall,
        velocity_wall=velocity_wall, traction_wall=traction_wall,
        coords_mitral=empty3, velocity_mitral=empty2,
        coords_aortic=empty3, velocity_aortic=empty2,
        cycle_period=float(config.flow.period),
        rho=float(config.flow.density), mu=float(config.flow.viscosity),
        spatial_dim=2,
    )
