"""Assemble the training dataset: measurements + collocation + wall points.

A :class:`ReconstructionDataset` bundles everything the trainer consumes:

* Doppler measurements (sparse, single-component) -- the data-fit target.
* Interior collocation points -- where the PDE residual is enforced.
* Precomputed FSI body forcing at those collocation points -- handed to the
  FSI-informed backbone (ignored by the baseline).
* Wall points with their prescribed (no-slip) velocity -- the boundary term.
* Reference scales for non-dimensionalisation.

The FSI forcing is precomputed here so the trainer never needs the analytic
ground truth at run time -- exactly as a real pipeline would consume a stored
IBAMR/IBFE forcing field.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from ..config import Config
from .synthetic_lv import SyntheticLVFSI
from .doppler import DopplerSampler


@dataclass
class Scales:
    """Reference scales used to non-dimensionalise the network I/O."""

    length: float
    time: float
    velocity: float
    pressure: float
    center_x: float = 0.0
    center_y: float = 0.0


@dataclass
class ReconstructionDataset:
    # Measurements
    meas_X: torch.Tensor
    meas_beam: torch.Tensor
    meas_doppler: torch.Tensor
    # Collocation (interior)
    col_X: torch.Tensor
    col_forcing: torch.Tensor
    # Wall
    wall_X: torch.Tensor
    wall_velocity: torch.Tensor
    wall_normal: torch.Tensor
    # Meta
    scales: Scales
    period: float
    density: float
    viscosity: float
    extras: Dict = field(default_factory=dict)

    def to(self, device: str) -> "ReconstructionDataset":
        for name in [
            "meas_X", "meas_beam", "meas_doppler", "col_X", "col_forcing",
            "wall_X", "wall_velocity", "wall_normal",
        ]:
            setattr(self, name, getattr(self, name).to(device))
        return self

    def as_float(self, dtype: torch.dtype) -> "ReconstructionDataset":
        for name in [
            "meas_X", "meas_beam", "meas_doppler", "col_X", "col_forcing",
            "wall_X", "wall_velocity", "wall_normal",
        ]:
            setattr(self, name, getattr(self, name).to(dtype))
        return self

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            meas_X=self.meas_X.cpu().numpy(),
            meas_beam=self.meas_beam.cpu().numpy(),
            meas_doppler=self.meas_doppler.cpu().numpy(),
            col_X=self.col_X.cpu().numpy(),
            col_forcing=self.col_forcing.cpu().numpy(),
            wall_X=self.wall_X.cpu().numpy(),
            wall_velocity=self.wall_velocity.cpu().numpy(),
            wall_normal=self.wall_normal.cpu().numpy(),
            scales=np.array([
                self.scales.length, self.scales.time, self.scales.velocity,
                self.scales.pressure, self.scales.center_x, self.scales.center_y,
            ]),
            meta=np.array([self.period, self.density, self.viscosity]),
        )

    @classmethod
    def load(cls, path: str | Path, dtype: torch.dtype = torch.float64) -> "ReconstructionDataset":
        data = np.load(path)
        s = data["scales"]
        meta = data["meta"]

        def t(name):
            return torch.as_tensor(data[name], dtype=dtype)

        return cls(
            meas_X=t("meas_X"), meas_beam=t("meas_beam"), meas_doppler=t("meas_doppler"),
            col_X=t("col_X"), col_forcing=t("col_forcing"),
            wall_X=t("wall_X"), wall_velocity=t("wall_velocity"), wall_normal=t("wall_normal"),
            scales=Scales(length=float(s[0]), time=float(s[1]), velocity=float(s[2]),
                          pressure=float(s[3]), center_x=float(s[4]), center_y=float(s[5])),
            period=float(meta[0]), density=float(meta[1]), viscosity=float(meta[2]),
        )


def build_dataset(cfg: Config, lv: SyntheticLVFSI | None = None,
                  dtype: torch.dtype = torch.float64) -> ReconstructionDataset:
    """Generate a full :class:`ReconstructionDataset` from a config."""
    if lv is None:
        lv = SyntheticLVFSI(cfg.geometry, cfg.flow, dtype=dtype)

    rng = np.random.default_rng(cfg.doppler.seed)

    # --- Doppler measurements -------------------------------------------------
    sampler = DopplerSampler(cfg.doppler)
    meas = sampler.sample(lv, rng)

    # --- Interior collocation + FSI forcing -----------------------------------
    col_rng = np.random.default_rng(cfg.seed + 1)
    t_values = np.linspace(0.0, cfg.flow.period, cfg.collocation.n_frames)
    col_X = lv.sample_interior(cfg.collocation.n_interior, t_values, col_rng)
    col_forcing = lv.forcing(col_X.clone().requires_grad_(True)).detach()

    # --- Wall points ----------------------------------------------------------
    wall = lv.sample_wall(cfg.collocation.n_wall, t_values, col_rng)

    # --- Reference scales -----------------------------------------------------
    stats = lv.field_statistics(seed=cfg.seed)
    velocity_scale = max(stats["speed_max"], lv.l_mean / cfg.flow.period)
    pressure_scale = max(cfg.flow.pressure_amplitude,
                         cfg.flow.density * velocity_scale ** 2)
    scales = Scales(
        length=lv.l_mean,
        time=cfg.flow.period,
        velocity=velocity_scale,
        pressure=pressure_scale,
        center_x=cfg.geometry.center_x,
        center_y=cfg.geometry.center_y,
    )

    ds = ReconstructionDataset(
        meas_X=meas["X"], meas_beam=meas["beam"], meas_doppler=meas["doppler"],
        col_X=col_X, col_forcing=col_forcing,
        wall_X=wall["X"], wall_velocity=wall["wall_velocity"], wall_normal=wall["normal"],
        scales=scales, period=cfg.flow.period,
        density=cfg.flow.density, viscosity=cfg.flow.viscosity,
        extras={"doppler_true": meas["doppler_true"], "noise_rms": meas["noise_rms"]},
    )
    return ds.as_float(dtype)
