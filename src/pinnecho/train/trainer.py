"""Training loop for the reconstruction PINN."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from ..config import Config
from ..data.dataset import ReconstructionDataset
from ..models.pinn import PINN
from ..utils.logging import get_logger
from . import losses as L

logger = get_logger(__name__)


@dataclass
class TrainState:
    history: Dict[str, List[float]] = field(default_factory=dict)

    def log(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().item()
            self.history.setdefault(k, []).append(float(v))


class Trainer:
    def __init__(self, cfg: Config, dataset: ReconstructionDataset,
                 model: Optional[PINN] = None, dtype: torch.dtype = torch.float64):
        self.cfg = cfg
        self.dtype = dtype
        self.device = cfg.device
        self.ds = dataset.to(cfg.device)
        self.model = (model or PINN(cfg.model, dataset.scales)).to(cfg.device).to(dtype)
        self.res_scales = L.ResidualScales.from_scales(dataset.scales, dataset.density)

        # FSI forcing is only injected for the FSI-informed backbone.
        if cfg.physics.backbone == "fsi_informed":
            self.forcing = self.ds.col_forcing * cfg.physics.forcing_scale
        elif cfg.physics.backbone == "baseline":
            self.forcing = None
        else:
            raise ValueError(f"Unknown backbone '{cfg.physics.backbone}'")

        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.train.lr)
        self.sched = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size=max(cfg.train.lr_decay_every, 1), gamma=cfg.train.lr_decay
        )
        self.state = TrainState()

    # ------------------------------------------------------------------ #
    def _batch_idx(self, n_total: int, batch: int, generator: torch.Generator):
        if batch <= 0 or batch >= n_total:
            return slice(None)
        return torch.randint(0, n_total, (batch,), generator=generator, device=self.device)

    def train(self) -> TrainState:
        cfg = self.cfg.train
        w = cfg.weights
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.cfg.seed)

        for it in range(1, cfg.iterations + 1):
            self.opt.zero_grad(set_to_none=True)

            # --- data (Doppler) --------------------------------------------
            l_data = L.data_loss(self.model, self.ds.meas_X, self.ds.meas_beam,
                                 self.ds.meas_doppler)

            # --- PDE (interior collocation) --------------------------------
            idx = self._batch_idx(self.ds.col_X.shape[0], cfg.batch_interior, gen)
            forcing_batch = None if self.forcing is None else self.forcing[idx]
            l_cont, l_mom = L.pde_loss(
                self.model, self.ds.col_X[idx], self.ds.density, self.ds.viscosity,
                self.res_scales, forcing=forcing_batch,
                enforce_continuity=self.cfg.physics.enforce_continuity,
            )

            # --- wall (no-slip) --------------------------------------------
            widx = self._batch_idx(self.ds.wall_X.shape[0], cfg.batch_wall, gen)
            l_wall = L.wall_loss(self.model, self.ds.wall_X[widx],
                                 self.ds.wall_velocity[widx], float(self.ds.scales.velocity))

            total = (w.data * l_data + w.continuity * l_cont
                     + w.momentum * l_mom + w.wall * l_wall)
            total.backward()
            self.opt.step()
            self.sched.step()

            if it % cfg.log_every == 0 or it == 1:
                vals = {k: float(v.detach()) for k, v in dict(
                    total=total, data=l_data, continuity=l_cont,
                    momentum=l_mom, wall=l_wall).items()}
                self.state.log(iter=it, lr=self.opt.param_groups[0]["lr"], **vals)
                logger.info(
                    "it=%d total=%.3e data=%.3e cont=%.3e mom=%.3e wall=%.3e",
                    it, vals["total"], vals["data"], vals["continuity"],
                    vals["momentum"], vals["wall"],
                )
        return self.state
