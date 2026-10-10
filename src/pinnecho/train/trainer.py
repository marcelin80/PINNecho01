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

    def _losses(self, physics_ramp: float, gen, full_batch: bool = False):
        """Compute the four loss terms. Returns ``(total, dict-of-terms)``."""
        cfg = self.cfg.train
        w = cfg.weights

        l_data = L.data_loss(self.model, self.ds.meas_X, self.ds.meas_beam,
                             self.ds.meas_doppler)

        if full_batch:
            idx = slice(None)
            widx = slice(None)
        else:
            idx = self._batch_idx(self.ds.col_X.shape[0], cfg.batch_interior, gen)
            widx = self._batch_idx(self.ds.wall_X.shape[0], cfg.batch_wall, gen)
        forcing_batch = None if self.forcing is None else self.forcing[idx]
        l_cont, l_mom = L.pde_loss(
            self.model, self.ds.col_X[idx], self.ds.density, self.ds.viscosity,
            self.res_scales, forcing=forcing_batch,
            enforce_continuity=self.cfg.physics.enforce_continuity,
        )
        l_wall = L.wall_loss(self.model, self.ds.wall_X[widx],
                             self.ds.wall_velocity[widx], float(self.ds.scales.velocity))

        total = (w.data * l_data + w.wall * l_wall
                 + physics_ramp * (w.continuity * l_cont + w.momentum * l_mom))
        return total, {"data": l_data, "continuity": l_cont, "momentum": l_mom, "wall": l_wall}

    def train(self) -> TrainState:
        cfg = self.cfg.train
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.cfg.seed)
        warmup_iters = max(int(cfg.physics_warmup_frac * cfg.iterations), 1)

        # --- Adam phase --------------------------------------------------------
        for it in range(1, cfg.iterations + 1):
            self.opt.zero_grad(set_to_none=True)
            physics_ramp = min(1.0, it / warmup_iters)
            total, terms = self._losses(physics_ramp, gen)
            total.backward()
            self.opt.step()
            self.sched.step()

            if it % cfg.log_every == 0 or it == 1:
                self._log(it, total, terms, self.opt.param_groups[0]["lr"])

        # --- L-BFGS polishing phase (full weights, full batch) -----------------
        if cfg.lbfgs_iters > 0:
            self._run_lbfgs(cfg.lbfgs_iters)

        return self.state

    def _run_lbfgs(self, n_iters: int) -> None:
        lbfgs = torch.optim.LBFGS(
            self.model.parameters(), max_iter=n_iters, history_size=50,
            line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14,
        )
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.cfg.seed + 999)
        step = {"n": 0}

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            total, terms = self._losses(1.0, gen, full_batch=True)
            total.backward()
            step["n"] += 1
            if step["n"] % 25 == 0 or step["n"] == 1:
                self._log(self.cfg.train.iterations + step["n"], total, terms, 0.0,
                          tag="lbfgs")
            return total

        logger.info("Starting L-BFGS polish (%d iters)", n_iters)
        lbfgs.step(closure)

    def _log(self, it, total, terms, lr, tag="adam") -> None:
        vals = {k: float(v.detach()) for k, v in
                dict(total=total, **terms).items()}
        self.state.log(iter=it, lr=lr, **vals)
        logger.info(
            "[%s] it=%d total=%.3e data=%.3e cont=%.3e mom=%.3e wall=%.3e",
            tag, it, vals["total"], vals["data"], vals["continuity"],
            vals["momentum"], vals["wall"],
        )
