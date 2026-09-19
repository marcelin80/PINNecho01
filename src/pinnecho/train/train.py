"""Training entry point (spec-aligned) for the kinematic baseline (Model A).

This is a scaffold with real signatures and a functional Adam(+optional L-BFGS)
loop over the :class:`~pinnecho.train.composite_loss.CompositeLoss`. The data
assembly for the toy 2D case (and later the real IBFE problem) is factored behind
a ``build_batches`` callable so the loop itself is backbone- and data-agnostic.

Staging note
------------
Per the project plan, actually *running* Model A end-to-end on the toy case is the
next stage and is gated on explicit confirmation. Model B (FSI wall velocity /
traction continuity) must not be started until Model A validates. The Model B
switch here is a ``forcing="fsi"`` flag wired through to the loss, but the
corresponding BC helpers still raise ``NotImplementedError`` by design.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

from ..models.mlp_pinn import PINNNet
from .composite_loss import AnnealSchedule, CompositeLoss, LossWeights

BatchBuilder = Callable[[int], Dict[str, dict]]


@dataclass
class TrainConfig:
    steps: int = 10000
    lr: float = 2e-3
    lbfgs_iters: int = 0
    log_every: int = 500
    seed: int = 0
    grad_clip: Optional[float] = None


def build_model(model_cfg: dict) -> PINNNet:
    """Instantiate the shared network from a config dict (identical for A and B)."""
    return PINNNet(
        spatial_dim=model_cfg.get("spatial_dim", 2),
        width=model_cfg.get("width", 256),
        depth=model_cfg.get("depth", 8),
        activation=model_cfg.get("activation", "tanh"),
        fourier_features=model_cfg.get("fourier_features", 0),
        fourier_scales=tuple(model_cfg.get("fourier_scales", (1.0, 10.0, 100.0))),
        predict_scalar=model_cfg.get("predict_scalar", True),
        input_lows=model_cfg.get("input_lows"),
        input_highs=model_cfg.get("input_highs"),
    )


def train_model(
    model: PINNNet,
    loss_fn: CompositeLoss,
    build_batches: BatchBuilder,
    cfg: TrainConfig,
    logger: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Dict[str, list]:
    """Run Adam then optional L-BFGS polishing. Returns per-term loss history.

    ``build_batches(step)`` returns the ``batches`` dict expected by
    :class:`CompositeLoss` (see its docstring). It may resample collocation
    points each call.
    """
    torch.manual_seed(cfg.seed)
    history: Dict[str, list] = {}

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)

    for step in range(cfg.steps):
        batches = build_batches(step)
        opt.zero_grad(set_to_none=True)
        terms = loss_fn(model, batches, step=step)
        terms["total"].backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            row = {k: float(v.detach().item()) for k, v in terms.items()
                   if v.dim() == 0}
            for k, val in row.items():
                history.setdefault(k, []).append(val)
            if logger:
                logger(step, row)

    if cfg.lbfgs_iters > 0:
        _lbfgs_polish(model, loss_fn, build_batches, cfg)

    return history


def _lbfgs_polish(model, loss_fn, build_batches, cfg: TrainConfig) -> None:
    """L-BFGS refinement on a fixed batch (quasi-Newton polishing)."""
    batches = build_batches(cfg.steps)  # one fixed batch for a deterministic closure
    opt = torch.optim.LBFGS(
        model.parameters(), max_iter=cfg.lbfgs_iters,
        line_search_fn="strong_wolfe", tolerance_grad=1e-9,
    )

    def closure():
        opt.zero_grad(set_to_none=True)
        terms = loss_fn(model, batches, step=cfg.steps)
        terms["total"].backward()
        return terms["total"]

    opt.step(closure)


def main() -> None:  # pragma: no cover - CLI wiring
    """CLI entry point.

    TODO (next stage, gated on confirmation): parse ``configs/default.yaml``,
    build the toy 2D moving-boundary data source (or ``load_ibfe_output`` once
    available), construct the ``build_batches`` closure that draws collocation /
    Doppler / wall / valve / IC / periodic samples, and call :func:`train_model`.
    """
    raise NotImplementedError(
        "Model A end-to-end training on the toy case is the next stage. "
        "Wire configs/default.yaml + a toy build_batches closure here once "
        "confirmed to proceed."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
