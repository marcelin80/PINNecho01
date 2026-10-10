"""High-level orchestration reused by the CLI scripts and tests.

Keeping generate / train / evaluate / compare logic here (rather than in the
argparse entry points) makes the pipeline importable and testable, and keeps the
CLI thin.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from .config import Config, load_config
from .data.dataset import ReconstructionDataset, build_dataset, Scales
from .data.synthetic_lv import SyntheticLVFSI
from .eval.metrics import evaluate_fields, evaluate_wss
from .eval.residence_time import evaluate_residence_time
from .models.pinn import PINN
from .train.trainer import Trainer, TrainState
from .utils.logging import get_logger
from .utils.seeding import seed_everything

logger = get_logger(__name__)
DTYPE = torch.float64


def build_lv(cfg: Config) -> SyntheticLVFSI:
    return SyntheticLVFSI(cfg.geometry, cfg.flow, dtype=DTYPE)


def build_or_load_dataset(cfg: Config, data_path: Optional[str] = None,
                          lv: Optional[SyntheticLVFSI] = None) -> ReconstructionDataset:
    if data_path and Path(data_path).exists():
        logger.info("Loading dataset from %s", data_path)
        return ReconstructionDataset.load(data_path, dtype=DTYPE)
    logger.info("Generating synthetic dataset")
    return build_dataset(cfg, lv=lv, dtype=DTYPE)


def train_model(cfg: Config, dataset: ReconstructionDataset) -> Tuple[PINN, TrainState]:
    seed_everything(cfg.seed)
    model = PINN(cfg.model, dataset.scales)
    trainer = Trainer(cfg, dataset, model=model, dtype=DTYPE)
    state = trainer.train()
    return trainer.model, state


def evaluate_model(model: PINN, lv: SyntheticLVFSI, cfg: Config,
                   with_residence_time: bool = True) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics.update(evaluate_fields(model, lv, cfg))
    metrics.update(evaluate_wss(model, lv, cfg))
    if with_residence_time:
        metrics.update(evaluate_residence_time(model, lv, cfg))
    return metrics


def save_checkpoint(path: str | Path, model: PINN, cfg: Config,
                    scales: Scales) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.to_dict(),
            "scales": scales.__dict__,
        },
        path,
    )


def load_checkpoint(path: str | Path) -> Tuple[PINN, Config, Scales]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    from .config import config_from_dict

    cfg = config_from_dict(ckpt["config"])
    scales = Scales(**ckpt["scales"])
    model = PINN(cfg.model, scales).to(DTYPE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, scales


def dump_json(path: str | Path, data: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
