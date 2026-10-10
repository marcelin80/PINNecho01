"""Deterministic seeding helpers."""

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if available) PyTorch RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:  # torch not installed / no CUDA
        pass
