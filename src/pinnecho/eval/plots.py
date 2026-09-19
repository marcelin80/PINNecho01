"""Matplotlib visualisations (headless / Agg backend).

Kept dependency-light and import-guarded so the core package works even if
matplotlib is unavailable.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..config import Config
from ..data.synthetic_lv import SyntheticLVFSI
from ..physics import operators as ops


def _grid_fields(lv: SyntheticLVFSI, model, cfg: Config, t: float, n: int = 90):
    X, inside = lv.grid(t, n=n)
    X = X.to(next(model.parameters()).dtype)
    truth = lv.all_fields(X)
    Xg = X.clone().requires_grad_(True)
    u, v, p = model(Xg)
    vort_pred = ops.curl_z(u, v, Xg).detach()
    speed_pred = torch.sqrt(u ** 2 + v ** 2).detach()

    def mask(arr):
        a = arr.reshape(-1).cpu().numpy().astype(float)
        a[~inside] = np.nan
        return a.reshape(n, n)

    return {
        "x": X[:, 0].cpu().numpy().reshape(n, n),
        "y": X[:, 1].cpu().numpy().reshape(n, n),
        "speed_true": mask(truth["speed"]),
        "speed_pred": mask(speed_pred),
        "vort_true": mask(truth["vorticity"]),
        "vort_pred": mask(vort_pred),
    }


def plot_field_comparison(lv: SyntheticLVFSI, model, cfg: Config, t: float,
                          out_path: str | Path) -> None:
    g = _grid_fields(lv, model, cfg, t)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    panels = [
        ("speed_true", "|u| truth [m/s]", "viridis"),
        ("speed_pred", "|u| PINN [m/s]", "viridis"),
        ("vort_true", "vorticity truth [1/s]", "RdBu_r"),
        ("vort_pred", "vorticity PINN [1/s]", "RdBu_r"),
    ]
    for ax, (key, title, cmap) in zip(axes.ravel(), panels):
        im = ax.pcolormesh(g["x"], g["y"], g[key], cmap=cmap, shading="auto")
        ax.set_title(title)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"Field comparison at t={t:.3f}s")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_loss_curves(states: Dict[str, "object"], out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, state in states.items():
        h = state.history
        if "iter" in h and "total" in h:
            ax.semilogy(h["iter"], h["total"], label=name)
    ax.set_xlabel("iteration")
    ax.set_ylabel("total loss")
    ax.set_title("Training loss")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_metric_comparison(metrics_by_backbone: Dict[str, Dict[str, float]],
                           keys: List[str], out_path: str | Path) -> None:
    backbones = list(metrics_by_backbone.keys())
    n_keys = len(keys)
    x = np.arange(n_keys)
    width = 0.8 / max(len(backbones), 1)

    fig, ax = plt.subplots(figsize=(1.6 * n_keys + 2, 5))
    for i, bb in enumerate(backbones):
        vals = [metrics_by_backbone[bb].get(k, np.nan) for k in keys]
        ax.bar(x + i * width, vals, width, label=bb)
    ax.set_xticks(x + width * (len(backbones) - 1) / 2)
    ax.set_xticklabels([k.replace("rel_l2_", "") for k in keys], rotation=30, ha="right")
    ax.set_ylabel("relative L2 error (lower is better)")
    ax.set_title("Backbone comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
