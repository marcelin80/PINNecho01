"""A-vs-B field visualisations for the spec ``PINNNet`` path.

Renders truth / Model-A / Model-B panels for speed, vorticity and pressure on a
masked cavity grid, and a cardiac-cycle animation. Import-guarded and headless
(Agg backend) so the core package still imports when matplotlib is missing.

These helpers operate on the spec models (``model.fields(X) -> {"u","v","p",...}``)
and the analytic :class:`~pinnecho.data.synthetic_lv.SyntheticLVFSI` ground truth.
"""

from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .. import evaluate as ev


_BACKBONE_TITLES = {"baseline": "Model A (kinematic)", "fsi_informed": "Model B (FSI)"}


def _masked(arr: torch.Tensor | np.ndarray, inside: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(arr).reshape(-1).astype(float)
    a[~inside] = np.nan
    return a.reshape(n, n)


def truth_grid_fields(lv, t: float, n: int = 90) -> Dict[str, np.ndarray]:
    """Ground-truth ``speed``/``vorticity``/``pressure`` on a masked grid."""
    X, inside = lv.grid(t, n=n)
    truth = lv.all_fields(X)
    return {
        "x": X[:, 0].cpu().numpy().reshape(n, n),
        "y": X[:, 1].cpu().numpy().reshape(n, n),
        "inside": inside,
        "speed": _masked(truth["speed"], inside, n),
        "vorticity": _masked(truth["vorticity"], inside, n),
        "pressure": _masked(truth["p"], inside, n),
    }


def model_grid_fields(lv, model, t: float, n: int = 90) -> Dict[str, np.ndarray]:
    """Model ``speed``/``vorticity``/``pressure`` on the same masked grid."""
    X, inside = lv.grid(t, n=n)
    X = X.to(next(model.parameters()).dtype)
    with torch.no_grad():
        f = model.fields(X.clone())
        speed = torch.sqrt(f["u"] ** 2 + f["v"] ** 2)
    vort = ev.vorticity_2d(model, X.clone()).detach()
    return {
        "speed": _masked(speed, inside, n),
        "vorticity": _masked(vort, inside, n),
        "pressure": _masked(f["p"], inside, n),
    }


def _symmetric_limits(*arrays) -> float:
    m = 0.0
    for a in arrays:
        v = np.nanmax(np.abs(a)) if np.isfinite(a).any() else 0.0
        m = max(m, float(v))
    return m if m > 0 else 1.0


def plot_ab_panels(lv, models: Mapping[str, object], config, t: float,
                   out_path: str | Path, n: int = 90) -> Path:
    """Grid of truth / Model-A / Model-B for speed, vorticity, pressure.

    Rows are the sources (truth, then each backbone in ``models``); columns are
    the three fields. Colour scales are shared per column so panels are directly
    comparable.
    """
    truth = truth_grid_fields(lv, t, n=n)
    preds = {name: model_grid_fields(lv, m, t, n=n) for name, m in models.items()}
    rows = [("truth", truth)] + [(name, preds[name]) for name in models]

    cols = [
        ("speed", "|u| [m/s]", "viridis", False),
        ("vorticity", "vorticity [1/s]", "RdBu_r", True),
        ("pressure", "pressure [Pa]", "coolwarm", True),
    ]
    # Shared colour limits per column, taken across every row.
    limits = {}
    for key, _, _, sym in cols:
        stacked = [truth[key]] + [preds[n_][key] for n_ in models]
        if sym:
            m = _symmetric_limits(*stacked)
            limits[key] = (-m, m)
        else:
            hi = max(np.nanmax(a) for a in stacked if np.isfinite(a).any())
            limits[key] = (0.0, float(hi) if hi > 0 else 1.0)

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 3, figsize=(12, 3.6 * nrows),
                             constrained_layout=True, squeeze=False)
    for r, (src, data) in enumerate(rows):
        for c, (key, label, cmap, _sym) in enumerate(cols):
            ax = axes[r][c]
            vmin, vmax = limits[key]
            im = ax.pcolormesh(truth["x"], truth["y"], data[key], cmap=cmap,
                               shading="auto", vmin=vmin, vmax=vmax)
            ax.set_aspect("equal")
            row_title = _BACKBONE_TITLES.get(src, "Ground truth" if src == "truth" else src)
            if r == 0:
                ax.set_title(label)
            if c == 0:
                ax.set_ylabel(row_title, fontsize=11)
            fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(f"Field comparison at t = {t:.3f} s", fontsize=14)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_metric_bars(results: Mapping[str, Mapping[str, float]],
                     out_path: str | Path,
                     keys=("vel_relL2_speed", "pressure_relL2",
                           "vorticity_relL2", "wss_relL2")) -> Path:
    """Grouped bar chart of relative-L2 metrics for each backbone."""
    backbones = list(results.keys())
    keys = [k for k in keys if any(k in results[b] for b in backbones)]
    x = np.arange(len(keys))
    width = 0.8 / max(len(backbones), 1)
    fig, ax = plt.subplots(figsize=(1.7 * len(keys) + 2, 5))
    for i, b in enumerate(backbones):
        vals = [results[b].get(k, np.nan) for k in keys]
        ax.bar(x + i * width, vals, width, label=_BACKBONE_TITLES.get(b, b))
    ax.set_xticks(x + width * (len(backbones) - 1) / 2)
    ax.set_xticklabels([k.replace("_relL2", "").replace("vel_", "")
                        for k in keys], rotation=20, ha="right")
    ax.set_ylabel("relative L2 error (lower is better)")
    ax.set_title("Held-out reconstruction error: Model A vs Model B")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def animate_cycle(lv, models: Mapping[str, object], config,
                  out_path: str | Path, field: str = "speed",
                  frames: int = 24, n: int = 80, fps: int = 8) -> Optional[Path]:
    """Cardiac-cycle GIF of ``field`` for truth and each backbone.

    Uses ``imageio`` when available (per-frame PNG assembly); otherwise falls
    back to matplotlib's Pillow writer. Returns the output path, or ``None`` if
    no animation writer is available.
    """
    import io

    period = float(config.flow.period)
    t_values = np.linspace(0.0, period, frames, endpoint=False)
    cmap = {"speed": "viridis", "vorticity": "RdBu_r", "pressure": "coolwarm"}[field]
    sym = field in ("vorticity", "pressure")

    # Precompute a global colour limit across time and sources for stability.
    per_frame = []
    gmax = 0.0
    for t in t_values:
        truth = truth_grid_fields(lv, float(t), n=n)
        preds = {name: model_grid_fields(lv, m, float(t), n=n)
                 for name, m in models.items()}
        per_frame.append((truth, preds))
        stacked = [truth[field]] + [preds[nm][field] for nm in models]
        gmax = max(gmax, _symmetric_limits(*stacked))
    vmin, vmax = (-gmax, gmax) if sym else (0.0, gmax)

    ncol = 1 + len(models)

    def render_frame(idx):
        truth, preds = per_frame[idx]
        fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.0),
                                 constrained_layout=True, squeeze=False)
        sources = [("truth", truth)] + [(nm, preds[nm]) for nm in models]
        for ax, (src, data) in zip(axes[0], sources):
            im = ax.pcolormesh(truth["x"], truth["y"], data[field], cmap=cmap,
                               shading="auto", vmin=vmin, vmax=vmax)
            ax.set_aspect("equal")
            ax.set_title(_BACKBONE_TITLES.get(src, "Ground truth"))
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(f"{field}  t = {t_values[idx]:.3f}s")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        import PIL.Image as Image
        imgs = [np.asarray(Image.open(render_frame(i)).convert("RGB"))
                for i in range(frames)]
        imageio.mimsave(out_path, imgs, duration=1.0 / fps, loop=0)
        return out_path
    except Exception:
        return None
