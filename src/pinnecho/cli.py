"""Command-line entry points for PINNecho.

Exposed as console scripts (see ``pyproject.toml``):

* ``pinnecho-generate``  -- generate & save a synthetic dataset
* ``pinnecho-train``     -- train a single backbone
* ``pinnecho-evaluate``  -- evaluate a checkpoint
* ``pinnecho-compare``   -- train + evaluate baseline vs FSI-informed and report
"""

import argparse
import dataclasses
from pathlib import Path
from typing import Dict, List

from .config import Config, load_config
from .pipeline import (
    build_lv,
    build_or_load_dataset,
    train_model,
    evaluate_model,
    save_checkpoint,
    load_checkpoint,
    dump_json,
)
from .utils.logging import get_logger

logger = get_logger("pinnecho.cli")

COMPARE_KEYS: List[str] = [
    "rel_l2_speed",
    "rel_l2_pressure",
    "rel_l2_vorticity",
    "rel_l2_wss",
    "rel_l2_residence_time",
]


def _load_cfg(path: str | None) -> Config:
    return load_config(path) if path else Config()


def _apply_overrides(cfg: Config, args) -> Config:
    if getattr(args, "iterations", None) is not None:
        cfg.train.iterations = args.iterations
    if getattr(args, "seed", None) is not None:
        cfg.seed = args.seed
    if getattr(args, "backbone", None):
        cfg.physics.backbone = args.backbone
    return cfg


# --------------------------------------------------------------------------- #
def generate_main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic LV FSI dataset")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="data/generated/dataset.npz")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = _apply_overrides(_load_cfg(args.config), args)
    lv = build_lv(cfg)
    logger.info("Field statistics: %s", lv.field_statistics(seed=cfg.seed))
    ds = build_or_load_dataset(cfg, data_path=None, lv=lv)
    ds.save(args.out)
    logger.info("Saved dataset to %s (meas=%d col=%d wall=%d)",
                args.out, ds.meas_X.shape[0], ds.col_X.shape[0], ds.wall_X.shape[0])


def train_main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train a PINN backbone")
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", default=None, help="optional dataset .npz")
    ap.add_argument("--out", default="outputs/run")
    ap.add_argument("--backbone", default=None, choices=["baseline", "fsi_informed"])
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args(argv)

    cfg = _apply_overrides(_load_cfg(args.config), args)
    out = Path(args.out)
    lv = build_lv(cfg)
    ds = build_or_load_dataset(cfg, data_path=args.data, lv=lv)

    model, state = train_model(cfg, ds)
    save_checkpoint(out / "model.pt", model, cfg, ds.scales)
    dump_json(out / "loss_history.json", state.history)
    cfg.save(out / "config.yaml")

    if not args.no_eval:
        metrics = evaluate_model(model, lv, cfg)
        dump_json(out / "metrics.json", metrics)
        logger.info("Metrics: %s", metrics)


def evaluate_main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)

    model, cfg, _ = load_checkpoint(args.checkpoint)
    lv = build_lv(cfg)
    metrics = evaluate_model(model, lv, cfg)
    dump_json(Path(args.out) / "metrics.json", metrics)
    logger.info("Metrics: %s", metrics)

    if args.plot:
        from .eval.plots import plot_field_comparison

        t = 0.5 * cfg.flow.period
        plot_field_comparison(lv, model, cfg, t, Path(args.out) / "fields.png")
        logger.info("Wrote field comparison plot")


def _format_table(metrics_by_backbone: Dict[str, Dict[str, float]],
                  keys: List[str]) -> str:
    backbones = list(metrics_by_backbone.keys())
    header = f"{'metric':<26}" + "".join(f"{bb:>16}" for bb in backbones)
    if len(backbones) == 2:
        header += f"{'improvement':>16}"
    lines = [header, "-" * len(header)]
    for k in keys:
        row = f"{k:<26}"
        vals = [metrics_by_backbone[bb].get(k, float('nan')) for bb in backbones]
        for val in vals:
            row += f"{val:>16.4f}"
        if len(backbones) == 2 and vals[0] != 0:
            impr = 100.0 * (vals[0] - vals[1]) / abs(vals[0])
            row += f"{impr:>15.1f}%"
        lines.append(row)
    return "\n".join(lines)


def compare_main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Compare baseline vs FSI-informed backbones on identical data")
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="outputs/compare")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-residence-time", action="store_true")
    args = ap.parse_args(argv)

    base_cfg = _apply_overrides(_load_cfg(args.config), args)
    out = Path(args.out)
    lv = build_lv(base_cfg)
    # Identical dataset for both backbones -> a fair comparison.
    ds = build_or_load_dataset(base_cfg, data_path=args.data, lv=lv)

    metrics_by_backbone: Dict[str, Dict[str, float]] = {}
    states = {}
    for backbone in ["baseline", "fsi_informed"]:
        cfg = dataclasses.replace(base_cfg)
        cfg.physics = dataclasses.replace(base_cfg.physics, backbone=backbone)
        cfg.experiment = backbone
        logger.info("=== Training backbone: %s ===", backbone)
        # Fresh dataset copy so in-place device moves don't interfere.
        ds_copy = build_or_load_dataset(base_cfg, data_path=args.data, lv=lv)
        model, state = train_model(cfg, ds_copy)
        states[backbone] = state
        metrics = evaluate_model(model, lv, cfg,
                                 with_residence_time=not args.no_residence_time)
        metrics_by_backbone[backbone] = metrics
        save_checkpoint(out / backbone / "model.pt", model, cfg, ds_copy.scales)
        logger.info("[%s] metrics: %s", backbone, metrics)

    dump_json(out / "comparison.json", metrics_by_backbone)
    keys = [k for k in COMPARE_KEYS
            if all(k in m for m in metrics_by_backbone.values())]
    table = _format_table(metrics_by_backbone, keys)
    print("\n" + table + "\n")
    (out / "comparison.txt").write_text(table + "\n")

    if not args.no_plots:
        from .eval.plots import plot_loss_curves, plot_metric_comparison

        plot_loss_curves(states, out / "loss_curves.png")
        plot_metric_comparison(metrics_by_backbone, keys, out / "metric_comparison.png")
        logger.info("Wrote comparison plots to %s", out)


if __name__ == "__main__":  # pragma: no cover
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "compare"
    {"generate": generate_main, "train": train_main,
     "evaluate": evaluate_main, "compare": compare_main}[cmd](sys.argv[2:])
