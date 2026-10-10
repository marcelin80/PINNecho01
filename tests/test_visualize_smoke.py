"""Smoke test: the A/B visualisation helpers run headlessly and write files.

Uses tiny, *untrained* models -- we only check the plotting/animation plumbing,
not reconstruction quality.
"""

import torch

from pinnecho.config import Config
from pinnecho.train.train import build_toy_dataset, build_toy_model
from pinnecho.eval import visualize as viz


def _tiny_config() -> Config:
    cfg = Config()
    cfg.flow.period = 0.8
    return cfg


def test_visualize_panels_and_bars(tmp_path):
    torch.set_default_dtype(torch.float32)
    cfg = _tiny_config()
    dataset, lv = build_toy_dataset(cfg)
    models = {}
    for bb in ("baseline", "fsi_informed"):
        model, _ = build_toy_model(
            cfg, dataset, backbone=bb, init_seed=0,
            model_overrides={"width": 32, "depth": 2},
        )
        models[bb] = model

    t = 0.3 * cfg.flow.period
    panel = viz.plot_ab_panels(lv, models, cfg, t, tmp_path / "fields.png", n=24)
    assert panel.exists() and panel.stat().st_size > 0

    results = {"baseline": {"vel_relL2_speed": 0.5, "pressure_relL2": 0.4,
                            "vorticity_relL2": 0.6, "wss_relL2": 0.7},
               "fsi_informed": {"vel_relL2_speed": 0.4, "pressure_relL2": 0.3,
                                "vorticity_relL2": 0.5, "wss_relL2": 0.6}}
    bars = viz.plot_metric_bars(results, tmp_path / "bars.png")
    assert bars.exists() and bars.stat().st_size > 0


def test_animate_cycle_optional(tmp_path):
    torch.set_default_dtype(torch.float32)
    cfg = _tiny_config()
    dataset, lv = build_toy_dataset(cfg)
    model, _ = build_toy_model(cfg, dataset, backbone="baseline", init_seed=0,
                               model_overrides={"width": 32, "depth": 2})
    out = viz.animate_cycle(lv, {"baseline": model}, cfg,
                            tmp_path / "cycle.gif", field="speed",
                            frames=4, n=20, fps=4)
    # imageio/pillow may be unavailable in minimal envs -> None is acceptable.
    if out is not None:
        assert out.exists() and out.stat().st_size > 0
