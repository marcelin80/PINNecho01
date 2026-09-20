"""3D end-to-end smoke: the training path runs and reduces the loss."""

import torch

from pinnecho.config import Config
from pinnecho.train.train3d import (
    build_toy_dataset_3d, build_toy_model_3d, make_batch_builder_3d,
    train_model, TrainConfig, evaluate_3d,
)


def test_train_3d_smoke():
    torch.set_default_dtype(torch.float32)
    cfg = Config()
    ds, lv = build_toy_dataset_3d(cfg, n_interior=1500, n_wall=400,
                                  n_data_per_frame=80, n_frames=6, seed=0)
    model, loss_fn = build_toy_model_3d(
        cfg, ds, backbone="fsi_informed", init_seed=0,
        model_overrides={"width": 48, "depth": 3})
    loss_fn.anneal.total_steps = 120
    build_batches = make_batch_builder_3d(ds, seed=0, n_data=512, n_col=512, n_wall=128)
    tcfg = TrainConfig(steps=120, lr=3e-3, lbfgs_iters=0, log_every=120, seed=0)

    first = {}
    def logger(step, row):
        if not first:
            first.update(row)
    history = train_model(model, loss_fn, build_batches, tcfg, logger=logger)

    total = history["total"]
    assert total[-1] < total[0]  # loss decreased

    metrics = evaluate_3d(model, lv, cfg, n_interior=1500)
    for k, v in metrics.items():
        assert v == v and v >= 0.0  # finite, non-negative
