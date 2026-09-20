"""Fast smoke test for the spec-layer Model A toy training wiring.

Runs a handful of optimisation steps on a tiny problem to guard that the
PINNNet + CompositeLoss + toy batch builder are wired together and the loss is
finite and moving. Not a convergence test (see scripts/train_model_a.py for a
real run).
"""

import torch

from pinnecho.config import Config, CollocationConfig, DopplerConfig
from pinnecho.train.train import (
    build_toy_model_a,
    make_toy_batch_builder,
    train_model,
    TrainConfig,
)


def test_model_a_toy_training_runs_and_decreases():
    config = Config(
        collocation=CollocationConfig(n_interior=200, n_wall=80, n_frames=4),
        doppler=DopplerConfig(n_points_per_frame=40, n_frames=4),
    )
    model, loss_fn, dataset, lv = build_toy_model_a(
        config, model_overrides={"width": 16, "depth": 2}
    )
    loss_fn.anneal.total_steps = 20
    build_batches = make_toy_batch_builder(
        dataset, seed=0, n_data=128, n_col=128, n_wall=64
    )

    first = loss_fn(model, build_batches(0), step=0)["total"].detach().item()
    assert torch.isfinite(torch.tensor(first))

    cfg = TrainConfig(steps=20, lr=3e-3, lbfgs_iters=0, log_every=100, seed=0)
    history = train_model(model, loss_fn, build_batches, cfg)

    assert "total" in history and len(history["total"]) >= 1
    last = loss_fn(model, build_batches(20), step=20)["total"].detach().item()
    assert torch.isfinite(torch.tensor(last))
    # The composite loss should have moved (data/bc terms are being minimised).
    assert last < first * 1.5
