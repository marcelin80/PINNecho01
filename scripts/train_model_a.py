#!/usr/bin/env python3
"""Train Model A (baseline, kinematic no-slip) on the toy 2D case.

Spec-aligned entry point using ``pinnecho.models.mlp_pinn.PINNNet`` +
``pinnecho.train.composite_loss.CompositeLoss``. See ``pinnecho.train.train.main``.
"""

import _bootstrap  # noqa: F401
from pinnecho.train.train import main

if __name__ == "__main__":
    main()
