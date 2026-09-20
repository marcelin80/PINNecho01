#!/usr/bin/env python3
"""Train Model A vs Model B on the toy case and render comparison figures.

Writes truth/A/B field panels (speed, vorticity, pressure), a metric bar chart,
cardiac-cycle animations, and checkpoints. See
``pinnecho.train.train.visualize_main`` for options.
"""

import _bootstrap  # noqa: F401
from pinnecho.train.train import visualize_main

if __name__ == "__main__":
    visualize_main()
