#!/usr/bin/env python3
"""Train a single PINN backbone. See ``pinnecho.cli.train_main``."""

import _bootstrap  # noqa: F401
from pinnecho.cli import train_main

if __name__ == "__main__":
    train_main()
