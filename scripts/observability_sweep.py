#!/usr/bin/env python3
"""Sweep Doppler acquisition conditions (windows / SNR / sparsity) and report
held-out reconstruction error. See ``pinnecho.train.observability.sweep_main``.
"""

import _bootstrap  # noqa: F401
from pinnecho.train.observability import sweep_main

if __name__ == "__main__":
    sweep_main()
