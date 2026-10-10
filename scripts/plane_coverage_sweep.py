#!/usr/bin/env python3
"""Sweep 3D Doppler acquisition protocols (single plane -> 4-view -> idealized
point windows) and report held-out reconstruction error. See
``pinnecho.train.plane_coverage.coverage_main``.
"""

import _bootstrap  # noqa: F401
from pinnecho.train.plane_coverage import coverage_main

if __name__ == "__main__":
    coverage_main()
