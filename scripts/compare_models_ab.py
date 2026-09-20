#!/usr/bin/env python3
"""Compare Model A (baseline) vs Model B (FSI-informed) on the toy 2D case.

Identical architecture, initialisation, data and optimiser schedule; the only
differences are the FSI momentum forcing, FSI wall-velocity BC and (with
``--traction``) the traction-continuity penalty. See
``pinnecho.train.train.compare_main``.
"""

import _bootstrap  # noqa: F401
from pinnecho.train.train import compare_main

if __name__ == "__main__":
    compare_main()
