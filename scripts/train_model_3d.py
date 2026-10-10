#!/usr/bin/env python3
"""End-to-end 3D toy validation of the spec PINN on the volume-preserving
ellipsoid ground truth. See ``pinnecho.train.train3d.main``.
"""

import _bootstrap  # noqa: F401
from pinnecho.train.train3d import main

if __name__ == "__main__":
    main()
