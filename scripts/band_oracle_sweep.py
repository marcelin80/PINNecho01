#!/usr/bin/env python3
"""Band-localized forcing *oracle* dry-run (A / exact / support-preserving shuffle
/ geometric band-mask) on synthetic data. See
``pinnecho.train.band_oracle.band_oracle_main``.
"""

import _bootstrap  # noqa: F401
from pinnecho.train.band_oracle import band_oracle_main

if __name__ == "__main__":
    band_oracle_main()
