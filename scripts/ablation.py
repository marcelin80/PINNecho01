#!/usr/bin/env python3
"""Scientific-control ablations for the FSI-informed backbone claim.

Runs, within the synthetic setup, (1) the physics-term isolation with matched
exact wall information (A_exact vs B_forcing vs B_forcing_traction) and (2) the
traction-uncertainty stress test. See ``pinnecho.train.ablation.main``.
"""

import _bootstrap  # noqa: F401
from pinnecho.train.ablation import main

if __name__ == "__main__":
    main()
