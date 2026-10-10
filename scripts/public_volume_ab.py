#!/usr/bin/env python3
"""A vs deliverable-B on a public 4D-flow / phantom volume (f = 0)."""

import _bootstrap  # noqa: F401
from pinnecho.train.public_volume_ab import public_volume_ab_main

if __name__ == "__main__":
    public_volume_ab_main()
