#!/usr/bin/env python3
"""Phase 1: merge transaction+identity, exploratory analysis, sanity-check plots."""

import logging

from sfraud.data.eda import run_eda

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_eda()
