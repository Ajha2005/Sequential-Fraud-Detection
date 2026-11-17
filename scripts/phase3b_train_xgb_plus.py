#!/usr/bin/env python3
"""Phase 3b: stronger XGBoost on the richer feature table, plus its
no-history ablation (run after Phase 2)."""

import logging

from sfraud.training.train_xgb_plus import run

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
