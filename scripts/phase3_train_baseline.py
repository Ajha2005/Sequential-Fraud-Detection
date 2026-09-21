#!/usr/bin/env python3
"""Phase 3: classical baselines (XGBoost, Logistic Regression)."""

import logging

from sfraud.training.train_baseline import run

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
