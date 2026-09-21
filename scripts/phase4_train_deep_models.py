#!/usr/bin/env python3
"""Phase 4: LSTM fraud head, plus the Transformer stretch-goal variant."""

import logging

from sfraud.training.train_lstm import run as run_lstm
from sfraud.training.train_transformer import run as run_transformer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("=== Training LSTM ===")
    run_lstm()
    logging.info("=== Training Transformer (stretch goal) ===")
    run_transformer()
