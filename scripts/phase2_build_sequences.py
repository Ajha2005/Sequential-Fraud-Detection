#!/usr/bin/env python3
"""Phase 2: per-step feature engineering + padded sequence construction."""

import logging

from sfraud.data.sequences import build_sequences

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_sequences()
