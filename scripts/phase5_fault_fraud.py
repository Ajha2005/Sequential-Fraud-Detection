#!/usr/bin/env python3
"""Phase 5: autoencoder anomaly head, then the fault/fraud quadrant logic
from Section 3, evaluated on the held-out val+test predictions of the LSTM."""

import logging

from sfraud.evaluation.fault_fraud import run as run_quadrant
from sfraud.training.train_autoencoder import run as run_autoencoder

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("=== Training autoencoder anomaly head ===")
    run_autoencoder()
    logging.info("=== Fault/fraud quadrant logic ===")
    run_quadrant()
