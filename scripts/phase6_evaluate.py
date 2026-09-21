#!/usr/bin/env python3
"""Phase 6: model comparison (incl. accuracy/AUC-vs-N), history-length
stratification, and false positive/negative error analysis."""

import logging

from sfraud.evaluation.compare import run as run_compare
from sfraud.evaluation.error_analysis import run as run_error_analysis

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("=== Model comparison + confidence curve ===")
    run_compare()
    logging.info("=== Error analysis ===")
    run_error_analysis()
