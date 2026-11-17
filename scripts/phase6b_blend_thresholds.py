#!/usr/bin/env python3
"""Phase 6b: every model on the same rows, a validation-chosen blend, and
validation-chosen decision thresholds (run after Phases 3b, 4 and 5)."""

import logging

from sfraud.evaluation.blend import run

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
