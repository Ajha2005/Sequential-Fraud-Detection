#!/usr/bin/env python3
"""Phase 7: the seven required plots plus the bonus embedding visualization."""

import logging

from sfraud.visualization.plots import generate_all_plots

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    generate_all_plots()
