#!/usr/bin/env python3
"""Optional, after Phase 6b: bootstrap 95% confidence intervals (resampling test
cards) for the differences between models. Writes
reports/metrics/bootstrap_differences.json. The interval covers test-sampling
noise only; models are not retrained."""

import argparse
import logging

from sfraud.evaluation.bootstrap import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-resamples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(n_resamples=args.n_resamples, seed=args.seed)
