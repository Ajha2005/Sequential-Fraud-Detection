#!/usr/bin/env python3
"""Optional, after Phase 5: characterize the Fault quadrant on real data.

Measures over all Fault rows whether the anomaly scores look driven by extreme
feature values, and draws a stratified random sample across the whole score
range for a manual look. Writes reports/metrics/fault_review_summary.json and
reports/samples/fault_review_sample.csv (with blank reviewer_label /
reviewer_note columns to fill in). Characterizes the quadrant; it cannot prove
it is benign (there is no "fault" ground truth)."""

import argparse
import logging

from sfraud.evaluation.fault_review import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-band", type=int, default=12, help="sampled rows per score band (5 bands)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(per_band=args.per_band, seed=args.seed)
