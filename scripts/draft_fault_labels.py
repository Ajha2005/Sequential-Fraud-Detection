#!/usr/bin/env python3
"""Optional, after review_fault_quadrant.py: fill reviewer_label / reviewer_note of the
stratified Fault sample with a rule-based DRAFT reading (benign-looking / suspicious /
unclear). The labels are heuristics over the available fields, not ground truth --
confirm or override them by hand. Rewrites the CSV in place and prints a summary."""

import argparse
import json
import logging

import pandas as pd

from sfraud.evaluation.fault_review import draft_labels
from sfraud.utils.config import metrics_dir, samples_dir

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sample", default=str(samples_dir() / "fault_review_sample.csv"))
    ap.add_argument("--threshold", type=float, default=None, help="fraud-probability threshold (default: from fault_fraud_quadrant.json)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    thr = args.threshold if args.threshold is not None else json.load(open(metrics_dir() / "fault_fraud_quadrant.json"))["fraud_prob_threshold"]
    df = draft_labels(pd.read_csv(args.sample), thr)
    df.to_csv(args.sample, index=False)
    print(df.groupby(["band", "reviewer_label"]).size().unstack(fill_value=0).to_string())
    print("\noverall:", df["reviewer_label"].value_counts().to_dict())
