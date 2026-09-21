"""Phase 1: exploratory data analysis.

Produces the "sanity-check the data before modeling" plots and stats the
blueprint calls for: class imbalance, missing values, amount/time-gap
distributions, fraud rate by category, and the per-card sequence-length
histogram used later to justify the fixed padding length in sequences.py.
"""

from __future__ import annotations

import json
import logging

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sfraud.data.merge import load_merged_dataset
from sfraud.utils.config import figures_dir, load_config, metrics_dir
from sfraud.visualization.style import COLOR_FRAUD, COLOR_NEUTRAL, COLOR_NORMAL

logger = logging.getLogger(__name__)


def _time_deltas_per_card(df: pd.DataFrame) -> np.ndarray:
    df = df.sort_values(["card_id", "TransactionDT"])
    deltas = df.groupby("card_id")["TransactionDT"].diff().dropna()
    return deltas.values


def run_eda(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    df = load_merged_dataset(cfg)
    fig_dir = figures_dir()

    stats = {}
    stats["n_rows"] = int(len(df))
    stats["n_cards"] = int(df["card_id"].nunique())
    stats["fraud_rate"] = float(df["isFraud"].mean())
    stats["missing_values"] = {k: int(v) for k, v in df.isnull().sum().items() if v > 0}
    stats["data_source"] = "synthetic" if bool(df["is_synthetic"].iloc[0]) else "real_kaggle"

    seq_lengths = df.groupby("card_id").size()
    stats["seq_len_stats"] = {
        "mean": float(seq_lengths.mean()),
        "median": float(seq_lengths.median()),
        "p90": float(seq_lengths.quantile(0.90)),
        "p95": float(seq_lengths.quantile(0.95)),
        "max": int(seq_lengths.max()),
    }

    # --- Figure: class imbalance ---
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["isFraud"].value_counts().sort_index()
    ax.bar(["Non-fraud", "Fraud"], counts.values, color=[COLOR_NORMAL, COLOR_FRAUD])
    ax.set_ylim(top=counts.values.max() * 1.18)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / len(df):.1%})", ha="center", va="bottom")
    ax.set_title(f"Class imbalance (fraud rate = {stats['fraud_rate']:.2%})")
    ax.set_ylabel("Transaction count")
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_class_imbalance.png", dpi=150)
    plt.close(fig)

    # --- Figure: transaction amount distribution (log scale), fraud vs not ---
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, color, name in [(0, COLOR_NORMAL, "Non-fraud"), (1, COLOR_FRAUD, "Fraud")]:
        vals = np.log1p(df.loc[df["isFraud"] == label, "TransactionAmt"])
        ax.hist(vals, bins=40, alpha=0.55, label=name, color=color, density=True)
    ax.set_xlabel("log(1 + TransactionAmt)")
    ax.set_ylabel("Density")
    ax.set_title("Transaction amount distribution by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_amount_distribution.png", dpi=150)
    plt.close(fig)

    # --- Figure: time gaps between consecutive transactions per card ---
    deltas_hours = _time_deltas_per_card(df) / 3600.0
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.log1p(deltas_hours), bins=40, color=COLOR_NEUTRAL)
    ax.set_xlabel("log(1 + hours since previous transaction)")
    ax.set_ylabel("Count")
    ax.set_title("Inter-transaction time gaps")
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_time_gaps.png", dpi=150)
    plt.close(fig)

    # --- Figure: fraud rate by merchant category ---
    fig, ax = plt.subplots(figsize=(6, 4))
    rate_by_cat = df.groupby("merchant_category")["isFraud"].mean().sort_values(ascending=False)
    ax.bar(rate_by_cat.index, rate_by_cat.values, color=COLOR_FRAUD)
    ax.axhline(stats["fraud_rate"], color="black", linestyle="--", linewidth=1, label="overall rate")
    ax.set_ylabel("Fraud rate")
    ax.set_title("Fraud rate by merchant category")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_fraud_rate_by_category.png", dpi=150)
    plt.close(fig)

    # --- Figure: sequence length histogram (justifies max_seq_len choice) ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(seq_lengths.values, bins=40, color=COLOR_NORMAL)
    chosen = cfg["sequence"]["max_seq_len"]
    ax.axvline(chosen, color=COLOR_FRAUD, linestyle="--", label=f"chosen max_seq_len={chosen}")
    ax.set_xlabel("Transactions per card")
    ax.set_ylabel("Number of cards")
    ax.set_title("Per-card sequence length distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_sequence_length_hist.png", dpi=150)
    plt.close(fig)

    with open(metrics_dir() / "eda_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("EDA stats: %s", json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_eda()
