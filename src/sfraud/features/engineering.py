"""Phase 2: per-step feature engineering.

All numeric features that depend on a card's own history (amt_dev,
amt_zscore, time_delta) are computed *causally*: at transaction t they only
use transactions 1..t-1 for that card, via a groupby + expanding().shift(1).
This mirrors how the model would actually see data in production (it cannot
peek at its own current transaction's contribution to "the user's typical
behavior") and keeps Phase 6 error analysis honest.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

PAD_IDX = 0
UNK_IDX = 1

# Categorical fields worth a personal-baseline "stability" feature: whether
# the current value matches this card's most common *prior* value. These are
# exactly the fields the synthetic fraud/fault injection perturbs (see
# synthetic.py); card_type/card_category are excluded because they never
# change for a given physical card, so a stability feature for them would be
# a constant. Without this, the autoencoder in Phase 5 only has a
# personalized baseline for *amount* (amt_dev/amt_zscore) and has to judge
# "did this user switch device/email/product" against the whole population
# instead of against their own history -- which is a much weaker signal.
STABILITY_COLUMNS = ["ProductCD", "device_type", "email_domain"]


def _stability_series(s: pd.Series) -> pd.Series:
    seen: Counter = Counter()
    out = np.empty(len(s), dtype=np.float32)
    for i, v in enumerate(s.to_numpy()):
        if not seen:
            out[i] = 1.0  # no history yet -> neutral, consistent with amt_dev=0 for the first txn
        else:
            modal_val, _ = seen.most_common(1)[0]
            out[i] = 1.0 if v == modal_val else 0.0
        seen[v] += 1
    return pd.Series(out, index=s.index)


def add_categorical_stability_features(df: pd.DataFrame, columns: list[str] = STABILITY_COLUMNS) -> pd.DataFrame:
    for col in columns:
        df[f"stab_{col}"] = df.groupby("card_id")[col].transform(_stability_series)
    return df


def add_time_and_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["card_id", "TransactionDT"]).reset_index(drop=True)

    df["log_amt"] = np.log1p(df["TransactionAmt"])

    prev_dt = df.groupby("card_id")["TransactionDT"].shift(1)
    time_delta = (df["TransactionDT"] - prev_dt).fillna(0.0).clip(lower=0.0)
    df["time_delta"] = time_delta
    df["log_time_delta"] = np.log1p(time_delta)

    grp = df.groupby("card_id")["log_amt"]
    causal_mean = grp.transform(lambda s: s.expanding().mean().shift(1))
    causal_std = grp.transform(lambda s: s.expanding().std().shift(1))
    df["amt_dev"] = (df["log_amt"] - causal_mean).fillna(0.0)
    df["amt_zscore"] = (df["amt_dev"] / (causal_std.fillna(0.0) + 1e-3)).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    seconds_per_day = 86400
    df["txn_hour"] = (df["TransactionDT"] % seconds_per_day) / 3600.0
    df["txn_dow"] = (df["TransactionDT"] // seconds_per_day) % 7

    df["txn_index_in_card"] = df.groupby("card_id").cumcount()  # 0-based position -> N = index + 1
    return df


def build_vocab(df: pd.DataFrame, columns: list[str], min_freq: int = 1) -> dict[str, dict[str, int]]:
    """index 0 = PAD, index 1 = UNK, then categories sorted by frequency desc."""
    vocabs = {}
    for col in columns:
        counts = df[col].astype(str).value_counts()
        counts = counts[counts >= min_freq]
        vocab = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
        for cat in counts.index:
            vocab[cat] = len(vocab)
        vocabs[col] = vocab
    return vocabs


def encode_categoricals(df: pd.DataFrame, vocabs: dict[str, dict[str, int]]) -> pd.DataFrame:
    df = df.copy()
    for col, vocab in vocabs.items():
        df[f"{col}_idx"] = df[col].astype(str).map(vocab).fillna(UNK_IDX).astype(int)
    return df


def engineer_features(df: pd.DataFrame, categorical_columns: list[str]) -> tuple[pd.DataFrame, dict]:
    df = add_time_and_amount_features(df)
    df = add_categorical_stability_features(df)
    vocabs = build_vocab(df, categorical_columns)
    df = encode_categoricals(df, vocabs)
    return df, vocabs
