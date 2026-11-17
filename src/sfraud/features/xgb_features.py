"""Feature table for the gradient-boosted models (XGBoost "plus" + ablation).

The sequence models and the original baselines work from the standardized,
z-scored, zero-filled feature set built in sequences.py. Gradient-boosted
trees can use much more of the raw data, so they get their own table
(`features_xgb.parquet`), built from the same card-level split:

  * C/D/V values stay NaN (XGBoost learns what a missing value means; on
    IEEE-CIS the missingness pattern is itself predictive), plus per-family
    missing-value counts.
  * Extra raw columns the standard feature set drops: card1/2/3/5, addr1/2,
    dist1/2, R_emaildomain and the id_* identity fields.
  * Frequency encodings (computed on TRAIN rows only, so no val/test
    information leaks into them) and frequency-ranked integer codes for
    categoricals.
  * Richer causal per-card history: count/amount aggregates over the prior
    transactions, rolling 1h/1d/7d counts and spend, and "changed since the
    previous transaction" flags. Everything uses only transactions strictly
    before the current one (ties at identical timestamps are ordered by row).

History features are tracked separately so a history-ablation model can be
trained on exactly the same features minus every history-derived column.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from sfraud.data.merge import EXTENSION_CATEGORICAL_COLUMNS, EXTENSION_NUMERIC_COLUMNS, EXTRA_RAW_COLUMNS
from sfraud.features.engineering import STABILITY_COLUMNS
from sfraud.utils.config import resolve_path

logger = logging.getLogger(__name__)

XGB_TABLE_FILENAME = "features_xgb.parquet"
XGB_META_FILENAME = "xgb_feature_meta.json"

META_COLUMNS = ["TransactionID", "card_id", "TransactionDT", "isFraud", "synthetic_fault_flag", "split"]

HISTORY_WINDOWS = {"1h": 3600, "1d": 86400, "7d": 7 * 86400}
CHANGE_COLUMNS = ["ProductCD", "email_domain", "device_type", "addr1"]
FREQ_NUMERIC_COLUMNS = ["card1", "card2", "card3", "card5", "addr1", "addr2"]

# History-derived columns that already exist from features/engineering.py.
EXISTING_HISTORY_COLUMNS = [
    "log_time_delta", "amt_dev", "amt_zscore", "txn_index_in_card",
    *[f"stab_{c}" for c in STABILITY_COLUMNS],
]

# Rows are keyed as card_code * 2**28 + TransactionDT so a single searchsorted
# finds each row's rolling-window start. TransactionDT in IEEE-CIS is < 2**24.
_KEY_SHIFT = 1 << 28


def compute_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal per-card history features for a frame sorted by (card_id,
    TransactionDT). Returns ONLY the new columns, aligned to df's index.

    All aggregates cover the card's earlier rows (row order within a card),
    never the current row. First transactions get NaN aggregates and zero
    counts."""
    n = len(df)
    if n == 0:
        return pd.DataFrame(index=df.index)

    codes, _ = pd.factorize(df["card_id"])
    dt = df["TransactionDT"].to_numpy(dtype=np.int64)
    amt = df["TransactionAmt"].to_numpy(dtype=np.float64)

    same_as_prev = np.zeros(n, dtype=bool)
    same_as_prev[1:] = codes[1:] == codes[:-1]
    if int((~same_as_prev).sum()) != int(codes.max()) + 1:
        raise ValueError("compute_history_features needs rows grouped by card_id (sort by card_id, TransactionDT)")
    if np.any(np.diff(dt)[same_as_prev[1:]] < 0):
        raise ValueError("compute_history_features needs TransactionDT non-decreasing within each card")
    if dt.min() < 0 or dt.max() + max(HISTORY_WINDOWS.values()) >= _KEY_SHIFT:
        raise ValueError("TransactionDT out of the supported range for rolling-window features")

    idx = np.arange(n)
    first_idx = np.flatnonzero(~same_as_prev)  # first row of each card; codes are in appearance order
    card_first = first_idx[codes]
    pos = idx - card_first  # number of prior transactions for this card

    csum = np.concatenate([[0.0], np.cumsum(amt)])
    csum_sq = np.concatenate([[0.0], np.cumsum(amt * amt)])
    prior_sum = csum[idx] - csum[card_first]
    prior_sum_sq = csum_sq[idx] - csum_sq[card_first]

    with np.errstate(divide="ignore", invalid="ignore"):
        prior_mean = np.where(pos >= 1, prior_sum / pos, np.nan)
        var_pop = prior_sum_sq / pos - prior_mean ** 2
        prior_std = np.where(pos >= 2, np.sqrt(np.maximum(var_pop * pos / (pos - 1), 0.0)), np.nan)
        ratio = amt / prior_mean
    ratio[~np.isfinite(ratio)] = np.nan

    prior_max = np.array(pd.Series(amt).groupby(codes).cummax().shift(1), dtype=np.float64)  # writable copy
    prior_max[pos == 0] = np.nan

    out = {
        "hist_time_since_first": (dt - dt[card_first]).astype(np.float32),
        "hist_amt_mean": prior_mean,
        "hist_amt_std": prior_std,
        "hist_amt_max": prior_max,
        "hist_amt_ratio": ratio,
    }

    key = codes.astype(np.int64) * _KEY_SHIFT + dt
    for name, window in HISTORY_WINDOWS.items():
        start = np.searchsorted(key, key - window, side="left")
        out[f"cnt_{name}"] = (idx - start).astype(np.float32)
        if name == "1d":
            out["amt_sum_1d"] = csum[idx] - csum[start]

    for col in CHANGE_COLUMNS:
        if col not in df.columns:
            continue
        # factorize (NaN gets its own code, so NaN == NaN) rather than astype(str):
        # pandas 2 turns NaN into "nan" but pandas 3 keeps it missing, and
        # NaN != NaN would flag a phantom "change" between two missing values.
        cur, _ = pd.factorize(df[col], use_na_sentinel=False)
        chg = np.zeros(n, dtype=np.float32)
        chg[1:] = ((cur[1:] != cur[:-1]) & same_as_prev[1:]).astype(np.float32)
        out[f"chg_prev_{col}"] = chg

    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in out.items()}, index=df.index)


def frequency_encode(df: pd.DataFrame, columns: list[str], train_mask: np.ndarray) -> pd.DataFrame:
    """Share of TRAIN rows having each value (0 for unseen or missing values)."""
    n_train = max(int(np.sum(train_mask)), 1)
    out = {}
    for col in columns:
        counts = df.loc[train_mask, col].value_counts(dropna=True)
        out[f"freq_{col}"] = (df[col].map(counts).fillna(0.0) / n_train).astype(np.float32)
    return pd.DataFrame(out, index=df.index)


def label_encode(df: pd.DataFrame, columns: list[str], train_mask: np.ndarray) -> pd.DataFrame:
    """Frequency-ranked integer codes from TRAIN rows (0 = most common). A value
    never seen in train gets -1; a missing value stays NaN so XGBoost treats it
    as missing."""
    out = {}
    for col in columns:
        s = df[col]
        ranked = s[train_mask].dropna().value_counts().index
        mapping = {v: i for i, v in enumerate(ranked)}
        codes = s.map(mapping)
        codes = codes.where(s.isna() | codes.notna(), -1.0)
        out[f"code_{col}"] = codes.astype(np.float32)
    return pd.DataFrame(out, index=df.index)


def _present(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in df.columns]


def build_xgb_table(
    df: pd.DataFrame, base_numeric: list[str], base_categorical: list[str]
) -> tuple[pd.DataFrame, dict]:
    """df: engineered frame (sorted by card_id, TransactionDT) with a `split`
    column and C/D/V values still NaN. Returns (table, meta) where meta lists
    every feature column and which of them are history-derived."""
    if "split" not in df.columns:
        raise ValueError("build_xgb_table needs a 'split' column (run assign_splits first)")
    train_mask = (df["split"] == "train").to_numpy()

    ext_numeric = _present(df, EXTENSION_NUMERIC_COLUMNS)
    ext_categorical = _present(df, EXTENSION_CATEGORICAL_COLUMNS)

    extras = _present(df, EXTRA_RAW_COLUMNS)
    if "addr1" in df.columns:
        extras = ["addr1"] + extras
    extras = list(dict.fromkeys(extras))
    extra_numeric = [c for c in extras if pd.api.types.is_numeric_dtype(df[c])]
    extra_categorical = [c for c in extras if c not in extra_numeric]

    categorical = list(dict.fromkeys(
        _present(df, base_categorical) + _present(df, ["device_info"]) + ext_categorical + extra_categorical
    ))
    raw_numeric = list(dict.fromkeys(
        _present(df, base_numeric) + _present(df, ["TransactionAmt", "txn_index_in_card"]) + extra_numeric + ext_numeric
    ))

    parts = [df[raw_numeric].astype(np.float32)]

    missing_counts = {}
    for label, family in (
        ("C", [c for c in ext_numeric if c.startswith("C")]),
        ("D", [c for c in ext_numeric if c.startswith("D")]),
        ("V", [c for c in ext_numeric if c.startswith("V")]),
        ("id", [c for c in extras if c.startswith("id_")]),
    ):
        if family:
            missing_counts[f"n_missing_{label}"] = df[family].isna().sum(axis=1).astype(np.float32)
    if missing_counts:
        parts.append(pd.DataFrame(missing_counts, index=df.index))

    parts.append(label_encode(df, categorical, train_mask))
    parts.append(frequency_encode(df, categorical + _present(df, FREQ_NUMERIC_COLUMNS), train_mask))
    history = compute_history_features(df)
    parts.append(history)

    features = pd.concat(parts, axis=1)
    features = features.loc[:, ~features.columns.duplicated()]

    history_cols = [c for c in EXISTING_HISTORY_COLUMNS if c in features.columns] + list(history.columns)
    meta_cols = _present(df, META_COLUMNS)
    table = pd.concat([df[meta_cols].reset_index(drop=True), features.reset_index(drop=True)], axis=1)

    meta = {
        "features": list(features.columns),
        "history_features": history_cols,
        "n_features": int(features.shape[1]),
        "n_history_features": len(history_cols),
        "groups": {
            "raw_numeric": raw_numeric,
            "categorical": categorical,
            "missing_counts": list(missing_counts),
            "history_new": list(history.columns),
        },
    }
    return table, meta


def write_xgb_table(table: pd.DataFrame, meta: dict, out_dir) -> None:
    table.to_parquet(out_dir / XGB_TABLE_FILENAME, index=False)
    with open(out_dir / XGB_META_FILENAME, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(
        "Saved %s (%d rows, %d features, %d of them history-derived) to %s",
        XGB_TABLE_FILENAME, len(table), meta["n_features"], meta["n_history_features"], out_dir,
    )


def load_xgb_table(cfg: dict) -> tuple[pd.DataFrame, dict]:
    processed_dir = resolve_path(cfg["data"]["processed_dir"])
    with open(processed_dir / XGB_META_FILENAME) as f:
        meta = json.load(f)
    return pd.read_parquet(processed_dir / XGB_TABLE_FILENAME), meta
