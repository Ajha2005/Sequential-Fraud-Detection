"""Phase 1, step 1: merge transaction + identity tables, and standardize
either real IEEE-CIS columns or our synthetic stand-in into one common
schema that every downstream stage (sequences, features, models) relies on.

Common/standardized columns produced here:
    TransactionID, card_id, TransactionDT, TransactionAmt, ProductCD,
    card_type, card_category, addr1, email_domain, merchant_category,
    device_type, device_info, isFraud, synthetic_fault_flag, is_synthetic
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from sfraud.data.synthetic import generate_synthetic_tables
from sfraud.utils.config import load_config, resolve_path

logger = logging.getLogger(__name__)

# IEEE-CIS's ProductCD has no explicit merchant-category field. We derive one
# deterministically so "merchant category" (asked for throughout the
# blueprint) is a well-defined categorical feature for both real and
# synthetic data, rather than fabricating an ungrounded column for the real
# dataset. This mapping is the *only* place merchant_category is defined.
PRODUCT_TO_MERCHANT_CATEGORY = {
    "W": "web/digital_goods",
    "C": "cash/travel",
    "H": "home/retail",
    "R": "retail/subscription",
    "S": "services",
}


def _derive_card_id(df: pd.DataFrame) -> pd.Series:
    """Real IEEE-CIS has no user identifier. card1-6 + addr1-2 is the
    standard community heuristic for approximating one distinct card/user
    entity (these fields together are stable for a given physical card)."""
    key_cols = ["card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2"]
    return df[key_cols].astype(str).agg("_".join, axis=1)


def _standardize(txn_df: pd.DataFrame, identity_df: pd.DataFrame, is_synthetic: bool) -> pd.DataFrame:
    df = txn_df.merge(identity_df, on="TransactionID", how="left")

    if "card_id" not in df.columns:
        df["card_id"] = _derive_card_id(df)

    df["card_type"] = df["card4"]
    df["card_category"] = df["card6"]
    df["email_domain"] = df["P_emaildomain"]
    df["device_type"] = df.get("DeviceType")
    df["device_info"] = df.get("DeviceInfo")
    df["merchant_category"] = df["ProductCD"].map(PRODUCT_TO_MERCHANT_CATEGORY).fillna("other")

    if "synthetic_fault_flag" not in df.columns:
        df["synthetic_fault_flag"] = 0
    df["is_synthetic"] = is_synthetic

    keep = [
        "TransactionID", "card_id", "TransactionDT", "TransactionAmt", "ProductCD",
        "card_type", "card_category", "addr1", "email_domain", "merchant_category",
        "device_type", "device_info", "isFraud", "synthetic_fault_flag", "is_synthetic",
    ]
    out = df[keep].copy()
    out["email_domain"] = out["email_domain"].fillna("missing")
    out["device_type"] = out["device_type"].fillna("missing")
    out["device_info"] = out["device_info"].fillna("missing")
    out["card_type"] = out["card_type"].fillna("missing")
    return out


def load_raw_tables(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Load real Kaggle CSVs if present in data/raw/, else fall back to the
    synthetic generator. Returns (transaction_df, identity_df, is_synthetic)."""
    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    txn_path = raw_dir / "train_transaction.csv"
    id_path = raw_dir / "train_identity.csv"

    if txn_path.exists() and id_path.exists():
        logger.info("Loading real IEEE-CIS data from %s", raw_dir)
        return pd.read_csv(txn_path), pd.read_csv(id_path), False

    if not cfg["data"].get("use_synthetic_if_missing", True):
        raise FileNotFoundError(
            f"Raw Kaggle files not found in {raw_dir} and synthetic fallback is disabled."
        )

    logger.info("Real Kaggle files not found in %s -- generating synthetic stand-in data.", raw_dir)
    txn_df, identity_df = generate_synthetic_tables(cfg)
    return txn_df, identity_df, True


def build_merged_dataset(cfg: dict | None = None, save: bool = True) -> pd.DataFrame:
    cfg = cfg or load_config()
    txn_df, identity_df, is_synthetic = load_raw_tables(cfg)
    merged = _standardize(txn_df, identity_df, is_synthetic)

    if save:
        out_dir = resolve_path(cfg["data"]["processed_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "merged.parquet"
        merged.to_parquet(out_path, index=False)
        logger.info("Saved merged dataset (%d rows) to %s", len(merged), out_path)

    return merged


def load_merged_dataset(cfg: dict | None = None, rebuild: bool = False) -> pd.DataFrame:
    cfg = cfg or load_config()
    out_path = resolve_path(cfg["data"]["processed_dir"]) / "merged.parquet"
    if out_path.exists() and not rebuild:
        return pd.read_parquet(out_path)
    return build_merged_dataset(cfg, save=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = build_merged_dataset()
    print(df.head())
    print("\nShape:", df.shape)
    print("Fraud rate: {:.3%}".format(df["isFraud"].mean()))
    print("Source:", "synthetic" if df["is_synthetic"].iloc[0] else "real Kaggle data")
