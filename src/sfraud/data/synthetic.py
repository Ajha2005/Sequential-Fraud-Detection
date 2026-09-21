"""Synthetic IEEE-CIS-shaped data generator.

The real IEEE-CIS Fraud Detection dataset ships as two tables
(train_transaction.csv, train_identity.csv) joined on TransactionID, with a
binary isFraud label and no user identifier (a `card_id` proxy is built from
the card1-card6/addr1/addr2 columns in merge.py). It also lives behind a
Kaggle login, so it is not fetchable from this environment (see download.py).

This module generates two tables with the *same shape and join key* as the
real dataset, so that the rest of the pipeline (merge -> sequences ->
baselines -> LSTM/Transformer -> autoencoder -> evaluation -> plots) is
byte-for-byte the same code path whether it runs on the real Kaggle CSVs or
on this synthetic stand-in. Swap in the real CSVs under data/raw/ and
nothing downstream changes.

Behavioral realism that matters for a *sequence* model is injected on
purpose:
  - each cardholder has a stable baseline (typical amount, product,
    device, email, address) so per-user deviation is learnable
  - fraud is injected as a *regime shift* partway through a card's history
    (account takeover), often in short bursts, with amount/category/device
    drift away from that user's baseline
  - a separate, milder "fault" shift (benign anomaly: a legitimate but
    unusual purchase, a new device, ...) is injected into a subset of
    non-fraud transactions and tagged with `synthetic_fault_flag`

`synthetic_fault_flag` is *only* available here because we control the
generator. It is never used as a model input -- it exists purely so Phase 5's
qualitative fault-validation step has a known-answer sample to check itself
against. Real Kaggle data has no such column (see fault_fraud.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRODUCT_CODES = ["W", "C", "H", "R", "S"]
CARD_TYPES = ["visa", "mastercard", "amex", "discover"]
CARD_CATEGORIES = ["debit", "credit"]
EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com", "protonmail.com"]
DEVICE_TYPES = ["mobile", "desktop"]
DEVICE_INFO_POOL = [
    "SM-G960U", "iOS Device", "Windows", "MacOS", "Pixel 3", "Trident/7.0",
    "SM-N960U", "iOS", "rv:11.0",
]


def _make_card_profiles(n_cards: int, rng: np.random.Generator) -> pd.DataFrame:
    """One fixed baseline profile per cardholder (identity-like attributes)."""
    return pd.DataFrame(
        {
            "card_id": [f"C{idx:06d}" for idx in range(n_cards)],
            "card1": rng.integers(1000, 18000, n_cards),
            "card2": rng.integers(100, 600, n_cards),
            "card3": rng.choice([150, 185], n_cards),
            "card4": rng.choice(CARD_TYPES, n_cards, p=[0.5, 0.3, 0.15, 0.05]),
            "card5": rng.integers(100, 240, n_cards),
            "card6": rng.choice(CARD_CATEGORIES, n_cards, p=[0.75, 0.25]),
            "addr1": rng.integers(100, 500, n_cards),
            "addr2": np.full(n_cards, 87),
            "P_emaildomain": rng.choice(EMAIL_DOMAINS, n_cards),
            "device_type": rng.choice(DEVICE_TYPES, n_cards, p=[0.6, 0.4]),
            "device_info": rng.choice(DEVICE_INFO_POOL, n_cards),
            "preferred_product": rng.choice(PRODUCT_CODES, n_cards),
            # log-normal baseline spend: exp(N(mu, sigma))
            "baseline_amt_mu": rng.normal(3.2, 0.8, n_cards),
            "baseline_amt_sigma": rng.uniform(0.25, 0.55, n_cards),
        }
    )


def generate_synthetic_tables(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate (transaction_df, identity_df) mirroring the raw IEEE-CIS shape.

    Returns raw-schema tables (not yet standardized/merged) so that
    merge.standardize() can treat them identically to real Kaggle CSVs.
    """
    scfg = cfg["data"]["synthetic"]
    rng = np.random.default_rng(scfg["random_state"])

    n_cards = scfg["n_cards"]
    min_t, max_t = scfg["min_txns_per_card"], scfg["max_txns_per_card"]
    n_days = scfg["n_days"]
    fraud_rate = scfg["fraud_rate"]
    fault_rate = scfg["fault_rate"]

    profiles = _make_card_profiles(n_cards, rng)
    horizon_seconds = n_days * 86400

    rows = []
    txn_id = 0
    for _, prof in profiles.iterrows():
        n_txn = int(rng.integers(min_t, max_t + 1))
        # sorted uniform draws over the horizon -> irregular inter-arrival gaps
        timestamps = np.sort(rng.uniform(0, horizon_seconds, n_txn))
        timestamps = np.clip(timestamps, 0, horizon_seconds - 1).astype(int)

        # Decide if/when this card gets taken over by fraud (a single regime
        # shift, optionally bursty over the next few transactions), mimicking
        # account takeover rather than independent per-row coin flips.
        card_has_fraud = rng.random() < min(fraud_rate * 6.5, 0.4)
        fraud_start_idx = rng.integers(max(1, n_txn // 3), n_txn) if (card_has_fraud and n_txn > 2) else None
        fraud_burst_len = int(rng.integers(2, 14)) if fraud_start_idx is not None else 0

        for i in range(n_txn):
            is_fraud = 0
            is_fault = 0
            amt_mult = 1.0
            product = prof["preferred_product"]
            device_type = prof["device_type"]
            device_info = prof["device_info"]
            email = prof["P_emaildomain"]
            addr1 = prof["addr1"]

            in_fraud_burst = fraud_start_idx is not None and fraud_start_idx <= i < fraud_start_idx + fraud_burst_len
            if in_fraud_burst:
                is_fraud = 1
                amt_mult = rng.uniform(2.5, 9.0)
                if rng.random() < 0.7:
                    product = rng.choice([p for p in PRODUCT_CODES if p != prof["preferred_product"]])
                if rng.random() < 0.6:
                    device_type = rng.choice(DEVICE_TYPES)
                    device_info = rng.choice(DEVICE_INFO_POOL)
                if rng.random() < 0.5:
                    email = rng.choice(EMAIL_DOMAINS)
                if rng.random() < 0.3:
                    addr1 = rng.integers(100, 500)
            elif rng.random() < fault_rate:
                # benign anomaly: unusual but not malicious
                is_fault = 1
                amt_mult = rng.uniform(1.8, 4.0)
                if rng.random() < 0.4:
                    device_type = rng.choice(DEVICE_TYPES)
                    device_info = rng.choice(DEVICE_INFO_POOL)
                if rng.random() < 0.2:
                    product = rng.choice(PRODUCT_CODES)
            else:
                amt_mult = 1.0
                if rng.random() < 0.08:
                    product = rng.choice(PRODUCT_CODES)

            amt = float(np.exp(rng.normal(prof["baseline_amt_mu"], prof["baseline_amt_sigma"])) * amt_mult)
            amt = round(min(max(amt, 0.5), 25000.0), 2)

            rows.append(
                {
                    "TransactionID": txn_id,
                    "card_id": prof["card_id"],
                    "TransactionDT": int(timestamps[i]),
                    "TransactionAmt": amt,
                    "ProductCD": product,
                    "card1": prof["card1"],
                    "card2": prof["card2"],
                    "card3": prof["card3"],
                    "card4": prof["card4"],
                    "card5": prof["card5"],
                    "card6": prof["card6"],
                    "addr1": addr1,
                    "addr2": prof["addr2"],
                    "P_emaildomain": email,
                    "isFraud": is_fraud,
                    "synthetic_fault_flag": is_fault,
                    "_device_type": device_type,
                    "_device_info": device_info,
                }
            )
            txn_id += 1

    txn_df = pd.DataFrame(rows).sort_values("TransactionDT").reset_index(drop=True)

    # Split off identity-table-only columns and simulate partial coverage:
    # in the real dataset only a subset of TransactionIDs have a matching
    # identity row.
    identity_coverage = 0.70
    has_identity = rng.random(len(txn_df)) < identity_coverage
    identity_df = txn_df.loc[has_identity, ["TransactionID", "_device_type", "_device_info"]].rename(
        columns={"_device_type": "DeviceType", "_device_info": "DeviceInfo"}
    )
    for i in range(1, 6):
        identity_df[f"id_{i:02d}"] = rng.normal(0, 1, len(identity_df)).round(3)

    txn_df = txn_df.drop(columns=["_device_type", "_device_info"])

    return txn_df.reset_index(drop=True), identity_df.reset_index(drop=True)
