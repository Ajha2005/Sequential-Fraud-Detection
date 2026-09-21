import numpy as np
import pandas as pd

from sfraud.features.engineering import (
    PAD_IDX,
    UNK_IDX,
    add_categorical_stability_features,
    add_time_and_amount_features,
    build_vocab,
    encode_categoricals,
)


def _toy_df():
    return pd.DataFrame(
        {
            "card_id": ["A", "A", "A", "B", "B"],
            "TransactionDT": [0, 100, 500, 10, 20],
            "TransactionAmt": [10.0, 20.0, 1000.0, 5.0, 5.0],
            "ProductCD": ["W", "W", "C", "H", "H"],
            "device_type": ["mobile", "mobile", "desktop", "mobile", "mobile"],
            "email_domain": ["gmail.com", "gmail.com", "gmail.com", "yahoo.com", "yahoo.com"],
        }
    )


def test_first_transaction_has_no_causal_deviation():
    df = add_time_and_amount_features(_toy_df())
    first_rows = df.groupby("card_id").first()
    assert (first_rows["amt_dev"] == 0).all()
    assert (first_rows["amt_zscore"] == 0).all()
    assert (first_rows["time_delta"] == 0).all()


def test_time_delta_is_causal_gap():
    df = add_time_and_amount_features(_toy_df())
    card_a = df[df["card_id"] == "A"].sort_values("TransactionDT")
    assert list(card_a["time_delta"]) == [0, 100, 400]


def test_amount_spike_produces_large_deviation():
    df = add_time_and_amount_features(_toy_df())
    card_a = df[df["card_id"] == "A"].sort_values("TransactionDT")
    # third transaction (amt=1000) is a huge jump from the first two (~10-20) -> positive deviation
    assert card_a.iloc[2]["amt_dev"] > card_a.iloc[1]["amt_dev"]


def test_categorical_stability_flags_first_switch():
    df = add_time_and_amount_features(_toy_df())
    df = add_categorical_stability_features(df)
    card_a = df[df["card_id"] == "A"].sort_values("TransactionDT")
    assert list(card_a["stab_ProductCD"]) == [1.0, 1.0, 0.0]  # W, W, then switches to C
    assert list(card_a["stab_device_type"]) == [1.0, 1.0, 0.0]  # mobile, mobile, then desktop


def test_vocab_reserves_pad_and_unk_and_maps_unseen():
    df = pd.DataFrame({"cat": ["x", "x", "y"]})
    vocabs = build_vocab(df, ["cat"])
    assert vocabs["cat"]["<PAD>"] == PAD_IDX
    assert vocabs["cat"]["<UNK>"] == UNK_IDX

    unseen_df = pd.DataFrame({"cat": ["x", "z"]})  # "z" never seen during vocab building
    encoded = encode_categoricals(unseen_df, vocabs)
    assert encoded["cat_idx"].iloc[1] == UNK_IDX
