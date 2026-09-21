import numpy as np
import pandas as pd

from sfraud.data.sequences import _build_split_arrays, assign_splits


def _toy_cfg():
    return {"sequence": {"train_frac": 0.5, "val_frac": 0.25, "test_frac": 0.25}}


def test_assign_splits_is_disjoint_and_covers_all_cards():
    df = pd.DataFrame(
        {
            "card_id": [f"C{i}" for i in range(20) for _ in range(3)],
            "TransactionDT": list(range(60)),
        }
    )
    out = assign_splits(df, _toy_cfg())
    per_card_splits = out.groupby("card_id")["split"].nunique()
    assert (per_card_splits == 1).all(), "a card must not be split across train/val/test"
    assert set(out["split"].unique()) <= {"train", "val", "test"}
    assert out["card_id"].nunique() == 20


def test_padding_and_truncation():
    df = pd.DataFrame(
        {
            "card_id": ["short", "short", "long", "long", "long", "long"],
            "TransactionDT": [0, 10, 0, 10, 20, 30],
            "isFraud": [0, 1, 0, 0, 1, 0],
            "synthetic_fault_flag": [0, 0, 0, 0, 0, 0],
            "TransactionAmt": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "TransactionID": [100, 101, 200, 201, 202, 203],
            "num_feat": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "cat_idx": [1, 1, 2, 2, 2, 2],
        }
    )
    arrays = _build_split_arrays(df, numeric_cols=["num_feat"], categorical_cols=["cat"], max_seq_len=3)

    # "short" card: 2 real steps, right-padded to length 3
    short_i = list(arrays["card_ids"]).index("short")
    assert arrays["mask"][short_i].tolist() == [1.0, 1.0, 0.0]
    assert arrays["y"][short_i].tolist() == [0.0, 1.0, 0.0]

    # "long" card: 4 real steps, truncated to the FIRST 3
    long_i = list(arrays["card_ids"]).index("long")
    assert arrays["mask"][long_i].tolist() == [1.0, 1.0, 1.0]
    assert arrays["txn_ids"][long_i].tolist() == [200, 201, 202]
