import numpy as np
import pandas as pd

from sfraud.data.merge import _derive_card_id, _standardize
from sfraud.data.sequences import _effective_feature_lists


def _txn_row(txn_id, **overrides):
    row = {
        "TransactionID": txn_id, "card1": 1111, "card2": 222.0, "card3": 150.0, "card4": "visa",
        "card5": 226.0, "card6": "debit", "addr1": 315.0, "addr2": 87.0,
    }
    row.update(overrides)
    return row


def test_card_id_missing_field_does_not_collide_across_rows():
    """Regression: a naive astype(str) turned every missing addr1 into the
    shared literal "nan", merging unrelated cardholders into one mega-card."""
    df = pd.DataFrame([_txn_row(1, addr1=np.nan), _txn_row(2, addr1=np.nan)])
    card_ids = _derive_card_id(df)
    assert card_ids.iloc[0] != card_ids.iloc[1]
    assert "nan" not in card_ids.iloc[0]


def test_card_id_fully_matching_rows_still_collide():
    df = pd.DataFrame([_txn_row(1), _txn_row(2)])
    card_ids = _derive_card_id(df)
    assert card_ids.iloc[0] == card_ids.iloc[1]


def _standardize_inputs(extra_txn_cols=None):
    txn = pd.DataFrame(
        [
            {**_txn_row(1), "TransactionDT": 100, "TransactionAmt": 10.0, "ProductCD": "W",
             "P_emaildomain": "gmail.com", "isFraud": 0},
            {**_txn_row(2, addr1=np.nan), "TransactionDT": 200, "TransactionAmt": 20.0, "ProductCD": "C",
             "P_emaildomain": None, "isFraud": 1},
        ]
    )
    for col, values in (extra_txn_cols or {}).items():
        txn[col] = values
    identity = pd.DataFrame({"TransactionID": [1], "DeviceType": ["mobile"], "DeviceInfo": ["iOS"]})
    return txn, identity


def test_standardize_is_noop_without_extension_columns():
    txn, identity = _standardize_inputs()
    out = _standardize(txn, identity, is_synthetic=True)
    assert not any(c[0] in "CDMV" and c[1:].isdigit() for c in out.columns)


def test_standardize_carries_and_fills_extension_columns():
    txn, identity = _standardize_inputs(
        {"C1": [3.0, np.nan], "D1": [np.nan, 7.0], "V5": [np.nan, np.nan], "M4": ["M2", np.nan]}
    )
    out = _standardize(txn, identity, is_synthetic=False)
    assert out["C1"].tolist() == [3.0, 0.0]
    assert out["D1"].tolist() == [0.0, 7.0]
    assert out["V5"].tolist() == [0.0, 0.0]
    assert out["M4"].tolist() == ["M2", "missing"]


def test_effective_feature_lists_append_only_present_extension_columns():
    cfg = {"numeric_features": ["log_amt"], "categorical_features": ["ProductCD"]}

    base_numeric, base_cat = _effective_feature_lists(pd.DataFrame(columns=["log_amt", "ProductCD"]), cfg)
    assert base_numeric == ["log_amt"] and base_cat == ["ProductCD"]

    df = pd.DataFrame(columns=["log_amt", "ProductCD", "C1", "V2", "M3"])
    numeric, cat = _effective_feature_lists(df, cfg)
    assert numeric == ["log_amt", "C1", "V2"]
    assert cat == ["ProductCD", "M3"]
    assert cfg["numeric_features"] == ["log_amt"], "must not mutate the config's own lists"
