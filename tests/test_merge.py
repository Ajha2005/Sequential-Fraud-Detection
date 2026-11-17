import numpy as np
import pandas as pd

from sfraud.data.merge import _derive_card_id, _standardize
from sfraud.data.sequences import _effective_feature_lists


def _txn_row(txn_id, **overrides):
    row = {
        "TransactionID": txn_id, "card1": 1111, "card2": 222.0, "card3": 150.0, "card4": "visa",
        "card5": 226.0, "card6": "debit", "addr1": 315.0, "addr2": 87.0,
        # Account start day = floor(TransactionDT / 86400 - D1) = 10 - 3 = 7 by default.
        "TransactionDT": 10 * 86400, "D1": 3.0,
    }
    row.update(overrides)
    return row


def test_card_id_different_core_fields_do_not_collide():
    """Regression (original bug): a naive astype(str) turned every missing
    addr1 into the shared literal "nan", merging unrelated cardholders --
    different core card fields -- into one mega-card."""
    df = pd.DataFrame([_txn_row(1, card1=1111, addr1=np.nan), _txn_row(2, card1=2222, addr1=np.nan)])
    card_ids = _derive_card_id(df)
    assert card_ids.iloc[0] != card_ids.iloc[1]
    assert "nan" not in card_ids.iloc[0]


def test_card_id_ignores_soft_fields_for_same_core_identity():
    """Regression (overcorrection bug): an earlier version made *every*
    missing field unique per row -- including low-signal fields like addr1,
    which is blank on ~11% of real rows -- which fragmented a single
    person's own history into singletons the moment one of their
    transactions had a missing address, and crushed the real-data median
    sequence length from 2 to 1. card4/card6/addr1/addr2 are excluded from
    the identity key entirely (see _derive_card_id's docstring for why a
    missing-vs-present placeholder scheme can't work here), so rows that
    agree on the core numeric fields (card1/card2/card3/card5) collide
    regardless of what those soft fields are doing -- missing, present, or
    disagreeing outright."""
    df = pd.DataFrame(
        [_txn_row(1, addr1=np.nan, card4="visa"), _txn_row(2, addr1=315.0, card4="visa"),
         _txn_row(3, addr1=999.0, card4="mastercard")]
    )
    card_ids = _derive_card_id(df)
    assert card_ids.iloc[0] == card_ids.iloc[1] == card_ids.iloc[2]


def test_card_id_missing_core_field_does_not_collide():
    """A missing *core* field (card1/card2/card3/card5) is a genuine loss of
    identifying signal -- unlike a missing soft field, it must still isolate
    the row rather than guess at a match."""
    df = pd.DataFrame([_txn_row(1, card1=np.nan), _txn_row(2, card1=np.nan)])
    card_ids = _derive_card_id(df)
    assert card_ids.iloc[0] != card_ids.iloc[1]


def test_card_id_same_account_across_time_collides():
    """The same account transacting on different days has a larger D1 later, so
    TransactionDT/86400 - D1 (the account start day) stays constant."""
    df = pd.DataFrame(
        [_txn_row(1, TransactionDT=10 * 86400, D1=3.0), _txn_row(2, TransactionDT=15 * 86400, D1=8.0)]
    )
    card_ids = _derive_card_id(df)
    assert card_ids.iloc[0] == card_ids.iloc[1]


def test_card_id_different_accounts_in_same_card_bucket_do_not_collide():
    """Regression (real-data finding): card1/2/3/5 alone identify an issuer or
    product bucket, so thousands of different cardholders (the largest group
    was 14,112 transactions) shared one id. A different account start day
    must split them even when every core field matches."""
    df = pd.DataFrame([_txn_row(1, D1=3.0), _txn_row(2, D1=0.0)])
    card_ids = _derive_card_id(df)
    assert card_ids.iloc[0] != card_ids.iloc[1]


def test_card_id_missing_d1_does_not_collide():
    """A missing D1 means the account start day is unknown: isolate the row,
    never merge it with another missing-D1 row or with a present-D1 row."""
    df = pd.DataFrame([_txn_row(1, D1=np.nan), _txn_row(2, D1=np.nan), _txn_row(3)])
    card_ids = _derive_card_id(df)
    assert len(set(card_ids)) == 3
    assert not any("nan" in c for c in card_ids)


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
    # Synthetic-style input: it ships its own card_id and has no D1 (D1 is an
    # extension column, and real data needs it only to derive card_id).
    txn, identity = _standardize_inputs()
    txn = txn.drop(columns=["D1"]).assign(card_id=["a", "b"])
    out = _standardize(txn, identity, is_synthetic=True)
    assert not any(c[0] in "CDMV" and c[1:].isdigit() for c in out.columns)


def test_standardize_carries_extension_columns_and_keeps_numeric_nan():
    """C/D/V stay NaN (XGBoost uses the missingness itself; sequences.py zero-fills
    a copy for the dense models). Categorical M columns are still filled."""
    txn, identity = _standardize_inputs(
        {"C1": [3.0, np.nan], "D1": [np.nan, 7.0], "V5": [np.nan, np.nan], "M4": ["M2", np.nan]}
    )
    out = _standardize(txn, identity, is_synthetic=False)
    assert out["C1"].isna().tolist() == [False, True] and out["C1"].iloc[0] == 3.0
    assert out["D1"].isna().tolist() == [True, False] and out["D1"].iloc[1] == 7.0
    assert out["V5"].isna().all()
    assert out["M4"].tolist() == ["M2", "missing"]


def test_standardize_carries_extra_raw_columns_for_boosted_models():
    txn, identity = _standardize_inputs({"dist1": [12.0, np.nan], "R_emaildomain": ["gmail.com", None]})
    identity = identity.assign(**{"id-01": [-5.0], "id-30": ["Windows 10"]})  # test-file style dashed names
    out = _standardize(txn, identity, is_synthetic=False)
    for col in ["card1", "card2", "card3", "card5", "addr2", "dist1", "R_emaildomain", "id_01", "id_30"]:
        assert col in out.columns, col
    assert out["id_01"].iloc[0] == -5.0 and np.isnan(out["id_01"].iloc[1])
    assert out["id_30"].iloc[0] == "Windows 10"


def test_effective_feature_lists_append_only_present_extension_columns():
    cfg = {"numeric_features": ["log_amt"], "categorical_features": ["ProductCD"]}

    base_numeric, base_cat = _effective_feature_lists(pd.DataFrame(columns=["log_amt", "ProductCD"]), cfg)
    assert base_numeric == ["log_amt"] and base_cat == ["ProductCD"]

    df = pd.DataFrame(columns=["log_amt", "ProductCD", "C1", "V2", "M3"])
    numeric, cat = _effective_feature_lists(df, cfg)
    assert numeric == ["log_amt", "C1", "V2"]
    assert cat == ["ProductCD", "M3"]
    assert cfg["numeric_features"] == ["log_amt"], "must not mutate the config's own lists"
