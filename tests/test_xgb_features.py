import numpy as np
import pandas as pd
import pytest

from sfraud.data.merge import _standardize
from sfraud.data.sequences import assign_splits
from sfraud.data.synthetic import generate_synthetic_tables
from sfraud.features.engineering import engineer_features
from sfraud.features.xgb_features import (
    HISTORY_WINDOWS, build_xgb_table, compute_history_features, frequency_encode, label_encode,
)

BASE_NUMERIC = ["log_amt", "log_time_delta", "amt_dev", "amt_zscore", "txn_hour", "txn_dow",
                "stab_ProductCD", "stab_device_type", "stab_email_domain"]
BASE_CAT = ["ProductCD", "card_type", "card_category", "merchant_category", "device_type", "email_domain"]
SPLIT_CFG = {"sequence": {"train_frac": 0.6, "val_frac": 0.2, "test_frac": 0.2}}


def _history_df(seed=0, n_cards=8):
    rng = np.random.default_rng(seed)
    parts = []
    for c in range(n_cards):
        n = int(rng.integers(1, 26))
        dt = np.sort(rng.integers(0, 20 * 86400, n))
        dt[rng.random(n) < 0.15] = dt[0]  # force some duplicate timestamps; re-sort keeps order valid
        dt = np.sort(dt)
        parts.append(pd.DataFrame({
            "card_id": f"card_{c:02d}", "TransactionDT": dt,
            "TransactionAmt": np.round(rng.lognormal(3.5, 1.0, n), 2),
            "ProductCD": rng.choice(["W", "C"], n), "email_domain": rng.choice(["a.com", "b.com"], n),
            "device_type": rng.choice(["mobile", "desktop"], n), "addr1": rng.choice([300.0, 310.0, np.nan], n),
        }))
    return pd.concat(parts, ignore_index=True)


def _brute_force(df):
    out = []
    for i, r in df.iterrows():
        prior = df[(df["card_id"] == r["card_id"]) & (df.index < i)]
        first_dt = df[df["card_id"] == r["card_id"]]["TransactionDT"].iloc[0]
        row = {
            "hist_time_since_first": r["TransactionDT"] - first_dt,
            "hist_amt_mean": prior["TransactionAmt"].mean() if len(prior) >= 1 else np.nan,
            "hist_amt_std": prior["TransactionAmt"].std(ddof=1) if len(prior) >= 2 else np.nan,
            "hist_amt_max": prior["TransactionAmt"].max() if len(prior) >= 1 else np.nan,
        }
        for name, w in HISTORY_WINDOWS.items():
            in_win = prior[prior["TransactionDT"] >= r["TransactionDT"] - w]
            row[f"cnt_{name}"] = len(in_win)
            if name == "1d":
                row["amt_sum_1d"] = in_win["TransactionAmt"].sum()
        for col in ["ProductCD", "email_domain", "device_type", "addr1"]:
            prev = df[(df["card_id"] == r["card_id"]) & (df.index == i - 1)]
            row[f"chg_prev_{col}"] = float(len(prev) == 1 and str(prev[col].iloc[0]) != str(r[col]))
        out.append(row)
    return pd.DataFrame(out)


def test_history_features_match_brute_force():
    df = _history_df()
    got = compute_history_features(df)
    expected = _brute_force(df)
    for col in expected.columns:
        np.testing.assert_allclose(got[col].to_numpy(dtype=float), expected[col].to_numpy(dtype=float),
                                   rtol=1e-4, atol=1e-3, equal_nan=True, err_msg=col)


def test_first_transaction_has_no_history():
    df = _history_df()
    got = compute_history_features(df)
    first = ~df["card_id"].duplicated()
    assert got.loc[first, ["hist_amt_mean", "hist_amt_std", "hist_amt_max"]].isna().all().all()
    assert (got.loc[first, ["cnt_1h", "cnt_1d", "cnt_7d", "amt_sum_1d", "hist_time_since_first"]] == 0).all().all()


def test_history_features_are_causal():
    """Perturbation test: changing a card's LAST transaction must not change any
    feature of any other row (no leakage from the current/future transaction)."""
    df = _history_df(seed=1)
    last = df.index[df["card_id"] == "card_03"][-1]
    before = compute_history_features(df)
    df2 = df.copy()
    df2.loc[last, "TransactionAmt"] *= 1000
    df2.loc[last, "ProductCD"] = "ZZZ"
    after = compute_history_features(df2)
    pd.testing.assert_frame_equal(before.drop(index=last), after.drop(index=last))


def test_history_features_reject_unsorted_input():
    df = _history_df()
    interleaved = df.iloc[np.random.default_rng(0).permutation(len(df))].reset_index(drop=True)
    with pytest.raises(ValueError):
        compute_history_features(interleaved)
    backwards = df.copy()
    backwards["TransactionDT"] = backwards["TransactionDT"].iloc[::-1].to_numpy()
    with pytest.raises(ValueError):
        compute_history_features(backwards)


def test_frequency_and_label_encoding_use_train_rows_only():
    df = pd.DataFrame({
        "x": ["a", "a", "b", "c", "c", "c", "d", np.nan],
        "split": ["train", "train", "train", "train", "val", "test", "val", "train"],
    })
    mask = (df["split"] == "train").to_numpy()
    freq = frequency_encode(df, ["x"], mask)["freq_x"].to_numpy()
    # train rows: a, a, b, c, NaN -> n_train = 5
    np.testing.assert_allclose(freq, [0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0])

    codes = label_encode(df, ["x"], mask)["code_x"].to_numpy()
    assert codes[0] == codes[1] == 0                       # most common train value ranks first
    assert codes[6] == -1                                  # 'd' never seen in train
    assert np.isnan(codes[7])                              # missing stays missing for XGBoost
    assert {codes[2], codes[3]} == {1.0, 2.0}


def _real_shaped_frames(n_cards=30, seed=0):
    rng = np.random.default_rng(seed)
    rows, tid = [], 0
    for c in range(n_cards):
        start_day = int(rng.integers(0, 20))
        days = np.sort(rng.integers(start_day + 1, start_day + 60, int(rng.integers(1, 12))))
        for d in days:
            tid += 1
            rows.append({
                "TransactionID": tid, "TransactionDT": int(d * 86400 + rng.integers(0, 86400)),
                "TransactionAmt": float(np.round(rng.lognormal(3.5, 1.0), 2)),
                "ProductCD": str(rng.choice(["W", "C", "H"])), "card1": 1000 + c, "card2": 111.0, "card3": 150.0,
                "card4": "visa", "card5": 226.0, "card6": "debit",
                "addr1": 300.0 if rng.random() > 0.2 else np.nan, "addr2": 87.0,
                "P_emaildomain": str(rng.choice(["gmail.com", "yahoo.com"])),
                "R_emaildomain": "gmail.com" if rng.random() > 0.5 else None,
                "dist1": float(rng.integers(0, 100)) if rng.random() > 0.5 else np.nan,
                "C1": float(rng.integers(0, 5)), "D1": float(d - start_day),
                "V1": 1.0 if rng.random() > 0.4 else np.nan,
                "M1": ["T", "F", None][int(rng.integers(0, 3))],
                "isFraud": int(rng.random() < 0.1),
            })
    txn = pd.DataFrame(rows)
    ident_ids = txn["TransactionID"].sample(frac=0.5, random_state=seed).to_numpy()
    identity = pd.DataFrame({
        "TransactionID": ident_ids, "DeviceType": "mobile", "DeviceInfo": "iOS",
        "id_01": np.where(rng.random(len(ident_ids)) > 0.3, -5.0, np.nan),
        "id_30": np.where(rng.random(len(ident_ids)) > 0.5, "Windows 10", None),
    })
    return txn, identity


def _engineered(txn, identity, synthetic, categorical):
    df = _standardize(txn, identity, is_synthetic=synthetic)
    df, _ = engineer_features(df, categorical)
    return assign_splits(df, SPLIT_CFG)


def test_build_xgb_table_on_real_shaped_data():
    txn, identity = _real_shaped_frames()
    df = _engineered(txn, identity, False, BASE_CAT + ["M1"])
    table, meta = build_xgb_table(df, BASE_NUMERIC, BASE_CAT)

    assert len(table) == len(df)
    assert len(meta["features"]) == len(set(meta["features"])) == meta["n_features"]
    assert set(meta["features"]) <= set(table.columns)
    assert set(meta["history_features"]) <= set(meta["features"])

    # NaN is preserved (not zero-filled) and summarized as a missing count
    assert table["V1"].isna().any() and table["dist1"].isna().any()
    np.testing.assert_array_equal(table["n_missing_V"].to_numpy(), table["V1"].isna().to_numpy().astype(np.float32))
    # extra raw columns + identity fields made it through
    for col in ["card1", "addr1", "dist1", "id_01", "code_id_30", "freq_id_30", "code_M1", "freq_card1",
                "code_R_emaildomain", "freq_R_emaildomain"]:
        assert col in table.columns, col

    history = set(meta["history_features"])
    assert {"hist_amt_mean", "cnt_1d", "amt_dev", "txn_index_in_card", "stab_ProductCD", "chg_prev_ProductCD"} <= history
    assert not ({"log_amt", "TransactionAmt", "card1", "V1", "freq_card1", "n_missing_V"} & history)

    # frequency encodings come from train rows only: cards are split-disjoint
    # and card1 is unique per card, so non-train cards were never seen in train.
    assert (table.loc[table["split"] != "train", "freq_card1"] == 0).all()
    assert (table.loc[table["split"] == "train", "freq_card1"] > 0).all()
    assert (table.loc[table["txn_index_in_card"] == 0, "hist_time_since_first"] == 0).all()


def test_build_xgb_table_works_on_the_synthetic_path():
    cfg = {"data": {"synthetic": {"n_cards": 25, "min_txns_per_card": 1, "max_txns_per_card": 20,
                                   "n_days": 30, "fraud_rate": 0.035, "fault_rate": 0.05, "random_state": 0}}}
    txn, identity = generate_synthetic_tables(cfg)
    df = _engineered(txn, identity, True, BASE_CAT)
    table, meta = build_xgb_table(df, BASE_NUMERIC, BASE_CAT)
    assert len(table) == len(df) and meta["n_history_features"] > 0
    # the synthetic identity table has id_01..id_05 but no C/D/V columns
    assert {f for f in meta["features"] if f.startswith("n_missing_")} <= {"n_missing_id"}
    assert np.isfinite(table[meta["features"]].to_numpy(dtype=float)[~table[meta["features"]].isna().to_numpy()]).all()
