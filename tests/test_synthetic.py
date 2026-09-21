from sfraud.data.synthetic import generate_synthetic_tables


def _tiny_cfg(n_cards=25):
    return {
        "data": {
            "synthetic": {
                "n_cards": n_cards,
                "min_txns_per_card": 1,
                "max_txns_per_card": 20,
                "n_days": 30,
                "fraud_rate": 0.035,
                "fault_rate": 0.05,
                "random_state": 0,
            }
        }
    }


def test_shapes_and_columns():
    txn_df, identity_df = generate_synthetic_tables(_tiny_cfg())
    assert txn_df["card_id"].nunique() == 25
    assert set(["TransactionID", "card_id", "TransactionDT", "TransactionAmt", "isFraud"]).issubset(txn_df.columns)
    assert identity_df["TransactionID"].isin(txn_df["TransactionID"]).all()


def test_labels_are_binary():
    txn_df, _ = generate_synthetic_tables(_tiny_cfg())
    assert set(txn_df["isFraud"].unique()).issubset({0, 1})
    assert set(txn_df["synthetic_fault_flag"].unique()).issubset({0, 1})
    # a transaction is never labeled both fraud and fault at once
    assert ((txn_df["isFraud"] == 1) & (txn_df["synthetic_fault_flag"] == 1)).sum() == 0


def test_deterministic_with_fixed_seed():
    txn_df1, _ = generate_synthetic_tables(_tiny_cfg())
    txn_df2, _ = generate_synthetic_tables(_tiny_cfg())
    assert txn_df1["TransactionAmt"].equals(txn_df2["TransactionAmt"])
