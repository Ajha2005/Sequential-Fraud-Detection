import json

import numpy as np
import pandas as pd
import pytest

from sfraud.evaluation.fault_review import BAND_LABELS, assign_bands, extremeness, run

NUM = ["f1", "f2", "f3", "f4"]


def _write_processed(tmp_path, n_fault=400, n_normal=600, seed=0, with_fault=True):
    rng = np.random.default_rng(seed)
    n = n_fault + n_normal
    tid = np.arange(1, n + 1)
    quadrant = np.array(["Fault"] * n_fault + ["Normal"] * n_normal) if with_fault else np.array(["Normal"] * n)
    Z = rng.normal(0, 1, (n, len(NUM)))
    # the highest-scoring Fault rows carry a planted extreme value in f3
    anomaly = np.where(quadrant == "Fault", rng.uniform(1, 2, n), rng.uniform(0, 0.4, n))
    top = np.argsort(-anomaly)[: n_fault // 10]
    Z[top, 2] = 80.0
    pd.DataFrame(Z, columns=NUM).assign(TransactionID=tid).to_parquet(tmp_path / "features_flat.parquet")
    pd.DataFrame({
        "TransactionID": tid, "card_id": [f"c{i % 50}" for i in range(n)], "n_seen": 1 + (tid % 7),
        "TransactionAmt": rng.lognormal(3.5, 1, n).round(2), "isFraud": (rng.random(n) < 0.03).astype(int),
        "fraud_prob": rng.uniform(0, 0.2, n), "anomaly_score": anomaly, "quadrant": quadrant,
    }).to_parquet(tmp_path / "fault_fraud_quadrants.parquet")
    pd.DataFrame({
        "TransactionID": tid, "ProductCD": rng.choice(["W", "C", "R"], n), "email_domain": "gmail.com",
        "device_type": "mobile", "card_type": "visa", "card_category": "debit",
    }).to_parquet(tmp_path / "merged.parquet")
    (tmp_path / "feature_meta.json").write_text(json.dumps({"numeric_features": NUM}))


def _run(tmp_path, **kw):
    return run(cfg={"data": {"processed_dir": str(tmp_path)}}, processed_dir=tmp_path,
               out_sample=tmp_path / "sample.csv", out_summary=tmp_path / "summary.json", **kw)


def test_extremeness_finds_the_most_extreme_feature_with_its_sign():
    flat = pd.DataFrame({"f1": [0.1, -9.0], "f2": [3.0, 0.2], "f3": [-0.5, 1.0], "f4": [0.0, 2.0]},
                        index=pd.Index([10, 20], name="TransactionID"))
    ids = pd.Series([10, 20])
    out = extremeness(flat, list(flat.columns), ids, pd.RangeIndex(2))
    assert out.loc[0, "top1_feature"] == "f2" and out.loc[0, "top1_z"] == 3.0
    assert out.loc[1, "top1_feature"] == "f1" and out.loc[1, "top1_z"] == -9.0  # sign is kept
    assert out["max_abs_z"].tolist() == [3.0, 9.0]
    assert out.loc[1, "top2_feature"] == "f4"


def test_bands_are_rank_based_and_survive_tied_scores():
    bands = assign_bands(pd.Series(np.linspace(0, 1, 1000)))
    assert set(bands.dropna().unique()) == set(BAND_LABELS)
    assert bands.value_counts()["0-50%"] == 500 and bands.value_counts()["99-100%"] == 10
    tied = assign_bands(pd.Series([1.0] * 200))  # identical scores must not collapse the bin edges
    assert tied.notna().all()


def test_run_writes_a_stratified_sample_and_a_summary(tmp_path):
    _write_processed(tmp_path)
    s = _run(tmp_path, per_band=5, seed=1)
    sample = pd.read_csv(tmp_path / "sample.csv")
    assert s["n_fault_rows"] == 400
    assert set(sample["band"]) == set(BAND_LABELS)
    assert sample.groupby("band").size().max() <= 5
    assert sample["anomaly_score"].is_monotonic_decreasing
    assert sample["reviewer_label"].isna().all() and sample["reviewer_note"].isna().all()  # blank columns to fill in
    assert {"top1_feature", "top1_z", "max_abs_z", "ProductCD", "isFraud"} <= set(sample.columns)
    assert json.loads((tmp_path / "summary.json").read_text())["n_fault_rows"] == 400


def test_planted_extreme_values_show_up_in_the_measurements(tmp_path):
    _write_processed(tmp_path)
    s = _run(tmp_path)
    m = s["max_abs_z"]
    assert m["share_fault_gt_50"] == pytest.approx(0.10, abs=0.011)   # the 10% highest-scoring Fault rows
    assert m["share_normal_gt_50"] == 0.0
    assert m["fault_p90"] > m["normal_p90"]
    assert list(s["most_common_top_feature_in_fault"])[0] == "f3"
    top_band = [b for b in s["bands"] if b["band"] == "99-100%"][0]
    assert top_band["share_maxz_gt10"] == 1.0                          # the very top scores are all extreme


def test_sampling_is_deterministic_for_a_seed(tmp_path):
    _write_processed(tmp_path)
    _run(tmp_path, seed=3)
    a = pd.read_csv(tmp_path / "sample.csv")
    _run(tmp_path, seed=3)
    pd.testing.assert_frame_equal(a, pd.read_csv(tmp_path / "sample.csv"))


def test_run_refuses_when_there_is_no_fault_quadrant(tmp_path):
    _write_processed(tmp_path, with_fault=False)
    with pytest.raises(ValueError, match="Fault"):
        _run(tmp_path)
