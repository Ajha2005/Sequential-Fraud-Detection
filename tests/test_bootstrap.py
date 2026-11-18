import json

import numpy as np
import pandas as pd

from sfraud.evaluation.bootstrap import bootstrap_differences, run


def _frame(seed=0, n_cards=250):
    rng = np.random.default_rng(seed)
    cards = np.repeat([f"c{i}" for i in range(n_cards)], rng.integers(1, 6, n_cards))
    y = (rng.random(len(cards)) < 0.1).astype(int)
    return pd.DataFrame({
        "card_id": cards, "isFraud": y, "split": "test",
        "good": 0.8 * y + rng.normal(0, 0.5, len(y)),    # clearly better ranking
        "noise_a": rng.normal(0, 1, len(y)) + 0.3 * y,    # two equally good models with independent noise
        "noise_b": rng.normal(0, 1, len(y)) + 0.3 * y,
    })


PAIRS = [("good", "noise_a"), ("noise_a", "noise_b")]


def test_clearly_better_model_has_a_positive_interval_and_equal_models_straddle_zero():
    res = {(r["a"], r["b"]): r for r in bootstrap_differences(_frame(), PAIRS, n_resamples=120, seed=0)}
    lo, hi = res[("good", "noise_a")]["auc_ci"]
    assert lo > 0 and hi > lo
    lo, hi = res[("noise_a", "noise_b")]["auc_ci"]
    assert lo < 0 < hi
    assert res[("good", "noise_a")]["pr_ci"][0] > 0


def test_resampling_is_by_card_and_deterministic():
    d = _frame()
    a = bootstrap_differences(d, PAIRS, n_resamples=40, seed=5)
    b = bootstrap_differences(d, PAIRS, n_resamples=40, seed=5)
    assert a == b
    assert a != bootstrap_differences(d, PAIRS, n_resamples=40, seed=6)


def test_run_reads_predictions_and_writes_json(tmp_path):
    d = _frame()
    d["xgboost_plus"], d["xgboost_plus_nohist"], d["xgboost"], d["blend"], d["lstm"] = d["good"], d["noise_a"], d["noise_b"], d["good"], d["noise_a"]
    pd.concat([d, d.assign(split="val")]).to_parquet(tmp_path / "preds.parquet")
    res = run(cfg={"data": {"processed_dir": "unused"}}, predictions=tmp_path / "preds.parquet", out=tmp_path / "out.json", n_resamples=20)
    saved = json.loads((tmp_path / "out.json").read_text())
    assert saved["n_resamples"] == 20 and len(saved["differences"]) == 4 == len(res["differences"])
    assert {"auc_ci", "pr_ci", "auc_mean", "pr_mean"} <= set(saved["differences"][0])
