import numpy as np
import pandas as pd
import pytest

from sfraud.evaluation.blend import choose_blend, history_breakdown
from sfraud.evaluation.thresholds import (
    apply_platt, best_f1_threshold, fit_platt, operating_point_metrics, threshold_for_recall,
)


def _scores(seed=0, n=4000, rate=0.05, sep=2.0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    z = rng.normal(0, 1, n) + sep * y
    return y, 1 / (1 + np.exp(-(z - 3.0)))  # deliberately not centred on 0.5


def test_best_f1_threshold_is_perfect_on_separable_scores():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.05, 0.1, 0.2, 0.3, 0.7, 0.9])
    thr = best_f1_threshold(y, p)
    m = operating_point_metrics(y, p, thr)
    assert m["f1"] == 1.0 and m["precision"] == 1.0 and m["recall"] == 1.0
    assert 0.3 < thr <= 0.7


def test_best_f1_beats_a_fixed_half_threshold_when_scores_are_off_centre():
    y, p = _scores()
    f1_half = operating_point_metrics(y, p, 0.5)["f1"]
    f1_best = operating_point_metrics(y, p, best_f1_threshold(y, p))["f1"]
    assert f1_best > f1_half


def test_threshold_for_recall_reaches_target_and_is_the_highest_such_threshold():
    y, p = _scores(seed=1)
    thr = threshold_for_recall(y, p, 0.7)
    assert operating_point_metrics(y, p, thr)["recall"] >= 0.7
    higher = np.sort(np.unique(p[p > thr]))
    if len(higher):
        assert operating_point_metrics(y, p, higher[0])["recall"] < 0.7


def test_operating_point_metrics_counts():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    m = operating_point_metrics(y, p, 0.5)
    assert m["confusion_matrix"] == [[2, 1], [1, 1]]
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["alert_rate"] == pytest.approx(2 / 5)


def test_threshold_selection_requires_both_classes():
    with pytest.raises(ValueError):
        best_f1_threshold(np.zeros(10), np.linspace(0, 1, 10))


def test_platt_calibration_matches_base_rate_on_the_fitting_data():
    y, p = _scores(seed=2)
    params = fit_platt(y, p)
    cal = apply_platt(p, params)
    assert abs(cal.mean() - y.mean()) < 1e-3
    assert np.all((cal > 0) & (cal < 1))
    # monotone: calibration never changes the ranking
    order = np.argsort(p)
    assert np.all(np.diff(cal[order]) >= -1e-12)


def test_choose_blend_puts_weight_on_the_better_model():
    rng = np.random.default_rng(3)
    y = (rng.random(3000) < 0.1).astype(int)
    good = np.clip(0.1 + 0.6 * y + rng.normal(0, 0.15, 3000), 0.001, 0.999)
    noise = rng.random(3000)
    assert choose_blend(y, good, noise, 0.05)["weight_first_model"] >= 0.8
    assert choose_blend(y, noise, good, 0.05)["weight_first_model"] <= 0.2


def test_history_breakdown_buckets_and_counts():
    rng = np.random.default_rng(4)
    n = 600
    test = pd.DataFrame({"n_seen": rng.integers(1, 40, n), "isFraud": (rng.random(n) < 0.15).astype(int)})
    test["m1"] = test["isFraud"] * 0.5 + rng.random(n)
    out = history_breakdown(test, ["m1"], [1, 3, 6, 11, 21, 1000])
    assert out["n_seen_bucket"].tolist() == ["1-2", "3-5", "6-10", "11-20", "21+"]
    assert out["n_transactions"].sum() == n
    assert out["n_fraud"].sum() == test["isFraud"].sum()
    assert (out["m1_auc_roc"].dropna() > 0.5).all()
