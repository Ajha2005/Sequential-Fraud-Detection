import numpy as np

from sfraud.evaluation.metrics import binary_classification_metrics


def test_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.01, 0.02, 0.9, 0.95])
    m = binary_classification_metrics(y_true, y_prob)
    assert m["auc_roc"] == 1.0
    assert m["pr_auc"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]


def test_worst_case_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.9, 0.95, 0.01, 0.02])
    m = binary_classification_metrics(y_true, y_prob)
    assert m["auc_roc"] == 0.0
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0


def test_single_class_returns_nan_auc_without_crashing():
    y_true = np.array([0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3])
    m = binary_classification_metrics(y_true, y_prob)
    assert np.isnan(m["auc_roc"])
    assert m["accuracy"] == 1.0
