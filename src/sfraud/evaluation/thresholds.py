"""Operating-point selection and probability calibration.

AUC-ROC / PR-AUC measure ranking quality and ignore the decision threshold,
but a confusion matrix, precision and recall all depend on it. A fixed 0.5 is
arbitrary -- with imbalanced classes and a focal/weighted loss the model's
probabilities are not centred there (on real IEEE-CIS a 0.5 cut caught ~21%
of fraud). Thresholds here are always chosen on the VALIDATION split and then
applied unchanged to test, so the test numbers stay an honest estimate.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve


def _check_binary(y_true: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Need both classes present to choose a threshold")
    return y_true


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Score cut-off maximizing F1 (flag when score >= threshold)."""
    y_true = _check_binary(y_true)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    p, r = precision[:-1], recall[:-1]  # the final (1, 0) point has no threshold
    denom = p + r
    f1 = np.divide(2 * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[int(np.argmax(f1))])


def threshold_for_recall(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float) -> float:
    """Highest threshold (i.e. highest precision in practice) whose recall on
    this data is still >= target_recall."""
    y_true = _check_binary(y_true)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ok = recall[:-1] >= target_recall
    if not ok.any():
        return float(thresholds.min())
    return float(thresholds[ok].max())


def operating_point_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alert_rate": (tp + fp) / max(len(y_true), 1),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_platt(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Platt scaling: logistic regression on the logit of the model score.
    Puts scores from differently trained models (a weighted XGBoost, a focal-loss
    LSTM) on a common probability scale before they are averaged."""
    y_true = _check_binary(y_true)
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(_logit(y_prob).reshape(-1, 1), y_true)
    return {"coef": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0])}


def apply_platt(y_prob: np.ndarray, params: dict) -> np.ndarray:
    z = params["coef"] * _logit(y_prob) + params["intercept"]
    return 1.0 / (1.0 + np.exp(-z))
