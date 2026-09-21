"""Phase 3: classical single-transaction baselines (XGBoost, Logistic
Regression). These operate on the same engineered per-transaction features
as the sequence models (log_amt, time_delta, the causal amt_dev/amt_zscore
"rolling average spend" features, categorical fields) but ignore sequence
order -- they exist to prove the sequence model adds value over classical ML
(Section 6 of the blueprint)."""

from __future__ import annotations

from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_xgboost(cfg: dict, scale_pos_weight: float) -> XGBClassifier:
    bcfg = cfg["baseline"]["xgboost"]
    return XGBClassifier(
        n_estimators=bcfg["n_estimators"],
        max_depth=bcfg["max_depth"],
        learning_rate=bcfg["learning_rate"],
        subsample=bcfg["subsample"],
        colsample_bytree=bcfg["colsample_bytree"],
        scale_pos_weight=scale_pos_weight * bcfg["scale_pos_weight_multiplier"],
        eval_metric="aucpr",
        tree_method="hist",
        random_state=cfg["seed"],
        n_jobs=-1,
    )


def build_logreg_pipeline(cfg: dict, numeric_cols: list[str], categorical_idx_cols: list[str]) -> Pipeline:
    lcfg = cfg["baseline"]["logreg"]
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_idx_cols),
        ]
    )
    clf = LogisticRegression(max_iter=lcfg["max_iter"], class_weight=lcfg["class_weight"])
    return Pipeline([("preprocess", preprocess), ("clf", clf)])
