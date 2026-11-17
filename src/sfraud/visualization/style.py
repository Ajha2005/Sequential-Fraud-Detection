"""One shared palette so every figure in reports/figures/ (EDA included)
reads as one consistent set rather than ad hoc per-script colors."""

COLOR_NORMAL = "#4C72B0"
COLOR_FRAUD = "#C44E52"
COLOR_FAULT = "#DD8452"
COLOR_EDGE_CASE = "#8172B2"
COLOR_NEUTRAL = "#55A868"

CATEGORY_COLORS = {
    "Normal": COLOR_NORMAL,
    "Fraud": COLOR_FRAUD,
    "Fault": COLOR_FAULT,
    "Edge case": COLOR_EDGE_CASE,
}

MODEL_COLORS = {
    "logreg": "#8C8C8C",
    "xgboost": "#4C72B0",
    "xgboost_plus_nohist": "#9DB4D8",
    "xgboost_plus": "#1F3A6E",
    "lstm": "#C44E52",
    "transformer": "#55A868",
    "blend": "#8172B2",
}

MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "xgboost": "XGBoost (baseline)",
    "xgboost_plus_nohist": "XGBoost+ (no history)",
    "xgboost_plus": "XGBoost+",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "blend": "Blend (XGBoost+ & LSTM)",
}
