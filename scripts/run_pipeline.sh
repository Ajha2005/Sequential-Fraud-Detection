#!/usr/bin/env bash
# Runs the entire Sequential Fault & Fraud Detection pipeline end to end,
# phase by phase, matching Section 4 of the blueprint. Safe to re-run: every
# stage is idempotent (it just overwrites its own output files).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Phase 0: data acquisition ==="
# Best-effort: no Kaggle credentials in this environment is expected, not
# fatal -- every later phase falls back to the synthetic stand-in dataset.
python3 scripts/phase0_get_data.py || echo "(no Kaggle credentials found -- continuing with synthetic stand-in data)"

echo "=== Phase 1: EDA ==="
python3 scripts/phase1_eda.py

echo "=== Phase 2: sequence construction ==="
python3 scripts/phase2_build_sequences.py

echo "=== Phase 3: baselines ==="
python3 scripts/phase3_train_baseline.py

echo "=== Phase 4: LSTM + Transformer ==="
python3 scripts/phase4_train_deep_models.py

echo "=== Phase 5: autoencoder + fault/fraud quadrant ==="
python3 scripts/phase5_fault_fraud.py

echo "=== Phase 6: evaluation & comparison ==="
python3 scripts/phase6_evaluate.py

echo "=== Phase 7: visualization ==="
python3 scripts/phase7_visualize.py

echo "=== Done. See reports/figures, reports/metrics, reports/REPORT.md ==="
