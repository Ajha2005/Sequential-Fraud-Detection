# Sequential Fault & Fraud Detection

Predicting fraud probability from evolving transaction histories, with fault/fraud separation.

Fraud models usually score each transaction in isolation and treat every anomaly as fraud. This project asks whether a cardholder's **transaction history** helps: it compares sequence models (an LSTM, with a Transformer variant) against gradient-boosted trees with and without history features, on the real IEEE-CIS data under a like-for-like evaluation, and explores separating flagged transactions into **fraud** (malicious) vs. **fault** (a non-malicious anomaly -- a technical error, a mistyped detail, a legitimate but unusual purchase) with a combined supervised + anomaly-detection design. On the real data the sequence models do not beat the boosted trees; the write-up reports the results and the reasons we could and could not establish.

Dataset: [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle, public).

**Full write-up with results, figures, and honest limitations: [reports/REPORT.md](reports/REPORT.md).**
**Slide deck outline: [reports/SLIDES.md](reports/SLIDES.md).**

> **Results status.** The numbers in this README and in [reports/REPORT.md](reports/REPORT.md) come from a run on the **real IEEE-CIS data** (590,540 transactions), done in a Kaggle notebook with code version **v5** (commit `0b13e61`). The figures, metrics and samples under `reports/` and `reports/SLIDES.md` are from that same run. A synthetic stand-in generator (`synthetic.py`) remains for tests and as a fallback when no real CSVs are present; model checkpoints are gitignored.

## Architecture

```
raw Kaggle CSVs  ---\
                      >-- merge.py --> standardized transactions --> engineering.py (causal, per-user features)
synthetic.py     ---/                                                        |
                                                                              v
                                                            sequences.py (pad/truncate/split by card+time)
                                                                              |
                        +-----------------------------+-----------------------------+
                        v                                                           v
              features_flat.parquet                                    sequences_{train,val,test}.npz
                        |                                                           |
          +-------------+-------------+                     +----------------------+----------------------+
          v                           v                     v                      v                      v
      XGBoost                  Logistic Regr.            LSTM                Transformer            Autoencoder
   (models/baseline.py)     (models/baseline.py)      (models/lstm.py)   (models/transformer.py)  (models/autoencoder.py,
                                                                                                     normal-only, per-txn)
          \___________________________|_____________________|______________________/                      |
                                       v                                                                    v
                          evaluation/compare.py, error_analysis.py                        evaluation/fault_fraud.py
                       (benchmarks, accuracy-vs-N, stratification)                    (fraud-prob x anomaly-score quadrant)
                                       \_______________________________________________________/
                                                              v
                                              visualization/plots.py (7 required + 1 bonus plot)
```

## Repository layout

```
configs/config.yaml         All hyperparameters and paths -- the single source of truth every script reads.
src/sfraud/
  data/                     download.py (real Kaggle), synthetic.py (stand-in generator), merge.py, eda.py, sequences.py
  features/engineering.py   Causal per-user feature engineering (amount deviation, categorical stability, time features)
  datasets/torch_dataset.py PyTorch Dataset/DataLoader (sequences) + flat per-transaction dataset (autoencoder)
  models/                   layers.py (shared embeddings), baseline.py, lstm.py, transformer.py, autoencoder.py
  training/                 One train_*.py per model, sharing engine.py (masked loss, train/eval loop)
  evaluation/               metrics.py, predict.py (shared prediction plumbing), compare.py, fault_fraud.py, error_analysis.py
  visualization/            plots.py (all 7 required plots + 1 bonus), style.py (shared palette)
scripts/                    Thin phase0..phase7 CLI entry points + run_pipeline.sh to run everything in order
tests/                      pytest unit tests (feature engineering causality, padding, model shapes, causal masking, metrics)
reports/
  figures/                  All generated PNGs (EDA + the 7+1 Phase 7 plots)
  metrics/                  All generated JSON/CSV metrics
  samples/                  Qualitative review samples (e.g. the Fault-quadrant sample for manual inspection)
  REPORT.md, SLIDES.md      Phase 8 write-up and slide outline
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
pip install -e .        # makes `sfraud` importable everywhere (editable install)
```

For running tests too: `pip install -r requirements-dev.txt`.

## Running the pipeline

End to end, matching the blueprint's Phase 0-7 breakdown:

```bash
bash scripts/run_pipeline.sh
```

Or phase by phase (each is idempotent -- safe to re-run, it just overwrites its own output):

```bash
python3 scripts/phase0_get_data.py           # Phase 0: real data if Kaggle creds exist, else synthetic fallback later
python3 scripts/phase1_eda.py                # Phase 1: merge + EDA plots/stats -> reports/figures, reports/metrics
python3 scripts/phase2_build_sequences.py    # Phase 2: feature engineering + padded sequences -> data/processed
python3 scripts/phase3_train_baseline.py     # Phase 3: XGBoost + Logistic Regression
python3 scripts/phase3b_train_xgb_plus.py    # Phase 3b: stronger XGBoost (richer features, tuned) + its no-history ablation
python3 scripts/phase4_train_deep_models.py  # Phase 4: LSTM + Transformer (stretch goal)
python3 scripts/phase5_fault_fraud.py        # Phase 5: autoencoder anomaly head + fault/fraud quadrant logic
python3 scripts/phase6_evaluate.py           # Phase 6: benchmark comparison, accuracy-vs-N, stratification, error analysis
python3 scripts/phase6b_blend_thresholds.py  # Phase 6b: same-rows comparison, validation-chosen blend and thresholds
python3 scripts/phase7_visualize.py          # Phase 7: all plots
python3 scripts/review_fault_quadrant.py     # Optional, after Phase 5: characterize the Fault quadrant on real data + a stratified random sample for manual review
python3 scripts/draft_fault_labels.py        # Optional, after review_fault_quadrant.py: rule-based DRAFT labels for the sampled Fault rows (not ground truth)
python3 scripts/bootstrap_differences.py     # Optional, after Phase 6b: bootstrap 95% CIs (resampling test cards) for model differences
```

### What Phases 3b and 6b add

- **Phase 3b** trains `xgboost_plus` on `features_xgb.parquet` (built in Phase 2 by `features/xgb_features.py`): C/D/V kept as NaN plus per-family missing counts, extra raw columns (`card1/2/3/5`, `addr1/2`, `dist1/2`, `R_emaildomain`, `id_*`), train-only frequency encodings, and causal per-card history (prior-transaction aggregates, rolling 1h/1d/7d counts and spend, "changed since last transaction" flags). It uses early stopping and a small grid scored on validation PR-AUC. `xgboost_plus_nohist` is the same model with every history-derived feature removed -- the fair test of whether card history helps.
- **Phase 6b** scores every model on the **same** validation/test rows (the sequence models only cover each card's first 30 transactions, so per-model numbers in Phase 6 are over slightly different rows), Platt-calibrates two models on validation and picks the blend weight on validation PR-AUC, and chooses decision thresholds on **validation** (F1-optimal, plus the highest-precision threshold reaching each target recall) before reporting them on test. Outputs: `reports/metrics/blend_and_thresholds.json`, `threshold_table.csv`, `by_history_length_common.csv`.
- The fault/fraud quadrant and error analysis use the validation-chosen fraud threshold (`fault_fraud.fraud_prob_threshold: auto`) rather than a fixed 0.5.

Everything is controlled by `configs/config.yaml` (sequence length, model hyperparameters, thresholds, train/val/test fractions, `seed: 42`). Change it there rather than passing flags -- every script reads the same file.

### Tests

```bash
pip install -r requirements-dev.txt
python3 -m pytest tests/ -v
```

Covers: causal feature engineering (no leakage into a transaction's own history), padding/truncation/split correctness, model output shapes, **the Transformer's causal mask actually blocking future information** (a perturbation test, not just a shape check), and metric correctness on known inputs.

## Plugging in the real Kaggle dataset

1. Create an API token at [kaggle.com/settings](https://www.kaggle.com/settings) -> "Create New Token" (downloads `kaggle.json`).
2. Place it at `~/.kaggle/kaggle.json` (`chmod 600`), or export `KAGGLE_USERNAME`/`KAGGLE_KEY`.
3. Accept the [competition rules](https://www.kaggle.com/c/ieee-fraud-detection/rules) once, in the browser (required before the API will allow the download).
4. `python3 scripts/phase0_get_data.py` -- downloads `train_transaction.csv` and `train_identity.csv` into `data/raw/`.
5. Re-run the pipeline from Phase 1 onward. `merge.py` auto-detects the real files and stops generating synthetic data -- no other code changes needed.

## Design choices worth knowing before reading the code

- **Padding/truncation:** a card's transactions are right-padded (real data first, padding after), and a card with more than `max_seq_len` (30) transactions is truncated to its *first* 30. This makes "sequence position" mean "number of transactions seen so far (N)" consistently across every card, which the accuracy-vs-N analysis (Phase 6) depends on.
- **Split:** by card, not by row -- cards are ordered by first-transaction time and cut into contiguous train/val/test blocks, so no card leaks across splits and train broadly precedes val/test in time.
- **`card_id` for real data** is a heuristic, since IEEE-CIS has no explicit user identifier: `card1, card2, card3, card5` plus the account start day `floor(TransactionDT/86400 - D1)`. `card1/2/3/5` alone identify an issuer/product bucket (the largest group was 14,112 transactions), and adding `addr1` over-fragments because it is blank on ~11% of rows; the candidate keys were compared on the real data (see `_derive_card_id`'s docstring). Rows missing a core field or `D1` become single-transaction cards.
- **`merchant_category`** doesn't exist in the real schema; it's derived deterministically from `ProductCD` (documented in `data/merge.py`) so the feature is well-defined for both real and synthetic data.
- **The autoencoder is per-transaction, not per-sequence** -- deliberately, to avoid the trivial-reconstruction problem a same-step recurrent autoencoder runs into (it would partially reconstruct its own input from a hidden state that already saw that input). See `models/autoencoder.py`'s docstring.
- **The Transformer is hand-rolled**, not `nn.TransformerEncoder`, specifically so per-layer attention weights can be extracted for the attention-heatmap plot -- PyTorch's built-in encoder discards them internally.

## Results summary

On the **real IEEE-CIS data** (full details and every caveat in [reports/REPORT.md](reports/REPORT.md)). Every model is scored on the same 60,182 test transactions (2,731 fraud, 4.5%); the split is by card in time order, so train precedes val/test:

| Model | Test AUC-ROC | Test PR-AUC |
|---|---|---|
| Logistic Regression | 0.849 | 0.398 |
| XGBoost (original baseline) | 0.884 | 0.538 |
| XGBoost+ (no history features) | 0.893 | **0.567** |
| **XGBoost+ (with history features)** | **0.896** | 0.558 |
| LSTM | 0.853 | 0.474 |
| Transformer | 0.850 | 0.471 |
| Blend (0.85 XGBoost+ + 0.15 LSTM) | 0.896 | 0.560 |

What this does and doesn't show:

- **Sequence models do not beat boosted trees on this data.** The LSTM is about 0.04 AUC behind XGBoost+, and a blend with the LSTM adds essentially nothing (0.8957 vs 0.8956 AUC). (On the earlier synthetic data the LSTM was best -- that was a property of the synthetic generator, not of real fraud.)
- **Card history did not help the boosted model.** XGBoost+ with history is +0.003 AUC (bootstrap 95% CI -0.001 to +0.006, includes zero) and -0.009 PR-AUC (CI -0.016 to -0.002, excludes zero) versus the same model without it: no gain, and slightly worse on PR-AUC. The gain over the original XGBoost (+0.012 AUC, CI +0.007 to +0.017; +0.020 PR-AUC, CI +0.008 to +0.030) is real and came from richer raw features, NaN handling and tuning, not from history. The blend with the LSTM adds nothing (both intervals include zero), and the LSTM is clearly worse than XGBoost+ (-0.043 AUC, CI -0.051 to -0.034). The intervals cover test-sampling noise only (models are not retrained).
- **The Fault quadrant mostly reflects extreme feature values.** Over all 11,687 Fault rows, 96.5% have a feature beyond 5 standard deviations (Normal rows: 17.5%) and two-thirds beyond 10. Its fraud rate is 6.1% vs 2.5% for Normal, but the highest-scoring rows are almost all non-fraud outliers. In a 60-row stratified sample only 1 row was unusual for its own card (the rest are unusual in anonymised V/D/C columns, with one extreme V pattern recurring on six different cards). Whether they are benign "faults" is not established (no ground truth).
- **Thresholds matter.** At a fixed 0.5 the LSTM catches only ~21% of fraud. With a threshold chosen on validation (F1-optimal), XGBoost+ reaches test precision 0.58 / recall 0.50 (F1 0.54); asking for ~70% recall costs precision (~0.28).
- **Card identity is approximate.** IEEE-CIS has no user id; of the 161,453 derived cards at least half have a single transaction (median history 1, mean 3.7), which limits what any sequence model can learn.

## Team split (from the blueprint)

| Role | Owns | Phases |
|---|---|---|
| Member A -- Data & Baseline | Data merging, sequence construction, DataLoader, train/test split, XGBoost/LogReg baseline | 1-3 |
| Member B -- Deep Model & Fault/Fraud | LSTM/Transformer fraud head, autoencoder anomaly head, evaluation, visualizations, report/slides | 4-8 |

## Final deliverables checklist

- [x] Working data pipeline (raw data -> padded transaction sequences)
- [x] Trained baseline (XGBoost + Logistic Regression) with recorded metrics
- [x] Trained LSTM fraud model + Transformer stretch goal
- [x] Trained autoencoder anomaly head + fault/fraud quadrant logic
- [x] Accuracy/AUC-vs-N curve
- [x] Annotated per-user prediction traces (fraud and normal cases; the fault panel is empty on real data, which has no fault labels)
- [x] Fault/fraud quadrant scatter plot
- [x] Baseline vs. deep learning vs. published-benchmark comparison chart
- [x] Confusion matrix + bonus embedding visualization
- [x] Final written report + slide deck outline (both updated to the real-data results)
- [x] Re-run on real Kaggle data (Kaggle notebook, code v5 / commit `0b13e61`)
- [x] Real-run figures, metrics and samples committed under `reports/`
- [x] Bootstrap confidence intervals for the model differences (`scripts/bootstrap_differences.py`)
- [x] Characterize the Fault quadrant over all its rows (`scripts/review_fault_quadrant.py`)
- [x] Review the 60-row stratified Fault sample (`reports/samples/fault_review_sample.csv`): a rule-based draft reading is recorded (22 benign-looking / 6 suspicious / 32 unclear) -- heuristics over the available fields, not ground truth
- [ ] A person confirms or overrides the draft labels (there is no ground truth for "fault" on real data)
