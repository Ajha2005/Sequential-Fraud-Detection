# Sequential Fault & Fraud Detection

Predicting fraud probability from evolving transaction histories, with fault/fraud separation.

Traditional fraud detection scores each transaction in isolation and treats every anomaly as fraud. This project instead models a cardholder's transactions as an ordered **sequence** (LSTM, with a Transformer variant for comparison), predicting fraud probability continuously as new transactions arrive, and separates flagged transactions into **fraud** (malicious) vs. **fault** (a non-malicious anomaly -- a technical error, a mistyped detail, a legitimate but unusual purchase) using a combined supervised + anomaly-detection design.

Dataset: [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle, public).

**Full write-up with results, figures, and honest limitations: [reports/REPORT.md](reports/REPORT.md).**
**Slide deck outline: [reports/SLIDES.md](reports/SLIDES.md).**

> This repository has no Kaggle credentials attached to it, so every result currently in `reports/` was generated on a **synthetic stand-in dataset** with the same schema as the real one (see [Plugging in the real Kaggle dataset](#plugging-in-the-real-kaggle-dataset)). The code path is identical either way -- only the data source changes.

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
python3 scripts/phase4_train_deep_models.py  # Phase 4: LSTM + Transformer (stretch goal)
python3 scripts/phase5_fault_fraud.py        # Phase 5: autoencoder anomaly head + fault/fraud quadrant logic
python3 scripts/phase6_evaluate.py           # Phase 6: benchmark comparison, accuracy-vs-N, stratification, error analysis
python3 scripts/phase7_visualize.py          # Phase 7: all plots
```

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
- **`card_id` for real data** is the standard community heuristic (`card1..card6 + addr1 + addr2`, concatenated) since IEEE-CIS has no explicit user identifier.
- **`merchant_category`** doesn't exist in the real schema; it's derived deterministically from `ProductCD` (documented in `data/merge.py`) so the feature is well-defined for both real and synthetic data.
- **The autoencoder is per-transaction, not per-sequence** -- deliberately, to avoid the trivial-reconstruction problem a same-step recurrent autoencoder runs into (it would partially reconstruct its own input from a hidden state that already saw that input). See `models/autoencoder.py`'s docstring.
- **The Transformer is hand-rolled**, not `nn.TransformerEncoder`, specifically so per-layer attention weights can be extracted for the attention-heatmap plot -- PyTorch's built-in encoder discards them internally.

## Results summary

On the synthetic stand-in dataset (full details, all 8 plots, and every caveat in [reports/REPORT.md](reports/REPORT.md)):

| Model | Test AUC-ROC | Test PR-AUC |
|---|---|---|
| Logistic Regression | 0.984 | 0.896 |
| XGBoost | 0.989 | 0.927 |
| **LSTM** | **0.995** | **0.961** |
| Transformer | 0.991 | 0.943 |

Both sequence models beat the classical XGBoost baseline, and AUC-ROC on the LSTM climbs from 0.991 (users with 3-5 transactions of history) to 0.998 (21+) -- direct evidence the model benefits from longer histories, which is the core thesis of the project.

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
- [x] Annotated per-user prediction traces (fraud, fault, normal cases)
- [x] Fault/fraud quadrant scatter plot
- [x] Baseline vs. deep learning vs. published-benchmark comparison chart
- [x] Confusion matrix + bonus embedding visualization
- [x] Final written report + slide deck outline
- [ ] Re-run on real Kaggle data (requires Kaggle credentials this environment doesn't have -- see above)
