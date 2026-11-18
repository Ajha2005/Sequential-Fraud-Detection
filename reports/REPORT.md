# Sequential Fault & Fraud Detection -- Project Report

*Predicting fraud probability from evolving transaction histories, with fault/fraud separation.*

> **Read this first.**
> - **Data and code:** every number in this report comes from one run of the full pipeline on the **real IEEE-CIS Fraud Detection data** (590,540 transactions), in a CPU-only Kaggle notebook, with code version **v5** (commit `0b13e61`).
> - **Headline:** on this data, sequence models do **not** beat gradient-boosted trees. Trees with richer features are the best models, adding card-history features to them did not help, and the LSTM/Transformer are behind (Sections 1, 4 and 6).
> - **Artifacts:** the figures, metrics and samples under `reports/` are the outputs of this same run and match the numbers below (model checkpoints are gitignored and not committed).
> - **Statistical caution:** bootstrap 95% intervals (resampling test cards, Section 6.1) separate real differences from noise: XGBoost+ over the original XGBoost, and the boosted model over the LSTM, are real; the blend over XGBoost+ is not. The intervals cover test-sampling noise only, not training randomness (single run, single seed).

## 1. Problem statement and research questions

Fraud models usually score each transaction in isolation and treat every anomaly as fraud. A cardholder's earlier transactions could in principle help (velocity, category drift, deviation from the user's own baseline), and some flagged transactions may be harmless oddities ("faults": a technical error, a mistyped detail, a legitimate but unusual purchase) rather than fraud. On the real IEEE-CIS data, with an evaluation designed to be fair, this project asks:

1. **Does sequence modelling help?** Do an LSTM and a Transformer over a card's history beat classical models on the same transactions?
2. **Does card history help at all?** Does the same boosted-tree model do better with history-derived features than without them?
3. **What matters most for accuracy?** Richer features, missing-value handling, tuning, or the choice of decision threshold?
4. **Can "fault" be separated from fraud** with a combined supervised + anomaly-detection design, given that the dataset only labels `isFraud` and "fault" has no ground truth?

The evaluation is built to answer them fairly: identical rows for every model, a history ablation, and thresholds chosen on validation rather than on test.

**Answers in brief** (under these conditions: approximate card ids, untuned deep models, a single run and seed):

| Question | Answer | Evidence |
|---|---|---|
| 1. Does sequence modelling help? | No -- the LSTM (0.853 AUC) and Transformer (0.850) trail XGBoost+ (0.896) | Section 4 |
| 2. Does card history help? | No -- with vs. without history is +0.003 AUC (95% CI -0.001 to +0.006, includes zero) and -0.009 PR-AUC (CI -0.016 to -0.002): no gain, and slightly worse on PR-AUC | Section 6.1 |
| 3. What matters most? | Richer raw features, NaN handling and tuning (about +0.012 AUC over the original XGBoost); and the threshold (the LSTM catches ~21% of fraud at 0.5, ~43% at its validation-chosen threshold) | Sections 4 and 6 |
| 4. Can fault be separated from fraud? | Only weakly -- the Fault quadrant has about 2.4x the fraud rate of Normal (6.1% vs 2.5%, validation + test) but is ~94% non-fraud, and the anomaly score mostly measures extreme feature values (96.5% of Fault rows have a feature beyond 5 standard deviations, vs 17.5% of Normal); not validated as benign | Section 5 |

## 2. Dataset

**Dataset:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle, public): `train_transaction.csv` and `train_identity.csv`, joined on `TransactionID`, binary `isFraud` label, and no explicit user/card identifier.

**This run:** 590,540 transactions, fraud rate **3.50%**. Missing values: `card_category` 1,571, `addr1` 65,706. Of the 414 extension and extra raw columns (C/D/V, identity fields, distances, ...), 399 have missing values (mean missing fraction 45.7%, max 99.2%).

**Cards are derived, not given.** IEEE-CIS has no user id, so `card_id` is a heuristic: `card1, card2, card3, card5` plus the account start day `floor(TransactionDT/86400 - D1)` (`src/sfraud/data/merge.py`, `_derive_card_id`). The candidate keys were compared on this data -- `card1/2/3/5` alone produced a 14,112-transaction "card" (an issuer/product bucket, not a person), and adding `addr1` over-fragments because it is blank on ~11% of rows. The chosen key gives:

- **161,453 cards**: 149,167 "real" cards plus 12,286 rows missing a core field or `D1`, each treated as its own single-transaction card;
- history length per card: mean 3.66, **median 1**, p90 8, p95 13, max 1,414;
- among the 149,167 real cards: median 2, p90 9, p99 32; the 10 largest cards hold 0.7% of rows.

So **at least half of all cards have a single transaction**, which limits how much any sequence model can exploit. This is an approximation -- there is no ground truth for who is who (Section 7).

![Class imbalance](figures/eda_class_imbalance.png)

A synthetic stand-in generator (`src/sfraud/data/synthetic.py`) produces the same schema and is used for tests and as a fallback when no real CSVs are present. It was used for the earlier development runs and is not used for any number in this report.

## 3. Methodology

### 3.1 Features

**Shared per-transaction features** (`src/sfraud/features/engineering.py`). Every feature that depends on a card's own history is **causal** -- at transaction *t* it only uses transactions 1..*t*-1 for that card:

| Feature | Meaning |
|---|---|
| `log_amt`, `log_time_delta` | log1p of amount and of seconds since the card's previous transaction |
| `amt_dev`, `amt_zscore` | current `log_amt` vs. the causal expanding mean/std of the card's *prior* amounts |
| `txn_hour`, `txn_dow` | time-of-day / day-of-week |
| `stab_ProductCD`, `stab_device_type`, `stab_email_domain` | does this transaction match the card's most common *prior* value for that field? |

Plus the engineered/anonymized C1-C14, D1-D15, V1-V339 (numeric, 368 columns) and M1-M9 (categorical) columns. The sequence models and logistic regression see 377 numeric features (9 base + 368 extension, z-scored on the train split, missing values filled with 0) and 15 categorical ones (6 base + 9 extension).

**Gradient-boosted table** (`src/sfraud/features/xgb_features.py`, 497 features). Trees can use much more of the raw data, so XGBoost+ gets its own table built from the same card-level split:

- C/D/V values stay NaN (the missingness pattern is informative and XGBoost handles NaN), plus per-family missing-value counts;
- extra raw columns: `card1/2/3/5`, `addr1/2`, `dist1/2`, `R_emaildomain`, `id_01..id_38`;
- frequency encodings and frequency-ranked categorical codes computed on **train rows only**;
- **20 history-derived features**, all causal: log-time-since-previous, `amt_dev`, `amt_zscore`, the transaction's position in the card's history, the `stab_*` flags, plus prior-transaction mean/std/max amount, time since the card's first transaction, rolling 1h/1d/7d transaction counts and 1d spend, and "changed since the previous transaction" flags. These were verified against a brute-force reference and by a perturbation test (changing a later transaction changes no earlier row's features).

### 3.2 Sequence construction (`src/sfraud/data/sequences.py`)

- Transactions are grouped by `card_id`, sorted chronologically, and **right-padded** to length 30; a card with more than 30 transactions is truncated to its *first* 30. This drops 35,531 of the 590,540 transactions (6.0%) -- those beyond the 30th position of cards with more than 30 transactions -- from the sequence models. The tabular models are trained on all rows.
- **Split:** cards (not rows) are ordered by first-transaction time and cut into contiguous 70/15/15 train/val/test blocks. No card spans two splits and train broadly precedes val/test in time:

| Split | Cards | Transactions (sequences) | Fraud rate |
|---|---|---|---|
| train | 113,017 | 425,401 | 3.05% |
| val | 24,218 | 69,426 | 4.97% |
| test | 24,218 | 60,182 | 4.54% |

The fraud rate rises from train to val/test, so the evaluation includes some temporal distribution shift.

### 3.3 Models

| Model | File | Role |
|---|---|---|
| Logistic Regression | `models/baseline.py` | simplest baseline |
| XGBoost (original baseline) | `models/baseline.py` | single-transaction baseline on the shared features |
| **XGBoost+** | `training/train_xgb_plus.py` | XGBoost on the 497-feature table, early stopping and a small grid scored on validation PR-AUC (picked depth 6, column sample 0.4; stopped at 558 trees) |
| XGBoost+ (no history) | `training/train_xgb_plus.py` | the same model and hyperparameters with all 20 history-derived features removed -- the fair test of whether history helps |
| LSTM | `models/lstm.py` | sequence model; outputs a fraud probability at every step, focal loss, 210,449 parameters |
| Transformer | `models/transformer.py` | hand-rolled causal self-attention encoder, focal loss, 108,817 parameters |
| Autoencoder | `models/autoencoder.py` | per-transaction anomaly head for fault detection (Section 3.4) |
| Blend | `evaluation/blend.py` | 0.85 XGBoost+ + 0.15 LSTM after Platt calibration on validation; weight chosen on validation PR-AUC |

Both neural sequence models are strictly causal (the Transformer via a triangular attention mask), verified in `tests/test_models.py`. They were trained for a fixed 15 epochs with the default hyperparameters and **were not tuned**.

### 3.4 Fault vs. fraud design

Two independent scores are combined per transaction: the LSTM's fraud probability, and an autoencoder anomaly score (reconstruction loss of a model trained only on `isFraud==0` rows, relative to the 90th percentile of normal rows' scores).

| Fraud probability | Anomaly score | Category |
|---|---|---|
| Low | Low | Normal |
| Low | High | **Fault** |
| High | High | **Fraud** |
| High | Low | Edge case (flagged for manual review) |

"High" fraud probability uses a threshold **chosen on the validation split** (the F1-optimal one, 0.2649 for the LSTM) instead of a fixed 0.5.

## 4. Model comparison

Every model is scored on the **same 60,182 test transactions (2,731 fraud)**. The sequence models only cover each card's first 30 transactions, so scoring every model on the intersection is the like-for-like comparison.

| Model | Test AUC-ROC | Test PR-AUC |
|---|---|---|
| Logistic Regression | 0.8487 | 0.3975 |
| XGBoost (original baseline) | 0.8837 | 0.5376 |
| XGBoost+ (no history features) | 0.8930 | **0.5668** |
| **XGBoost+ (with history features)** | **0.8956** | 0.5580 |
| LSTM | 0.8528 | 0.4744 |
| Transformer | 0.8502 | 0.4711 |
| Blend (0.85 XGBoost+ + 0.15 LSTM) | 0.8957 | 0.5604 |

![Benchmark comparison](figures/plot5_benchmark_comparison.png)

(Scored on each model's own test rows in the earlier phases: XGBoost 0.8792 / 0.5287, XGBoost+ 0.8938 / 0.5519, XGBoost+ without history 0.8926 / 0.5606, LogReg 0.8452 / 0.3961 -- the sequence models' numbers are identical.)

What this shows:

- **The richer tree model beats the original baseline** by about +0.012 AUC and +0.02 to +0.03 PR-AUC.
- **The sequence models do not beat the trees.** The LSTM and Transformer are about 0.04 AUC and 0.08-0.09 PR-AUC behind XGBoost+. They beat only logistic regression.
- **The blend is no better than XGBoost+ alone** (0.8957 vs 0.8956 AUC; the LSTM gets 15% of the weight): the LSTM carries almost nothing the boosted model lacks.
- The published reference points the project originally set (strong supervised baselines 0.90-0.93 AUC, Kaggle winners about 0.945) were obtained on different splits and with far heavier modelling, so they are **not directly comparable** with these card-level time-split numbers.

**Decision thresholds.** AUC and PR-AUC ignore the threshold, but precision and recall depend on it, and a fixed 0.5 is meaningless across these models: at 0.5 the LSTM has precision 0.88 and recall 0.21, the Transformer 0.85 / 0.24, and the original XGBoost (trained with a 31x positive weight, which inflates its probabilities) precision 0.18 / recall 0.77. Thresholds were therefore chosen on **validation** and applied unchanged to test:

| Model | F1-optimal: precision / recall / F1 | ~70% recall target: precision / recall / F1 |
|---|---|---|
| Logistic Regression | 0.380 / 0.443 / 0.409 | 0.184 / 0.674 / 0.289 |
| XGBoost (original) | 0.545 / 0.501 / 0.522 | 0.248 / 0.699 / 0.366 |
| XGBoost+ (no history) | 0.613 / 0.497 / 0.549 | 0.290 / 0.696 / 0.410 |
| XGBoost+ | 0.575 / 0.502 / 0.536 | 0.284 / 0.695 / 0.403 |
| LSTM | 0.491 / 0.435 / 0.461 | 0.196 / 0.675 / 0.304 |
| Transformer | 0.415 / 0.474 / 0.443 | 0.188 / 0.671 / 0.293 |
| Blend | 0.608 / 0.494 / 0.545 | 0.291 / 0.692 / 0.410 |

Validation-chosen recall targets transferred well to test (target 70% gave 0.67-0.70). Operationally, XGBoost+ at its F1-optimal point flags about half the fraud at roughly 58% precision, and pushing to ~70% recall roughly halves the precision (about 0.28).

![Confusion matrix](figures/plot6_confusion_matrix.png)

Plot 6 shows the blend at its validation-chosen threshold (0.301) on the test split: 56,581 true negatives, 870 false positives, 1,381 missed fraud and 1,350 caught (precision 0.61, recall 0.49).

## 5. Fault/fraud results

On the held-out val+test transactions (129,608 rows), using the LSTM's validation-chosen fraud threshold (0.2649) and the 90th-percentile anomaly threshold (0.518):

| Quadrant | Count |
|---|---|
| Normal | 112,750 |
| Fault | 11,687 |
| Fraud | 3,397 |
| Edge case | 1,774 |

32.7% of true fraud lands in the "Fraud" quadrant and 89.1% of true non-fraud lands in "Normal".

**There is no ground truth for "fault" on the real data.** The synthetic data had an injected `synthetic_fault_flag`; the real data has none, so the fault-recovery numbers the earlier report quoted do not exist here. What can be measured is the fraud rate inside each quadrant on the test split (derived from the error-analysis breakdown; base rate 4.54%):

| Quadrant (test) | Transactions | Fraud rate |
|---|---|---|
| Normal | 52,000 | 2.4% |
| Fault | 5,762 | 5.0% |
| Edge case | 873 | 34.0% |
| Fraud | 1,547 | 57.5% |

The anomaly score carries a modest extra signal -- the Fault quadrant's fraud rate is about twice the Normal quadrant's -- but it is still 95% non-fraud, and **whether those transactions are genuinely benign anomalies is unverified**.

![Fault/fraud quadrant](figures/plot3_quadrant_scatter.png)

Plot 3 shows the four quadrants (anomaly score on a log axis). Fraud probability rises with anomaly score, and the Fault quadrant's anomaly scores reach into the thousands, far above the 0.518 cut.

**What the Fault quadrant contains, measured over all 11,687 rows** (`scripts/review_fault_quadrant.py`; `reports/metrics/fault_review_summary.json`; validation + test). For each transaction, the largest standardized feature value (the largest |z| among the 377 numeric features, standardized on the training split) was compared with a random sample of 20,000 Normal transactions:

| | Fault (11,687 rows) | Normal (20,000-row sample) |
|---|---|---|
| Median of the largest \|z\| | 11.8 | 3.1 |
| 90th percentile of the largest \|z\| | 25.1 | 6.0 |
| Share with a feature beyond 5 standard deviations | 96.5% | 17.5% |
| ... beyond 10 | 66.7% | 1.1% |
| ... beyond 50 | 1.0% | 0.0% |
| Fraud rate | 6.1% | 2.5% |
| Amount, median / 90th percentile | $73 / $527 | $68 / $303 |
| ProductCD W / C / R / S / H | 46.9 / 27.5 / 12.0 / 7.8 / 5.9 % | 76.2 / 12.8 / 5.7 / 1.1 / 4.2 % |

- **The anomaly score mostly measures extreme feature values.** 96.5% of Fault rows have at least one feature beyond 5 standard deviations (two-thirds beyond 10), against 17.5% of Normal rows. Within the quadrant, the share beyond 10 standard deviations rises from 45% in the lowest score band to 100% in the top two bands (table below).
- **No single mechanism.** The ten most frequent top-driving features (D14, D7, V300, V108, V107, V313, V114, V146, V141, D12) together account for 29% of the rows and none for more than 3.8%; they are anonymised time-delta (D) and Vesta (V) columns.
- **Not concentrated on a few cards.** The Fault rows spread over 6,434 cards, and the ten largest hold 1.7% of the rows.
- **A different mix.** Fault rows over-represent ProductCD C, R and S and have a heavier tail of large amounts (90th percentile $527 vs $303).
- **The score does not track fraud.** The fraud rate is flat across the first four score bands (about 6%) and drops to 1.7% in the top 1% (117 rows, median amount $200): the highest-scoring Fault transactions are almost entirely non-fraud extreme outliers.

| Score band (within Fault) | Rows | Score range | Fraud rate | Median amount | Share with a feature beyond 10 sd |
|---|---|---|---|---|---|
| 0-50% | 5,844 | 0.52-0.88 | 5.8% | $59 | 45% |
| 50-80% | 3,505 | 0.88-1.96 | 7.1% | $79 | 81% |
| 80-95% | 1,753 | 1.96-7.03 | 5.6% | $97 | 98% |
| 95-99% | 468 | 7.03-16.6 | 5.8% | $77 | 100% |
| 99-100% | 117 | 16.6-4,827 | 1.7% | $200 | 100% |

**The 20 highest-scoring rows** (`reports/samples/fault_quadrant_review_sample.csv`) are the most extreme 0.2% of the quadrant, so they are not representative (the full-quadrant figures above are). Inspecting them:

- All 20 are non-fraud, with low fraud probabilities (0.08-0.20) and anomaly scores from 94 to 4,827 (the cut is 0.518).
- They come from only 9 cards (one card has 6 of the 20, two have 3 each).
- Every amount is a whole-dollar figure between $110 and $800 (many of them round, such as 200, 500, 600, 800), and several cards repeat the same amount (for example card `11538_...` has $800 three times in its first five transactions).

Repeated whole-dollar purchases on the same card with no fraud label are *consistent with* benign unusual spending, and the full-quadrant measurements show why those rows score so high: they sit at the extreme end of the score range, where nearly every row has features tens of standard deviations out.

**Review of the 60-row stratified random sample** (12 rows from each of the five score bands; `reports/samples/fault_review_sample.csv`). The sample's own fields show:

- **Almost no behavioural anomalies.** The top-driving feature is an anonymised V column in 51 of the 60 rows (85%), a D column in 6, a C column in 2, and a behavioural feature -- an unusual amount for the card -- in just **1** (a $2.58 purchase on a card with larger earlier amounts). Nothing else in the sample is unusual relative to its own card's history; the rows are unusual in anonymised feature space.
- **Shared extreme patterns.** 10 of the 60 rows share an identical top-driving value with another sampled row, and one exact V-column pattern (V107 at -45.8, together with V117 and V119) recurs on **six different cards**. Much of the extremeness comes from a few recurring rare combinations of anonymised values, not from individual cards behaving oddly.
- **Not concentrated.** The 60 rows come from 58 distinct cards.
- **Little history.** 13 rows are a card's first transaction and 22 are its first or second, so "unusual for this card" has little meaning for them.
- **The top band** (99-100%) is 9 of 12 ProductCD S with an unknown email domain, amounts $30-$800, and feature values up to 468 standard deviations out. One row (a hotmail.fr email) has no extreme numeric value at all (largest |z| 1.7), so its score probably comes from the categorical fields (not verified).
- **Near-misses.** 11 rows have a fraud probability of at least 0.20 against the 0.265 threshold; the single fraud-labelled row (a first transaction, $67) scored 0.211.

**Draft labels.** A rule-based draft reading (`scripts/draft_fault_labels.py`; **not ground truth**): *suspicious* = labelled fraud, or a behavioural driver, or a fraud probability within 0.035 of the threshold; *benign-looking* = an established card (at least 4 transactions), fraud probability below 0.15, and extremeness only in anonymised columns; *unclear* = everything else, mainly first and second transactions.

| Score band | Rows | Benign-looking | Suspicious | Unclear |
|---|---|---|---|---|
| 0-50% | 12 | 1 | 3 | 8 |
| 50-80% | 12 | 6 | 0 | 6 |
| 80-95% | 12 | 3 | 2 | 7 |
| 95-99% | 12 | 4 | 0 | 8 |
| 99-100% | 12 | 8 | 1 | 3 |
| **All** | **60** | **22 (37%)** | **6 (10%)** | **32 (53%)** |

"Benign-looking" means *no red flag in the available fields*, not verified benign: in the top band it means the fraud model sees nothing and the card has history, not that values hundreds of standard deviations out were judged harmless. The shares by band partly mirror the rules (the benign rule requires a low fraud probability, which is lower in the top band), so they are not independent evidence, and with 60 rows each share is uncertain by roughly 12 percentage points. A person should confirm or override the labels.

**What this does and does not establish.** The Fault quadrant is a set of transactions with extreme anonymised feature values that the LSTM rates as low fraud risk; its fraud rate is 2.4 times that of Normal rows (6.1% vs 2.5%), so extremeness carries some risk signal, but it is still about 94% non-fraud and the most extreme rows are almost all non-fraud. That is *not* evidence that these transactions are behavioural "faults" (technical errors, mistyped details): an extreme value in an anonymised column and a large amount could be legitimate or not, and there is no label to say. The sample review above found almost no behavioural anomalies and many recurring extreme patterns shared across cards, which fits that reading; the draft labels are a starting point for the judgement, not a substitute for a person's.

**Autoencoder caveat (now supported by the measurements).** Its validation loss (about 92-149 over training) is roughly 300 times its training loss (0.32-0.57). The Fault-quadrant measurements above fit the explanation that some validation rows have extreme values relative to the train-fitted standardization -- the heavy-tailed C/D/V columns are z-scored but not clipped -- and that the anomaly score largely reflects those outliers. Clipping the features before the autoencoder was not tried.

## 6. Analysis

### 6.1 Does card history help?

**No -- it gives no gain and slightly lowers PR-AUC.** The cleanest test is the ablation -- the same XGBoost+ with and without the 20 history-derived features:

| | Test AUC-ROC | Test PR-AUC | Val PR-AUC |
|---|---|---|---|
| With history | 0.8956 | 0.5580 | 0.6104 |
| Without history | 0.8930 | 0.5668 | 0.6165 |

History is +0.003 AUC but -0.009 PR-AUC on test, and the no-history model is also slightly better on validation PR-AUC. The bootstrap below shows the AUC difference is within noise (its interval includes zero) while the PR-AUC deficit is outside it: the history features do not help and, if anything, hurt PR-AUC by a small amount. The improvement over the original XGBoost came from the richer raw features, NaN handling and tuning, not from history.

**Bootstrap confidence intervals** (`scripts/bootstrap_differences.py`; `reports/metrics/bootstrap_differences.json`). Test cards were resampled with replacement 200 times (a card's transactions together) and the difference in AUC-ROC and PR-AUC recomputed; the interval is the 2.5th-97.5th percentile:

| Comparison (a minus b) | AUC-ROC difference [95% CI] | PR-AUC difference [95% CI] | Reading |
|---|---|---|---|
| XGBoost+ minus XGBoost+ (no history) | +0.0026 [-0.0006, +0.0059] | -0.0092 [-0.0156, -0.0017] | No AUC gain; PR-AUC slightly worse |
| XGBoost+ minus original XGBoost | +0.0120 [+0.0073, +0.0174] | +0.0202 [+0.0084, +0.0303] | Real improvement |
| Blend minus XGBoost+ | +0.0001 [-0.0011, +0.0015] | +0.0024 [-0.0002, +0.0059] | No gain |
| XGBoost+ minus LSTM | +0.0427 [+0.0338, +0.0509] | +0.0830 [+0.0650, +0.0992] | LSTM clearly worse |

These intervals reflect sampling noise in the test set only: the models are fixed, so training randomness (the seed, the early-stopping point) is not included, and the true uncertainty is larger.

**AUC by how much history was available** (test, AUC-ROC; "n_seen" = the transaction's position in its card's history; full table, including PR-AUC and all models, in `reports/metrics/by_history_length_common.csv`):

| n_seen | Transactions | Fraud | XGBoost+ | XGBoost+ (no history) | LSTM | Blend |
|---|---|---|---|---|---|---|
| 1-2 | 33,578 | 1,289 | 0.912 | 0.912 | 0.871 | 0.911 |
| 3-5 | 12,496 | 662 | 0.880 | 0.876 | 0.835 | 0.881 |
| 6-10 | 7,529 | 422 | 0.875 | 0.866 | 0.836 | 0.875 |
| 11-20 | 5,047 | 286 | 0.861 | 0.855 | 0.814 | 0.860 |
| 21+ | 1,532 | 72 | 0.837 | 0.840 | 0.790 | 0.841 |

AUC **falls** as history grows, for every model including the one that cannot see history. So this is not evidence about history's usefulness; the longer-history transactions come from a different (and harder) population, and the last bucket has only 72 fraud cases. The LSTM-only stratification by a card's total history length shows the same non-improving pattern (AUC 0.892, 0.850, 0.808, 0.839, 0.825 for 1-2, 3-5, 6-10, 11-20, 21+ transactions). The earlier synthetic result -- AUC climbing with history -- was a property of the synthetic generator.

![AUC and accuracy vs. number of transactions seen](figures/plot1_confidence_vs_n.png)

Plot 1 (AUC-ROC within each position N, all models on the test split) shows the same thing: AUC drifts down from about 0.9 at N=1 and becomes very noisy beyond N≈20 as the samples shrink, with no upward trend for any model, sequence or not. The right-hand panel is accuracy at a fixed 0.5 threshold, which is misleading here: logistic regression and the original XGBoost look far worse (accuracy 0.6-0.8) only because their positive class weighting pushes probabilities above 0.5 -- the other models sit near 0.95-0.98 simply because 95% of transactions are non-fraud. It does not mean they rank better.

### 6.2 Error analysis (LSTM, test, validation-chosen threshold 0.2649)

True positives 1,187, false positives 1,233, false negatives 1,544, true negatives 56,218 (precision 0.49, recall 0.43).

- False negatives (missed fraud) occur earlier in a card's history (mean position 5.0) than false positives (6.9); 1,255 of the 1,544 sit in the Normal quadrant and 289 in the Fault quadrant.
- False positives split 657 into the Fraud quadrant and 576 into Edge case.
- Missed fraud and false alarms both have higher mean amounts (about $187-188) than correctly caught fraud (about $127).

### 6.3 Training behaviour and calibration

- LSTM and Transformer validation metrics are noisy from epoch to epoch (LSTM val AUC 0.857-0.875, PR-AUC 0.46-0.51) with training loss still falling slowly, so they are neither clearly converged nor stable. With a fixed 15 epochs and default hyperparameters, their gap to XGBoost+ may partly reflect that; it was not tested.
- Models are not calibrated to a common scale: the F1-optimal threshold is 0.22 for XGBoost+, 0.26 for the LSTM, and 0.83 for the original XGBoost (inflated by its positive-class weight). The blend therefore calibrates each model on validation first.
- Validation scores are optimistic relative to test (XGBoost+ PR-AUC 0.61 on validation vs 0.56 on test) because validation was used for early stopping, the hyperparameter grid, calibration and thresholds, and because the fraud rate shifts over time.

### 6.4 Interpretability plots (2, 4, 7)

These plots were designed to illustrate the synthetic data's injected structure and **do not support the sequence-model story on the real data**:

- **Plot 2 (prediction traces):** the "fault" panel is empty ("no example found") because it relies on the synthetic fault label, which real data does not have. The fraud example (card `12695_490.0_150.0_226.0_155.0`, one fraud at position 12) shows the LSTM's probability rising only to about 0.24 at the fraud transaction, below the plot's dotted 0.5 line -- i.e. a missed or at best borderline case, not the clean spike the synthetic data produced. (The dotted line is a hard-coded 0.5, not the validation-chosen operating threshold of 0.2649; at that threshold this fraud sits just under the cut.)
- **Plot 4 (Transformer attention, same card):** the last-layer attention is diffuse across prior transactions, heaviest on the earliest positions, with no visible concentration on the fraudulent transaction. The "attends to the first fraudulent transaction" interpretation from the synthetic run does not appear here.
- **Plot 7 (t-SNE of the LSTM's categorical embeddings):** points (2,407 normal, 93 fraud, 0 fault) form many tight clusters that appear to reflect category combinations; fraud points are scattered across clusters, so the categorical embeddings alone show no clean fraud separation.

## 7. Limitations

- **The confidence intervals cover test-sampling noise only.** The bootstrap over test cards (Section 6.1) separates real differences from noise, but the models are not retrained, so seed and early-stopping variation are not included. Test fraud counts are modest (2,731 overall, 72 in the largest-history bucket), so the per-history-bucket numbers in particular are noisy and have no intervals.
- **`card_id` is an approximation.** At least half of derived cards have a single transaction, 12,286 rows are isolated because a key field is missing, and the largest derived card has 1,414 transactions (probably several accounts opened the same day in one card bucket). History features and sequence models are only as good as this grouping.
- **The sequence models are untuned and use only the first 30 transactions per card.** The negative result for them is for these hyperparameters and this card grouping, not a proof that sequence models cannot help on this data.
- **"Fault" has no ground truth.** The Fault quadrant was characterised over all its rows (Section 5): its anomaly scores mostly reflect extreme anonymised feature values, and whether the transactions are benign is not established. A stratified 60-row sample has only a rule-based draft reading (Section 5), not a human review. Plot 2's fault panel and its 0.5 reference line also need rework for real data. The anomaly score is per-transaction (a deliberate design choice, see `models/autoencoder.py`), so it cannot detect anomalies that only appear as a pattern across several transactions.
- **Temporal shift:** the fraud rate is 3.05% in train and 4.5-5.0% in val/test; thresholds chosen on validation transfer reasonably (recall targets landed close to target) but are not guaranteed to on future data.
- **`merchant_category`** does not exist in the real schema; it is derived deterministically from `ProductCD` (`data/merge.py: PRODUCT_TO_MERCHANT_CATEGORY`).
- **Single run, single seed (`seed: 42`)**, CPU-only; no repeated runs.

## 8. Reproducibility

```bash
pip install -r requirements.txt
pip install -e .
bash scripts/run_pipeline.sh
```

Every stage reads its parameters from `configs/config.yaml` and writes to `reports/figures/`, `reports/metrics/`, and `checkpoints/`. `seed: 42` is set once and threaded through numpy/random/torch (`src/sfraud/utils/seed.py`). The real-data run used a Kaggle notebook: the competition data attached as an input (`/kaggle/input/competitions/ieee-fraud-detection/`), `train_transaction.csv` and `train_identity.csv` linked into `data/raw/`, then Phases 1, 2, 3, 3b, 4, 5, 6, 6b and 7 in order (Phase 0, the Kaggle API download, is not needed inside Kaggle). Phase 3b is slow on CPU (a four-model grid plus two final fits); `xgboost_plus.search.enabled: false` in the config skips the grid. See [README.md](../README.md) for the phase-by-phase breakdown.
