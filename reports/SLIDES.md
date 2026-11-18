# Slide Deck Outline -- Sequential Fault & Fraud Detection

Built around the Phase 7 visualizations so the deck doubles as the demo walkthrough. Each slide lists its figure/table source. **All numbers are from the run on the real IEEE-CIS data** (code v5, commit `d393f9c`); full detail and caveats are in `reports/REPORT.md`.

**Walkthrough order:** research questions -> data -> results (sequence models do not beat boosted trees) -> why / history ablation -> thresholds -> fault/fraud -> limitations.

---

### 1. Title
Does Transaction History Help Detect Fraud? A Like-for-Like Evaluation of Sequence Models and Boosted Trees on IEEE-CIS, with an Exploration of Fault/Fraud Separation. *Real-data results, including a negative finding.*

### 2. Research questions
- Fraud models usually score transactions in isolation and treat every anomaly as fraud.
- Q1: Does sequence modelling (LSTM / Transformer over a card's history) beat classical models? Q2: Does card history help at all (same model with vs. without history features)? Q3: What matters most -- features, tuning, or thresholds? Q4: Can **fault** (benign anomaly) be separated from fraud when only `isFraud` is labelled?
- The evaluation is built to answer these fairly: identical rows for every model, a history ablation, thresholds chosen on validation.
- Dataset: IEEE-CIS Fraud Detection (Kaggle), 590,540 transactions, 3.5% fraud.

### 3. Architecture
- Diagram: raw data -> merge -> feature engineering (causal, per-card) -> {XGBoost/LogReg baselines, XGBoost+ with richer features, LSTM / Transformer, Autoencoder anomaly head} -> fault/fraud quadrant -> evaluation (same rows for every model, thresholds chosen on validation).
- Source: `README.md` architecture section / `src/sfraud/` layout.

### 4. Data & the card-identity problem
- Class imbalance: `reports/figures/eda_class_imbalance.png`; history lengths: `reports/figures/eda_sequence_length_hist.png`.
- No user id in IEEE-CIS: `card_id` = `card1/2/3/5` + account start day `floor(TransactionDT/86400 - D1)`. Candidate keys were compared on the real data (core fields alone gave a 14,112-transaction "card").
- Result: 161,453 cards; **at least half have one transaction** (median 1, mean 3.7). Split by card in time order; fraud rate 3.05% train -> 4.97% val / 4.54% test.

### 5. Headline result (Q1): sequence models do not beat boosted trees
- `reports/figures/plot5_benchmark_comparison.png` (all models on the same 60,182 test rows)
- XGBoost+ **0.896 AUC / 0.558 PR-AUC**; original XGBoost 0.884 / 0.538; LSTM 0.853 / 0.474; Transformer 0.850 / 0.471; LogReg 0.849 / 0.398.
- Blend with the LSTM = 0.896 -- adds nothing. The LSTM/Transformer are ~0.04 AUC behind the trees.
- Say plainly: the published 0.90-0.95 figures use different splits and far heavier modelling -- not directly comparable.

### 6. Does card history help? (ablation)
- Same XGBoost+ with and without the 20 history features: 0.8956 vs 0.8930 AUC (+0.003, bootstrap 95% CI -0.001 to +0.006: noise), **0.558 vs 0.567 PR-AUC** (-0.009, CI -0.016 to -0.002: slightly *worse* with history). No gain from history.
- Bootstrap intervals (resampling test cards): XGBoost+ beats the original XGBoost (+0.012 AUC, CI +0.007 to +0.017) and the LSTM (+0.043 AUC, CI +0.034 to +0.051); the blend adds nothing (CI includes zero). Intervals cover test noise only.
- The gain over the original XGBoost came from richer raw features, NaN handling and tuning -- not history.
- `reports/figures/plot1_confidence_vs_n.png`, `reports/metrics/by_history_length_common.csv`: AUC does *not* rise with history for any model.

### 7. Choosing a threshold matters
- A fixed 0.5 is meaningless (LSTM recall 0.21 at 0.5).
- Thresholds chosen on **validation** transfer to test: XGBoost+ at its F1 point = precision 0.58 / recall 0.50; for ~70% recall, precision ~0.28.
- `reports/figures/plot6_confusion_matrix.png` (blend: 1,350 caught, 1,381 missed, 870 false alarms); `reports/metrics/threshold_table.csv`.

### 8. Fault vs. fraud quadrant
- `reports/figures/plot3_quadrant_scatter.png`
- 2x2: LSTM fraud probability x autoencoder anomaly score. Test fraud rate by quadrant: Normal 2.4%, **Fault 5.0%**, Edge case 34%, Fraud 57.5% (base rate 4.5%).
- Be explicit: **no ground truth for "fault"**; the Fault quadrant is ~94% non-fraud and is **not validated as benign**.
- Measured over all 11,687 Fault rows: **96.5% have a feature beyond 5 standard deviations** (Normal rows: 17.5%); two-thirds beyond 10. The anomaly score mostly measures *extreme anonymised feature values*. Fraud rate 6.1% vs 2.5% (Normal), but flat across score bands and only 1.7% in the top 1% -- the most extreme rows are almost all non-fraud outliers.

### 9. Interpretability plots (honest)
- `plot2_prediction_traces.png`: the fault panel is empty on real data, and the example fraud only reaches ~0.24 probability.
- `plot4_attention_heatmap.png`: attention is diffuse, no peak on the fraud transaction.
- `plot7_embedding_tsne.png`: categorical embeddings alone do not separate fraud.

### 10. Limitations
- `card_id` is an approximation (>= half the cards are single transactions; a 1,414-transaction card likely merges accounts).
- LSTM/Transformer are **untuned** (15 epochs, default hyperparameters) and see only each card's first 30 transactions -- the negative result is for this setup, not a proof sequence models cannot help.
- Single run and seed (bootstrap intervals cover test noise only); temporal fraud-rate shift; autoencoder validation loss ~300x training loss, consistent with extreme feature values driving the anomaly scores (clipping not tried).

### 11. Next steps
- Tune the deep models; clip heavy-tailed features (then re-test the anomaly score); repeat runs over several seeds.
- Causal aggregates across other entities (`card1`, `addr1`, email domain), not just same-card history.
- Manually label the 60-row stratified Fault sample (drawn, not yet reviewed).

### 12. Team & deliverables checklist
- Member A: data pipeline, sequence construction, baselines.
- Member B: LSTM/Transformer, autoencoder, evaluation, visualization.
- Checklist from `README.md`.
