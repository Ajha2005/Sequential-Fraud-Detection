# Slide Deck Outline -- Sequential Fault & Fraud Detection

Built around the Phase 7 visualizations so the deck doubles as the demo walkthrough (Phase 8). Each slide lists its figure/table source so it can be dropped straight into slides.

**Rehearsed walkthrough order:** architecture -> benchmark comparison -> accuracy-vs-N chart -> fault/fraud quadrant plot -> annotated prediction traces -> attention heatmap.

---

### 1. Title
Sequential Fault & Fraud Detection -- Predicting Fraud Probability from Evolving Transaction Histories, with Fault/Fraud Separation.

### 2. Problem statement
- Traditional fraud detection scores transactions in isolation and treats every anomaly as fraud.
- We model a cardholder's transactions as a sequence and predict fraud probability continuously.
- We additionally separate flagged transactions into **fraud** (malicious) vs. **fault** (benign anomaly) -- the dataset only labels `isFraud`.
- Dataset: IEEE-CIS Fraud Detection (Kaggle). *Numbers in this deck are on our synthetic stand-in dataset -- see backup slide.*

### 3. Architecture
- Diagram: raw data -> merge -> feature engineering (causal, per-user) -> padded sequences -> {XGBoost/LogReg baseline} + {LSTM / Transformer fraud head} + {Autoencoder anomaly head} -> fault/fraud quadrant -> evaluation.
- Source: `README.md` architecture section / `src/sfraud/` package layout.

### 4. Data & EDA
- Class imbalance chart: `reports/figures/eda_class_imbalance.png`
- Sequence length histogram (justifies max_seq_len=30): `reports/figures/eda_sequence_length_hist.png`
- Key stat: ~4% fraud rate, 3,000 cards, 92.7k transactions, zero missing values.

### 5. Benchmark comparison
- `reports/figures/plot5_benchmark_comparison.png`
- Headline: LSTM (0.997 AUC-ROC) and Transformer (0.996) both beat the XGBoost floor (0.991) and the published reference range (0.90-0.93), on synthetic data.
- One sentence on the synthetic-data caveat -- do not skip this on the actual day.

### 6. Accuracy vs. number of transactions seen
- `reports/figures/plot1_confidence_vs_n.png`
- Better story slide: history-length stratification table (`reports/metrics/stratify_by_history_length.csv`) -- AUC-ROC climbs from 0.987 (3-5 transactions of history) to 0.9997 (21+) -- direct evidence the model uses history.

### 7. Fault vs. fraud quadrant
- `reports/figures/plot3_quadrant_scatter.png`
- Explain the 2x2: fraud probability (LSTM) x anomaly score (autoencoder) -> Normal / Fault / Fraud / Edge case.
- Qualitative validation callout: 58.5% of the LSTM's "false positives" are actually synthetic faults, not model errors.

### 8. Per-user prediction traces
- `reports/figures/plot2_prediction_traces.png`
- Walk through the three panels: fraud case (clean spike, annotated), fault case (small bump, stays below threshold), normal case (flat).

### 9. Attention heatmap (interpretability)
- `reports/figures/plot4_attention_heatmap.png`
- Same fraud-case card as slide 8: the Transformer's last-layer attention peaks exactly on the first fraudulent transaction in the burst when predicting the rest of it.

### 10. Confusion matrix + bonus embedding plot
- `reports/figures/plot6_confusion_matrix.png`
- `reports/figures/plot7_embedding_tsne.png` (bonus) -- categorical embeddings alone give partial, not clean, separation; most signal comes from the personalized amount/time deviation features.

### 11. Limitations
- Synthetic stand-in data (biggest caveat -- say it plainly).
- `card_id` and `merchant_category` are documented heuristics for real IEEE-CIS data.
- Autoencoder is per-transaction, not per-sequence.
- Fault recovery rate is modest (16%) at the current threshold -- faults are subtle by design.

### 12. Team & deliverables checklist
- Member A: data pipeline, sequence construction, baselines.
- Member B: LSTM/Transformer, autoencoder, evaluation, visualization.
- Checklist from `README.md` -- all boxes checked on synthetic data, real-data run pending Kaggle credentials.

---

### Backup slide: synthetic data disclaimer (keep ready, use if asked)
- Kaggle requires authenticated download; this environment/CI has no such access.
- `src/sfraud/data/synthetic.py` generates a dataset with the *exact same schema* as the real one, so swapping in the real CSVs requires zero code changes.
- All absolute metrics should be re-validated on real data before being quoted outside this project.
