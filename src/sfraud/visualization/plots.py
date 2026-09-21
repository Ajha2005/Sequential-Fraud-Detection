"""Phase 7: the seven required plots plus the bonus embedding visualization.
Each `plot_*` function is self-contained (loads whatever metrics/predictions
it needs and saves one PNG to reports/figures/) so `scripts/08_visualize.py`
can just call all of them in sequence."""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D

from sfraud.datasets.torch_dataset import SequenceDataset
from sfraud.evaluation.compare import _sequence_predictions_with_position
from sfraud.evaluation.predict import load_trained_model
from sfraud.utils.config import figures_dir, load_config, metrics_dir, resolve_path
from sfraud.visualization.style import CATEGORY_COLORS, COLOR_FAULT, COLOR_FRAUD, COLOR_NEUTRAL, COLOR_NORMAL, MODEL_COLORS, MODEL_LABELS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plot 1: prediction confidence / AUC vs. number of transactions seen
# ---------------------------------------------------------------------------
def plot_confidence_curve(cfg: dict) -> None:
    df = pd.read_csv(metrics_dir() / "confidence_curve.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for model_name, g in df.groupby("model"):
        g = g.sort_values("n_seen")
        color = MODEL_COLORS.get(model_name, "gray")
        axes[0].plot(g["n_seen"], g["auc_roc"], marker="o", markersize=3, label=MODEL_LABELS.get(model_name, model_name), color=color)
        axes[1].plot(g["n_seen"], g["accuracy"], marker="o", markersize=3, label=MODEL_LABELS.get(model_name, model_name), color=color)

    axes[0].set_xlabel("Number of transactions seen (N)")
    axes[0].set_ylabel("AUC-ROC (within that N)")
    axes[0].set_title("Ranking quality vs. history length")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Number of transactions seen (N)")
    axes[1].set_ylabel("Accuracy @ threshold=0.5")
    axes[1].set_title("Accuracy vs. history length")
    axes[1].legend(fontsize=8)

    fig.suptitle("Plot 1: Prediction confidence vs. number of transactions seen")
    fig.tight_layout()
    fig.savefig(figures_dir() / "plot1_confidence_vs_n.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: per-user prediction traces (fraud / fault / normal), spike annotated
# ---------------------------------------------------------------------------
def _pick_example_cards(df: pd.DataFrame) -> dict[str, str]:
    lengths = df.groupby("card_id")["n_seen"].max()

    fraud_cards = df.loc[df["isFraud"] == 1, "card_id"].unique()
    fault_only_cards = set(df.loc[df["synthetic_fault_flag"] == 1, "card_id"]) - set(fraud_cards)
    clean_cards = set(df["card_id"]) - set(fraud_cards) - fault_only_cards

    def longest(candidates):
        candidates = [c for c in candidates if c in lengths.index]
        return max(candidates, key=lambda c: lengths[c]) if candidates else None

    return {
        "fraud": longest(fraud_cards),
        "fault": longest(fault_only_cards),
        "normal": longest(clean_cards),
    }


def plot_prediction_traces(cfg: dict, df: pd.DataFrame | None = None) -> dict[str, str]:
    df = df if df is not None else _sequence_predictions_with_position(cfg, "lstm")
    examples = _pick_example_cards(df)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=False)
    titles = {"fraud": "Fraud case", "fault": "Fault case (benign anomaly)", "normal": "Normal case"}
    spike_cols = {"fraud": "isFraud", "fault": "synthetic_fault_flag", "normal": None}

    for ax, key in zip(axes, ["fraud", "fault", "normal"]):
        card_id = examples[key]
        if card_id is None:
            ax.set_title(f"{titles[key]} -- no example found in test split")
            continue
        g = df[df["card_id"] == card_id].sort_values("n_seen")
        ax.plot(g["n_seen"], g["fraud_prob"], color=COLOR_NORMAL, linewidth=1.5, zorder=1)
        point_colors = np.where(g["isFraud"] == 1, COLOR_FRAUD, np.where(g["synthetic_fault_flag"] == 1, COLOR_FAULT, COLOR_NEUTRAL))
        ax.scatter(g["n_seen"], g["fraud_prob"], c=point_colors, s=28, zorder=2, edgecolor="white", linewidth=0.5)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)

        spike_col = spike_cols[key]
        if spike_col is not None and (g[spike_col] == 1).any():
            spike_n = g.loc[g[spike_col] == 1, "n_seen"].iloc[0]
            ax.axvline(spike_n, color="black", linestyle="--", linewidth=1)
            ax.annotate(
                "spike point", xy=(spike_n, 1.0), xytext=(spike_n, 1.12),
                ha="center", fontsize=8, arrowprops=dict(arrowstyle="->", lw=1),
            )
            ax.set_ylim(-0.05, 1.25)
        else:
            ax.set_ylim(-0.05, 1.1)

        ax.set_title(f"{titles[key]} -- card {card_id}")
        ax.set_ylabel("Predicted fraud probability")
        ax.set_xlabel("Transaction # in history (N)")

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_NEUTRAL, label="Normal txn", markersize=7),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_FAULT, label="Fault txn", markersize=7),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_FRAUD, label="Fraud txn", markersize=7),
    ]
    fig.legend(handles=legend_elems, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle("Plot 2: Per-user prediction traces (LSTM)", y=1.05)
    fig.tight_layout()
    fig.savefig(figures_dir() / "plot2_prediction_traces.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return examples


# ---------------------------------------------------------------------------
# Plot 3: fault/fraud quadrant scatter
# ---------------------------------------------------------------------------
def plot_quadrant_scatter(cfg: dict) -> None:
    processed_dir = resolve_path(cfg["data"]["processed_dir"])
    df = pd.read_parquet(processed_dir / "fault_fraud_quadrants.parquet")

    with open(metrics_dir() / "fault_fraud_quadrant.json") as f:
        summary = json.load(f)
    fraud_thresh = summary["fraud_prob_threshold"]
    anomaly_thresh = summary["anomaly_score_threshold"]

    fig, ax = plt.subplots(figsize=(7, 6))
    for quadrant, color in CATEGORY_COLORS.items():
        g = df[df["quadrant"] == quadrant]
        ax.scatter(
            g["anomaly_score"] + 1e-6, g["fraud_prob"], s=12, alpha=0.5, c=color, label=f"{quadrant} (n={len(g)})"
        )

    ax.set_xscale("log")
    ax.axhline(fraud_thresh, color="black", linestyle="--", linewidth=1)
    ax.axvline(anomaly_thresh, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Anomaly score (autoencoder reconstruction loss, log scale)")
    ax.set_ylabel("Fraud probability (LSTM)")
    ax.set_title("Plot 3: Fault/Fraud quadrant (held-out val+test)")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(figures_dir() / "plot3_quadrant_scatter.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: attention heatmap (Transformer)
# ---------------------------------------------------------------------------
def plot_attention_heatmap(cfg: dict, card_id: str | None = None) -> None:
    from sfraud.models.transformer import TransformerFraudModel  # noqa: F401 (type context)

    device = torch.device("cpu")
    model = load_trained_model(cfg, "transformer", device)
    ds = SequenceDataset("test", cfg)

    if card_id is None:
        df = _sequence_predictions_with_position(cfg, "transformer")
        card_id = _pick_example_cards(df)["fraud"]

    idx = int(np.where(ds.card_ids == card_id)[0][0])
    x_num = ds.X_num[idx : idx + 1].to(device)
    x_cat = ds.X_cat[idx : idx + 1].to(device)
    mask = ds.mask[idx].numpy()
    L = int(mask.sum())

    with torch.no_grad():
        _, attn_weights_all = model(x_num, x_cat, return_attn=True)
    last_layer_attn = attn_weights_all[-1][0, :L, :L].numpy()  # (L, L) query x key

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(last_layer_attn, cmap="viridis", aspect="auto")
    ax.set_xlabel("Attended-to transaction (key position)")
    ax.set_ylabel("Transaction being predicted (query position)")
    ax.set_title(f"Plot 4: Transformer self-attention (last layer)\ncard {card_id}")
    fig.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()
    fig.savefig(figures_dir() / "plot4_attention_heatmap.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5: baseline vs. sequence vs. published-benchmark comparison
# ---------------------------------------------------------------------------
def plot_benchmark_comparison(cfg: dict) -> None:
    with open(metrics_dir() / "model_comparison.json") as f:
        comparison = json.load(f)

    model_order = ["logreg", "xgboost", "lstm", "transformer"]
    aucs = [comparison[m]["auc_roc"] for m in model_order]
    colors = [MODEL_COLORS[m] for m in model_order]
    labels = [MODEL_LABELS[m] for m in model_order]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, aucs, color=colors)
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.3f}", ha="center", fontsize=9)

    bench = comparison["published_benchmarks"]
    low = bench["Strong supervised baseline (published, low)"]["auc_roc"]
    high = bench["Strong supervised baseline (published, high)"]["auc_roc"]
    ceiling = bench["Kaggle competition winners (ceiling)"]["auc_roc"]
    floor = bench["Random / majority-class baseline"]["auc_roc"]

    ax.axhspan(low, high, color="gold", alpha=0.25, label=f"Published strong baseline ({low:.2f}-{high:.2f})")
    ax.axhline(ceiling, color="black", linestyle="--", linewidth=1, label=f"Kaggle winners ceiling (~{ceiling:.3f})")
    ax.axhline(floor, color="gray", linestyle=":", linewidth=1, label=f"Random baseline (~{floor:.2f})")

    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("Test AUC-ROC")
    ax.set_title("Plot 5: Baseline vs. sequence model vs. published benchmarks\n(our numbers are on synthetic stand-in data -- see README)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(figures_dir() / "plot5_benchmark_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 6: confusion matrix for the final fraud model
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cfg: dict, model_name: str = "lstm") -> None:
    with open(metrics_dir() / "model_comparison.json") as f:
        comparison = json.load(f)
    cm = np.array(comparison[model_name]["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted normal", "Predicted fraud"])
    ax.set_yticklabels(["Actual normal", "Actual fraud"])
    ax.set_title(f"Plot 6: Confusion matrix -- {MODEL_LABELS.get(model_name, model_name)} (test)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(figures_dir() / "plot6_confusion_matrix.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 7 (bonus): t-SNE of learned categorical embeddings
# ---------------------------------------------------------------------------
def plot_embedding_tsne(cfg: dict, model_name: str = "lstm", max_points: int = 2500) -> None:
    from sklearn.manifold import TSNE

    device = torch.device("cpu")
    model = load_trained_model(cfg, model_name, device)
    ds = SequenceDataset("test", cfg)

    mask = ds.mask.numpy().astype(bool)
    n_cards, T = mask.shape
    card_grid = np.repeat(np.arange(n_cards)[:, None], T, axis=1)

    with torch.no_grad():
        cat_emb = model.encoder.cat_embedder(ds.X_cat)  # (n_cards, T, emb_dim_total)
    cat_emb = cat_emb.numpy()[mask]
    y = ds.y.numpy()[mask]
    fault = ds.fault.numpy()[mask]

    n = len(cat_emb)
    if n > max_points:
        rng = np.random.default_rng(cfg["seed"])
        idx = rng.choice(n, max_points, replace=False)
        cat_emb, y, fault = cat_emb[idx], y[idx], fault[idx]

    coords = TSNE(n_components=2, init="pca", perplexity=30, random_state=cfg["seed"]).fit_transform(cat_emb)

    label = np.where(y == 1, "Fraud", np.where(fault == 1, "Fault", "Normal"))
    fig, ax = plt.subplots(figsize=(7, 6))
    for cat in ["Normal", "Fault", "Fraud"]:
        sel = label == cat
        ax.scatter(coords[sel, 0], coords[sel, 1], s=10, alpha=0.6, c=CATEGORY_COLORS[cat], label=f"{cat} (n={sel.sum()})")
    ax.set_title(f"Plot 7 (bonus): t-SNE of learned categorical embeddings ({MODEL_LABELS.get(model_name, model_name)})")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir() / "plot7_embedding_tsne.png", dpi=150)
    plt.close(fig)


def generate_all_plots(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    logger.info("Plot 1: confidence vs N")
    plot_confidence_curve(cfg)
    logger.info("Plot 2: prediction traces")
    plot_prediction_traces(cfg)
    logger.info("Plot 3: quadrant scatter")
    plot_quadrant_scatter(cfg)
    logger.info("Plot 4: attention heatmap")
    plot_attention_heatmap(cfg)
    logger.info("Plot 5: benchmark comparison")
    plot_benchmark_comparison(cfg)
    logger.info("Plot 6: confusion matrix")
    plot_confusion_matrix(cfg)
    logger.info("Plot 7 (bonus): embedding t-SNE")
    plot_embedding_tsne(cfg)
    logger.info("All plots saved to %s", figures_dir())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    generate_all_plots()
