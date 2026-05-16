"""
Visualization Module
────────────────────
Matplotlib-based plots for training curves, confusion matrix, and
per-class metrics. All plots are saved as PNG images.

Usage:
    from visualize import plot_all
    plot_all()                           # reads from default MODELS_DIR
    plot_all(output_dir="output/models") # custom path
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
)

import config


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _label_names():
    """Returns {label_id: label_name} from config."""
    return {v: k for k, v in config.LABEL_MAP.items()}

def _label_names(inv_label=None):
    if inv_label is not None:
        return inv_label
    return {v: k for k, v in config.LABEL_MAP.items()}

def _save(fig, path_no_ext):
    """Saves a matplotlib figure as PNG."""
    fpath = f"{path_no_ext}.png"
    fig.savefig(fpath, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(fpath)}")


# ──────────────────────────────────────────────
# 1. Training curves
# ──────────────────────────────────────────────

def plot_training_curves(history_df, output_dir):
    """Loss and accuracy curves (train vs val) per epoch."""
    df = history_df

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df["epoch"], df["tr_loss"], "o-", color="#636EFA", label="Train loss", markersize=4)
    ax1.plot(df["epoch"], df["va_loss"], "s--", color="#EF553B", label="Val loss", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["epoch"], df["tr_acc"], "o-", color="#636EFA", label="Train acc", markersize=4)
    ax2.plot(df["epoch"], df["va_acc"], "s--", color="#EF553B", label="Val acc", markersize=4)
    if "va_f1w" in df.columns:
        ax2.plot(df["epoch"], df["va_f1w"], "^:", color="#00CC96", label="Val F1w", markersize=4)
    if "va_bacc" in df.columns:
        ax2.plot(df["epoch"], df["va_bacc"], "d:", color="#AB63FA", label="Val BAcc", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Accuracy & Metrics")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "training_curves"))


# ──────────────────────────────────────────────
# 2. Confusion matrix
# ──────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, output_dir, title="Confusion Matrix",inv_label=None):
    """Annotated heatmap confusion matrix with counts and percentages."""
    inv = _label_names(inv_label)
    labels_present = sorted(set(y_true) | set(y_pred))
    names = [inv.get(i, str(i)) for i in labels_present]

    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    cm_norm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    n = len(names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.9), max(6, n * 0.8)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall")

    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.0%})",
                    ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontweight="bold")

    fname = title.lower().replace(" ", "_")
    _save(fig, os.path.join(output_dir, fname))


# ──────────────────────────────────────────────
# 3. Per-class metrics heatmap
# ──────────────────────────────────────────────

def plot_class_metrics(y_true, y_pred, output_dir, title="Per-Class Metrics",inv_label=None):
    """Heatmap of precision / recall / F1 per class."""
    inv = _label_names(inv_label)
    labels_present = sorted(set(y_true) | set(y_pred))
    names = [inv.get(i, str(i)) for i in labels_present]

    report = classification_report(
        y_true, y_pred, labels=labels_present, target_names=names,
        output_dict=True, zero_division=0,
    )

    metrics = ["precision", "recall", "f1-score"]
    rows = []
    row_names = list(names)
    for name in names:
        rows.append([report[name][m] for m in metrics])

    for avg_key in ["macro avg", "weighted avg"]:
        if avg_key in report:
            rows.append([report[avg_key][m] for m in metrics])
            row_names.append(avg_key)

    z = np.array(rows)
    n_rows = len(row_names)

    fig, ax = plt.subplots(figsize=(5, max(4, n_rows * 0.5)))
    im = ax.imshow(z, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Score")

    for i in range(n_rows):
        for j in range(3):
            color = "white" if z[i, j] < 0.3 else "black"
            ax.text(j, i, f"{z[i, j]:.2f}",
                    ha="center", va="center", fontsize=10, color=color)

    ax.set_xticks(range(3))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(["Precision", "Recall", "F1-Score"], fontsize=10)
    ax.set_yticklabels(row_names, fontsize=9)
    ax.set_title(title, fontweight="bold")

    fname = title.lower().replace(" ", "_").replace("-", "_")
    _save(fig, os.path.join(output_dir, fname))


# ──────────────────────────────────────────────
# 4. Label distribution
# ──────────────────────────────────────────────

def plot_label_distribution(y_true, output_dir, title="Walk Label Distribution", inv_label=None):
    """Bar chart of class counts in the dataset."""
    inv = _label_names(inv_label)
    labels, counts = np.unique(y_true, return_counts=True)
    names = [inv.get(int(l), str(l)) for l in labels]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.8), 5))
    bars = ax.bar(range(len(names)), counts, color="#636EFA", edgecolor="white")

    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                str(c), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    fname = title.lower().replace(" ", "_")
    _save(fig, os.path.join(output_dir, fname))


# ──────────────────────────────────────────────
# 5. Main entry points
# ──────────────────────────────────────────────

def plot_test_results(y_true, y_pred, output_dir=None,inv_label=None):
    """Generates confusion matrix + per-class metrics from predictions."""
    output_dir = output_dir or config.MODELS_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating test plots...")
    plot_confusion_matrix(y_true, y_pred, output_dir,inv_label=inv_label)
    plot_class_metrics(y_true, y_pred, output_dir,inv_label=inv_label)
    plot_label_distribution(y_true, output_dir, title="Test Label Distribution",inv_label=inv_label)


def plot_history(output_dir=None, history_df=None):
    """Generates training curves from a DataFrame or a history.csv file."""
    output_dir = output_dir or config.MODELS_DIR
    if history_df is None:
        history_csv = os.path.join(output_dir, "history.csv")
        if not os.path.exists(history_csv):
            print(f"Warning: {history_csv} not found, skipping training curves.")
            return
        history_df = pd.read_csv(history_csv)
    print("\nGenerating training curves...")
    plot_training_curves(history_df, output_dir)


def plot_all(output_dir=None):
    """Generates all plots from saved CSV files.

    Usage:
        from visualize import plot_all
        plot_all()
    """
    output_dir = output_dir or config.MODELS_DIR

    # Training curves
    plot_history(output_dir)

    # Test predictions
    preds_csv = os.path.join(output_dir, "test_predictions.csv")
    if os.path.exists(preds_csv):
        df = pd.read_csv(preds_csv)
        print("\nGenerating test plots...")
        plot_confusion_matrix(
            df["y_true"].tolist(), df["y_pred"].tolist(), output_dir,
        )
        plot_class_metrics(
            df["y_true"].tolist(), df["y_pred"].tolist(), output_dir,
        )
        plot_label_distribution(
            df["y_true"].tolist(), output_dir,
            title="Test Label Distribution",
        )
    else:
        print(f"Warning: {preds_csv} not found, skipping test plots.")

    print(f"\nAll plots saved to {output_dir}/")
