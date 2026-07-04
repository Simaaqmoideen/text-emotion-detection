"""
visualize.py — Evaluation Visualization Utilities
===================================================
Publication-quality plots for model evaluation: confusion matrices,
per-class metric bar charts, and class distribution charts.

Usage:
    from src.visualize import plot_confusion_matrix, plot_class_metrics
"""

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Use non-interactive backend so plots save without a display server
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Style Configuration
# ---------------------------------------------------------------------------
# Apply a clean, modern aesthetic globally.
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FAFAFA",
    "axes.edgecolor": "#CCCCCC",
    "grid.color": "#E0E0E0",
    "font.family": "sans-serif",
})

# Curated color palette for emotion classes
EMOTION_PALETTE = {
    "happy": "#4CAF50",
    "sad": "#5C6BC0",
    "angry": "#EF5350",
    "surprise": "#FF9800",
    "fear": "#AB47BC",
    "neutral": "#78909C",
    "love": "#E91E63",
    "disgust": "#795548",
}


def _get_color(label: str) -> str:
    """Return a palette color for a known emotion, else default grey."""
    return EMOTION_PALETTE.get(label.lower(), "#90A4AE")


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    normalize: bool = True,
) -> None:
    """
    Plot a clean, annotated confusion matrix using Seaborn's heatmap.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    labels : list of str
        Ordered class names for axis labels.
    title : str
        Plot title.
    save_path : str or None
        If provided, save the figure to this path.
    figsize : tuple
        Figure dimensions in inches.
    normalize : bool, default True
        If True, show percentages instead of raw counts.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        # Normalize each row (true label) to show recall percentages
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".1%"
        vmin, vmax = 0, 1
    else:
        cm_display = cm
        fmt = "d"
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Proportion" if normalize else "Count"},
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold", labelpad=12)
    ax.set_ylabel("True Label", fontsize=13, fontweight="bold", labelpad=12)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=16)
    ax.tick_params(axis="both", labelsize=11)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Confusion matrix saved → {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-Class Metric Bar Chart
# ---------------------------------------------------------------------------
def plot_class_metrics(
    report_dict: dict,
    metric: str = "f1-score",
    title: str = "Per-Class F1-Score",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 5),
) -> None:
    """
    Horizontal bar chart of a chosen metric from sklearn's classification
    report dictionary.

    Parameters
    ----------
    report_dict : dict
        Output of ``classification_report(..., output_dict=True)``.
    metric : str
        One of 'precision', 'recall', 'f1-score'.
    title : str
        Plot title.
    save_path : str or None
        If provided, save figure.
    figsize : tuple
        Figure size.
    """
    # Filter out aggregate keys
    skip = {"accuracy", "macro avg", "weighted avg"}
    classes = [k for k in report_dict if k not in skip]
    values = [report_dict[k][metric] for k in classes]

    # Sort by value descending
    sorted_pairs = sorted(zip(classes, values), key=lambda x: x[1])
    classes, values = zip(*sorted_pairs)

    colors = [_get_color(c) for c in classes]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(classes, values, color=colors, edgecolor="white", height=0.6)

    # Annotate values on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#333333",
        )

    ax.set_xlim(0, 1.12)
    ax.set_xlabel(metric.capitalize(), fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.axvline(x=1.0, color="#CCCCCC", linestyle="--", linewidth=0.8)

    sns.despine(left=True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Class metrics chart saved → {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Class Distribution Chart
# ---------------------------------------------------------------------------
def plot_class_distribution(
    labels: np.ndarray,
    title: str = "Emotion Class Distribution",
    save_path: Optional[str] = None,
    figsize: tuple = (9, 5),
) -> None:
    """
    Vertical bar chart showing the number of samples per emotion class.

    Parameters
    ----------
    labels : array-like
        Array of label strings.
    title : str
        Plot title.
    save_path : str or None
        If provided, save figure.
    figsize : tuple
        Figure size.
    """
    unique, counts = np.unique(labels, return_counts=True)
    # Sort by count descending
    order = np.argsort(-counts)
    unique, counts = unique[order], counts[order]
    colors = [_get_color(u) for u in unique]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(unique, counts, color=colors, edgecolor="white", width=0.6)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.02,
            str(count),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("Emotion", fontsize=12, fontweight="bold")
    ax.set_ylabel("Sample Count", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)

    sns.despine()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Distribution chart saved → {save_path}")
    plt.close(fig)
