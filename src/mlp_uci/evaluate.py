"""Metrics, permutation importance and figures."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def classification_metrics(y_true, proba, class_names=None) -> dict:
    y_pred = np.argmax(proba, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc_ovr": float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, target_names=class_names,
                                        output_dict=True, zero_division=0),
    }


def regression_metrics(y_true, y_pred) -> dict:
    y_pred = np.asarray(y_pred).ravel()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def permutation_importance(predict: Callable, X, y, score_fn: Callable, feature_names,
                           n_repeats=5, seed=0) -> list[dict]:
    """Mean drop in a higher-is-better score when one feature is shuffled."""
    rng = np.random.default_rng(seed)
    base = score_fn(y, predict(X))
    out = []
    for j, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            drops.append(base - score_fn(y, predict(Xp)))
        out.append({"feature": name, "mean_drop": float(np.mean(drops)),
                    "std_drop": float(np.std(drops))})
    return sorted(out, key=lambda d: d["mean_drop"], reverse=True)


def plot_learning_curves(history: dict, path: Path, metric: str, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(history["loss"], label="train")
    axes[0].plot(history["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="epoch")
    axes[0].legend()
    axes[1].plot(history[metric], label="train")
    axes[1].plot(history[f"val_{metric}"], label="validation")
    axes[1].set(title=metric, xlabel="epoch")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, path: Path, title: str) -> None:
    cm = np.asarray(cm)
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xlabel="predicted", ylabel="true", title=title)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.0%})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pred_vs_true(y_true, y_pred, path: Path, title: str) -> None:
    y_pred = np.asarray(y_pred).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    axes[0].scatter(y_true, y_pred, s=8, alpha=0.4)
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[0].set(xlabel="true", ylabel="predicted", title="Predicted vs true")
    axes[1].hist(y_pred - y_true, bins=40)
    axes[1].set(xlabel="residual (pred - true)", title="Residuals")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(items: list[dict], path: Path, title: str, top: int = 16) -> None:
    items = items[:top][::-1]
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(items) + 1.2))
    ax.barh([d["feature"] for d in items], [d["mean_drop"] for d in items],
            xerr=[d["std_drop"] for d in items])
    ax.set(xlabel="drop in score when shuffled", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _n_params(n_in: int, hidden: tuple[int, ...], n_out: int) -> int:
    sizes = [n_in, *hidden, n_out]
    return sum((a + 1) * b for a, b in zip(sizes[:-1], sizes[1:], strict=True))


def plot_architectures(architectures: dict, n_in: int, n_out: int, task: str, path: Path,
                       max_nodes: int = 8) -> None:
    """Draw each architecture as a layered diagram (large layers are truncated with '...').

    Node counts printed under every column are the REAL layer sizes.
    """
    head = f"Output\n{n_out} (softmax)" if task == "classification" else "Output\n1 (linear)"
    fig, axes = plt.subplots(1, len(architectures), figsize=(4.6 * len(architectures), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (name, hidden) in zip(axes, architectures.items(), strict=True):
        sizes = [n_in, *hidden, n_out]
        labels = [f"Input\n{n_in}"] + [f"Dense {h}\nReLU" for h in hidden] + [head]
        shown = [min(s, max_nodes) for s in sizes]
        xs = np.linspace(0, 1, len(sizes))
        coords = []
        for x, k in zip(xs, shown, strict=True):
            ys = np.linspace(0.15, 0.85, k) if k > 1 else np.array([0.5])
            coords.append([(x, y) for y in ys])
        for left, right in zip(coords[:-1], coords[1:], strict=True):
            for (x0, y0) in left:
                for (x1, y1) in right:
                    ax.plot([x0, x1], [y0, y1], color="0.8", lw=0.5, zorder=1)
        palette = ["#4c72b0"] + ["#dd8452"] * len(hidden) + ["#55a868"]
        for pts, size, color in zip(coords, sizes, palette, strict=True):
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=140, color=color,
                       edgecolor="k", zorder=3)
            if size > max_nodes:
                ax.text(pts[0][0], 0.5, "⋮", ha="center", va="center", fontsize=18, zorder=4,
                        bbox={"fc": "white", "ec": "none", "pad": 0.5})
        for x, text in zip(xs, labels, strict=True):
            ax.text(x, 0.02, text, ha="center", va="top", fontsize=9)
        ax.set(xlim=(-0.12, 1.12), ylim=(-0.2, 1.0))
        ax.axis("off")
        ax.set_title(f"{name}\n{len(hidden)} hidden layer(s), "
                     f"{_n_params(n_in, tuple(hidden), n_out):,} parameters", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_grid_heatmap(table, path: Path, title: str) -> None:
    """Best validation loss for every architecture x optimizer pair (min over dropout/L2)."""
    piv = table.pivot_table(index="architecture", columns="optimizer", values="val_loss",
                            aggfunc="min")
    fig, ax = plt.subplots(figsize=(1.8 * len(piv.columns) + 2.2, 0.9 * len(piv) + 1.6))
    im = ax.imshow(piv.values, cmap="viridis_r", aspect="auto")
    ax.set(xticks=range(len(piv.columns)), yticks=range(len(piv.index)), title=title,
           xlabel="optimizer", ylabel="architecture")
    ax.set_xticklabels(piv.columns)
    ax.set_yticklabels(piv.index)
    vmin, vmax = np.nanmin(piv.values), np.nanmax(piv.values)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if (v - vmin) < 0.5 * (vmax - vmin) else "black")
    fig.colorbar(im, ax=ax, label="best val loss")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_optimizer_curves(table, histories: dict, path: Path, title: str) -> None:
    """Validation-loss curves of the best configuration found for each optimizer."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for opt, grp in table.groupby("optimizer"):
        row = grp.sort_values("val_loss").iloc[0]
        key = (row["architecture"], opt, float(row.get("dropout", 0.0)),
               float(row.get("l2", 0.0)))
        ax.plot(histories[key]["val_loss"], label=f"{opt} ({row['architecture']})")
    ax.set(xlabel="epoch", ylabel="validation loss", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
