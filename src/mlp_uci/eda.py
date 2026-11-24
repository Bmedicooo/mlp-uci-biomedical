"""Exploratory data analysis for ACTG 175 (figures + a small JSON summary)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import data as D  # noqa: E402
from .utils import ensure_dir, save_json  # noqa: E402

COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]


def _ci95(x: np.ndarray) -> tuple[float, float]:
    m, se = float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x)))
    return m - 1.96 * se, m + 1.96 * se


def run_eda_actg175(df: pd.DataFrame, out_dir: str | Path) -> dict:
    """Save EDA figures to ``out_dir`` and return/write a summary (``eda_summary.json``)."""
    out = ensure_dir(out_dir)
    d = df.copy()
    d["delta_cd4"] = d["cd420"] - d["cd40"]
    arms = D.ACTG_TREATMENTS

    # 1) distributions of baseline and week-20 CD4
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    ax[0].hist(d["cd40"], bins=40, color=COLORS[0])
    ax[0].set(title="CD4 at baseline (cd40)", xlabel="cells / mm3", ylabel="patients")
    ax[1].hist(d["cd420"], bins=40, color=COLORS[1])
    ax[1].set(title="CD4 at ~20 weeks (cd420) - TARGET", xlabel="cells / mm3")
    fig.tight_layout()
    fig.savefig(out / "eda_target_distribution.png", dpi=150)
    plt.close(fig)

    # 2) baseline vs week 20, by treatment arm
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for k, name in arms.items():
        sub = d[d["trt"] == k]
        ax.scatter(sub["cd40"], sub["cd420"], s=8, alpha=0.35, color=COLORS[k], label=name)
    lim = [0, float(max(d["cd40"].max(), d["cd420"].max()))]
    ax.plot(lim, lim, "k--", lw=1, label="no change")
    ax.set(xlabel="cd40 (baseline)", ylabel="cd420 (week 20)", title="CD4: baseline vs week 20")
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "eda_cd40_vs_cd420.png", dpi=150)
    plt.close(fig)

    # 3) change in CD4 by arm
    groups = [d.loc[d["trt"] == k, "delta_cd4"].to_numpy() for k in arms]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.boxplot(groups, tick_labels=list(arms.values()), showfliers=False)
    for i, g in enumerate(groups, start=1):
        ax.scatter([i], [g.mean()], color="red", zorder=3, label="mean" if i == 1 else None)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set(ylabel="cd420 - cd40", title="Change in CD4 by treatment arm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "eda_treatment_effect.png", dpi=150)
    plt.close(fig)

    # 4) correlations among numeric variables
    cols = D.ACTG_NUMERIC + [D.ACTG_TARGET]
    corr = d[cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set(xticks=range(len(cols)), yticks=range(len(cols)), title="Correlations")
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "eda_correlations.png", dpi=150)
    plt.close(fig)

    summary = {
        "n_patients": int(len(d)),
        "missing_values": {c: int(v) for c, v in d.isna().sum().items() if v},
        "cd40": d["cd40"].describe().round(2).to_dict(),
        "cd420": d["cd420"].describe().round(2).to_dict(),
        "corr_cd40_cd420": float(d["cd40"].corr(d["cd420"])),
        "change_by_arm": {
            name: {"n": int(len(g)), "mean_change": float(g.mean()),
                   "ci95": [float(v) for v in _ci95(g)]}
            for name, g in zip(arms.values(), groups, strict=True)
        },
        "excluded_columns": D.ACTG_EXCLUDED,
        "sensitive_columns_excluded_by_default": D.ACTG_SENSITIVE,
    }
    save_json(summary, out / "eda_summary.json")
    return summary
