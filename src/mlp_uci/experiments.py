"""End-to-end experiments: classification (BEED) and regression (ACTG 175)."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import joblib
import keras
import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler

from . import data as D
from .baselines import classification_baselines, regression_baselines
from .evaluate import (
    classification_metrics,
    permutation_importance,
    plot_architectures,
    plot_confusion_matrix,
    plot_grid_heatmap,
    plot_importance,
    plot_learning_curves,
    plot_optimizer_curves,
    plot_pred_vs_true,
    regression_metrics,
)
from .models import ARCHITECTURES, build_shallow_mlp
from .training import DEFAULT_GRID, fit, grid_search
from .utils import ensure_dir, save_json, set_seed


@dataclass
class RunConfig:
    out_dir: str
    data_path: str | None = None
    seed: int = 42
    n_seeds: int = 5
    max_epochs: int = 300
    batch_size: int = 128
    patience: int = 20
    split: str = "default"          # classification: stratified | regression: random
    dataset: str = "actg175"                 # regression dataset
    include_sensitive: bool = False          # actg175: race, gender, homo, drugs
    grid: dict = field(default_factory=lambda: dict(DEFAULT_GRID))


def _mean_std(rows: list[dict], keys) -> dict:
    return {k: {"mean": float(np.mean([r[k] for r in rows])),
                "std": float(np.std([r[k] for r in rows]))} for k in keys}


def _search_figures(out, table, hists, n_in, n_out, task, name) -> None:
    """Architecture diagrams + optimizer/architecture comparison figures."""
    plot_architectures(ARCHITECTURES, n_in, n_out, task, out / "architectures.png")
    plot_grid_heatmap(table, out / "grid_heatmap.png",
                      f"{name}: best validation loss per architecture x optimizer")
    plot_optimizer_curves(table, hists, out / "optimizer_curves.png",
                          f"{name}: validation loss of the best config per optimizer")


def _search_summary(table) -> dict:
    """Best validation loss per optimizer and per architecture (for metrics.json / README)."""
    return {
        "by_optimizer": table.groupby("optimizer")["val_loss"].min().round(5).to_dict(),
        "by_architecture": table.groupby("architecture")["val_loss"].min().round(5).to_dict(),
        "best": table.iloc[0].to_dict(),
    }


def run_classification(cfg: RunConfig) -> dict:
    """BEED: 16 EEG channels -> 4 classes. Test set is used once, at the very end."""
    out = ensure_dir(cfg.out_dir)
    X, y = D.load_beed(cfg.data_path)
    split = D.stratified_split(y.values, seed=cfg.seed)
    Xa, ya = X.values, y.values

    scaler = StandardScaler().fit(Xa[split.train])           # fit on train only
    Xtr, Xva, Xte = (scaler.transform(Xa[i]) for i in (split.train, split.val, split.test))
    ytr, yva, yte = ya[split.train], ya[split.val], ya[split.test]
    n_classes = len(D.BEED_CLASS_NAMES)

    t0 = time.time()
    best, table, hists = grid_search("classification", Xtr, ytr, Xva, yva, n_classes, cfg.grid,
                                     cfg.max_epochs, cfg.batch_size, cfg.patience, cfg.seed)
    table.to_csv(out / "grid_search.csv", index=False)
    _search_figures(out, table, hists, Xtr.shape[1], n_classes, "classification", "BEED")

    runs, first = [], None
    for i in range(cfg.n_seeds):
        set_seed(cfg.seed + i)
        model = build_shallow_mlp(Xtr.shape[1], "classification", n_classes, **best)
        hist = fit(model, Xtr, ytr, Xva, yva, cfg.max_epochs, cfg.batch_size, cfg.patience)
        proba = model.predict(Xte, verbose=0)
        m = classification_metrics(yte, proba, D.BEED_CLASS_NAMES)
        m["epochs_trained"] = len(hist.history["loss"])
        runs.append(m)
        if first is None:                                    # report seed 0, not the best seed
            first = (model, hist.history, proba, m)

    model, history, proba, m0 = first
    plot_learning_curves(history, out / "learning_curves.png", "accuracy",
                         "BEED - shallow MLP (seed 0)")
    plot_confusion_matrix(m0["confusion_matrix"], D.BEED_CLASS_NAMES,
                          out / "confusion_matrix.png", "BEED - test set (seed 0)")
    imp = permutation_importance(lambda a: np.argmax(model.predict(a, verbose=0), axis=1),
                                 Xte, yte, accuracy_score, D.BEED_FEATURES)
    plot_importance(imp, out / "channel_importance.png", "EEG channel importance (test)")
    model.save(out / "model.keras")
    joblib.dump(scaler, out / "scaler.joblib")

    summary = {
        "task": "classification", "dataset": "BEED (UCI 1134)",
        "config": asdict(cfg), "split_sizes": split.sizes(),
        "class_distribution": {c: int((ya == k).sum()) for k, c in enumerate(D.BEED_CLASS_NAMES)},
        "best_hyperparameters": best, "n_parameters": int(model.count_params()),
        "search": _search_summary(table),
        "test_mlp": _mean_std(runs, ["accuracy", "macro_f1", "roc_auc_ovr"]),
        "test_mlp_seed0": {k: m0[k] for k in ("accuracy", "macro_f1", "roc_auc_ovr")},
        "per_class_seed0": {c: m0["report"][c] for c in D.BEED_CLASS_NAMES},
        "baselines": classification_baselines(Xtr, ytr, Xte, yte, cfg.seed),
        "channel_importance": imp,
        "wall_time_s": round(time.time() - t0, 1),
    }
    save_json(summary, out / "metrics.json")
    keras.backend.clear_session()
    return summary


def _load_regression(cfg: RunConfig):
    """Return (X, y, dataset label, target name)."""
    if cfg.dataset == "actg175":
        X, y = D.load_actg175(cfg.data_path, cfg.include_sensitive)
        return X, y, "ACTG 175 (UCI 890)", D.ACTG_TARGET
    raise ValueError(f"unknown dataset {cfg.dataset!r}")


def run_regression(cfg: RunConfig) -> dict:
    """Regression with a shallow MLP. The test set is evaluated once, at the very end.

    ACTG 175: CD4 at ~20 weeks from baseline data; random split (1 row per patient).
    """
    out = ensure_dir(cfg.out_dir)
    X, y, label, target = _load_regression(cfg)
    mode = "random" if cfg.split == "default" else cfg.split
    if mode == "random":
        split = D.random_split(len(y), seed=cfg.seed)
    else:
        raise ValueError(f"unknown split {cfg.split!r}")
    Xa, ya = X.values, y.values

    xs = StandardScaler().fit(Xa[split.train])
    ys = StandardScaler().fit(ya[split.train].reshape(-1, 1))   # scale target, report in raw units
    Xtr, Xva, Xte = (xs.transform(Xa[i]) for i in (split.train, split.val, split.test))
    sc = lambda v: ys.transform(v.reshape(-1, 1)).ravel()       # noqa: E731
    ytr, yva = sc(ya[split.train]), sc(ya[split.val])
    yte_raw = ya[split.test]

    t0 = time.time()
    best, table, hists = grid_search("regression", Xtr, ytr, Xva, yva, None, cfg.grid,
                                     cfg.max_epochs, cfg.batch_size, cfg.patience, cfg.seed)
    table.to_csv(out / "grid_search.csv", index=False)
    _search_figures(out, table, hists, Xtr.shape[1], 1, "regression", label)

    def to_raw(p):
        return ys.inverse_transform(np.asarray(p).reshape(-1, 1)).ravel()

    runs, first = [], None
    for i in range(cfg.n_seeds):
        set_seed(cfg.seed + i)
        model = build_shallow_mlp(Xtr.shape[1], "regression", None, **best)
        hist = fit(model, Xtr, ytr, Xva, yva, cfg.max_epochs, cfg.batch_size, cfg.patience)
        pred = to_raw(model.predict(Xte, verbose=0))
        m = regression_metrics(yte_raw, pred)
        m["epochs_trained"] = len(hist.history["loss"])
        runs.append(m)
        if first is None:
            first = (model, hist.history, pred, m)

    model, history, pred, m0 = first
    plot_learning_curves(history, out / "learning_curves.png", "rmse",
                         f"{label} - shallow MLP (seed 0, {mode} split)")
    plot_pred_vs_true(yte_raw, pred, out / "pred_vs_true.png", f"Test set ({mode} split, seed 0)")
    imp = permutation_importance(lambda a: to_raw(model.predict(a, verbose=0)), Xte, yte_raw,
                                 r2_score, list(X.columns))
    plot_importance(imp, out / "feature_importance.png", "Feature importance (test, drop in R2)")
    model.save(out / "model.keras")
    joblib.dump({"x": xs, "y": ys}, out / "scaler.joblib")

    baselines = regression_baselines(Xtr, ya[split.train], Xte, yte_raw, cfg.seed)
    if "cd40" in X.columns:                        # clinical baseline: "CD4 stays the same"
        baselines["no_change (cd420 = cd40)"] = regression_metrics(
            yte_raw, Xa[split.test][:, list(X.columns).index("cd40")])

    summary = {
        "task": "regression", "dataset": label, "target": target, "split_mode": mode,
        "config": asdict(cfg), "features": list(X.columns), "split_sizes": split.sizes(),
        "target_stats_test": {"mean": float(yte_raw.mean()), "std": float(yte_raw.std())},
        "best_hyperparameters": best, "n_parameters": int(model.count_params()),
        "search": _search_summary(table),
        "test_mlp": _mean_std(runs, ["rmse", "mae", "r2"]),
        "test_mlp_seed0": {k: m0[k] for k in ("rmse", "mae", "r2")},
        "baselines": baselines,
        "feature_importance": imp,
        "wall_time_s": round(time.time() - t0, 1),
    }
    save_json(summary, out / "metrics.json")
    keras.backend.clear_session()
    return summary
