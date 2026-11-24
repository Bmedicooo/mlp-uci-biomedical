"""Training loop with early stopping and validation-based hyperparameter search."""
from __future__ import annotations

import itertools
from collections.abc import Iterable

import keras
import numpy as np
import pandas as pd

from .models import ARCHITECTURES, build_shallow_mlp
from .utils import set_seed

# 3 architectures x 3 optimizers x 2 dropout rates = 18 configurations.
DEFAULT_GRID = {
    "architecture": list(ARCHITECTURES),
    "optimizer": ["adam", "sgd", "rmsprop"],
    "dropout": [0.0, 0.2],
    "l2": [0.0],
}


def fit(model, X_train, y_train, X_val, y_val, max_epochs=300, batch_size=128, patience=20):
    """Fit with early stopping on val_loss (restoring the best weights)."""
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=max(2, patience // 3), min_lr=1e-6),
    ]
    return model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=max_epochs,
                     batch_size=batch_size, callbacks=callbacks, verbose=0)


def _combos(grid: dict[str, Iterable]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*grid.values())]


def to_build_kwargs(combo: dict, architectures=ARCHITECTURES) -> dict:
    """Translate one grid combination into keyword arguments for build_shallow_mlp."""
    kw = {k: v for k, v in combo.items() if k != "architecture"}
    kw["hidden_layers"] = tuple(int(u) for u in architectures[combo["architecture"]])
    for key in ("dropout", "l2", "learning_rate"):
        if key in kw and kw[key] is not None:
            kw[key] = float(kw[key])
    return kw


def grid_search(task, X_train, y_train, X_val, y_val, n_classes=None, grid=None,
                max_epochs=300, batch_size=128, patience=20, seed=0,
                architectures=ARCHITECTURES):
    """Try every (architecture x optimizer x dropout x l2) combination; pick by VAL loss.

    Returns (best_build_kwargs, table, histories) where ``table`` has one row per combination
    (sorted by val_loss) and ``histories`` maps (architecture, optimizer, dropout, l2) to the
    per-epoch train/validation loss. The test set is never touched here.
    """
    grid = grid or DEFAULT_GRID
    rows, histories = [], {}
    for combo in _combos(grid):
        set_seed(seed)
        kw = to_build_kwargs(combo, architectures)
        model = build_shallow_mlp(X_train.shape[1], task, n_classes, **kw)
        hist = fit(model, X_train, y_train, X_val, y_val, max_epochs, batch_size, patience)
        best = int(np.argmin(hist.history["val_loss"]))
        key = (combo["architecture"], combo["optimizer"],
               float(combo.get("dropout", 0.0)), float(combo.get("l2", 0.0)))
        histories[key] = {"loss": list(hist.history["loss"]),
                          "val_loss": list(hist.history["val_loss"])}
        rows.append({**combo, "val_loss": float(hist.history["val_loss"][best]),
                     "epochs": best + 1, "n_params": int(model.count_params())})
        keras.backend.clear_session()
    table = pd.DataFrame(rows).sort_values("val_loss").reset_index(drop=True)
    best_kwargs = to_build_kwargs(table.loc[0, list(grid)].to_dict(), architectures)
    return best_kwargs, table, histories
