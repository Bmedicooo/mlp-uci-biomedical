"""Dataset loading (UCI), schema validation and leakage-aware splitting.

Datasets:
* BEED: Bangalore EEG Epilepsy Dataset, UCI id 1134, DOI 10.24432/C5K33B.
* ACTG 175 (AIDS Clinical Trials Group Study 175), UCI id 890, DOI 10.24432/C5ZG8F.

Each loader accepts a local file (``path``) or falls back to the ``ucimlrepo``
package, caching the raw CSV under ``data/raw`` so later runs work offline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BEED_ID = 1134
ACTG_ID = 890

BEED_FEATURES = [f"X{i}" for i in range(1, 17)]
BEED_TARGET = "y"
BEED_CLASS_NAMES = ["healthy", "generalized_seizure", "focal_seizure", "seizure_events"]

# --- ACTG 175 (AIDS Clinical Trials Group Study 175, UCI id 890, DOI 10.24432/C5ZG8F) ---
ACTG_TARGET = "cd420"                      # CD4 count at 20 +/- 5 weeks
ACTG_NUMERIC = ["age", "wtkg", "karnof", "preanti", "cd40", "cd80"]
ACTG_BINARY = ["hemo", "oprior", "z30", "zprior", "str2", "symptom"]
ACTG_CATEGORICAL = {"trt": [0, 1, 2, 3], "strat": [1, 2, 3]}   # one-hot encoded
ACTG_SENSITIVE = ["race", "gender", "homo", "drugs"]            # excluded unless requested
ACTG_TREATMENTS = {0: "ZDV", 1: "ZDV+ddI", 2: "ZDV+Zal", 3: "ddI"}
# Columns that must NEVER be predictors, and why (post-baseline or redundant):
ACTG_EXCLUDED = {
    "pidnum": "patient identifier",
    "cid": "outcome: event indicator (observed after baseline)",
    "time": "outcome: days to event/censoring (observed after baseline)",
    "offtrt": "off treatment before 96 weeks (observed after baseline)",
    "cd820": "CD8 count at 20 weeks (measured after baseline)",
    "treat": "redundant with trt",
}

DEFAULT_CACHE = Path("data/raw")


def _fetch_uci(uci_id: int) -> pd.DataFrame:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install `ucimlrepo` or pass --data-path with a local file.") from exc
    ds = fetch_ucirepo(id=uci_id)
    return ds.data.original.copy()


def _read_or_fetch(path: str | Path | None, cache_name: str, uci_id: int) -> pd.DataFrame:
    if path is not None:
        df = pd.read_csv(path)
    else:
        cached = DEFAULT_CACHE / cache_name
        if cached.exists():
            df = pd.read_csv(cached)
        else:
            df = _fetch_uci(uci_id)
            cached.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cached, index=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}. Found: {list(df.columns)}")


def load_beed(path: str | Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for BEED. X has 16 EEG channels (X1..X16); y in {0,1,2,3}."""
    df = _read_or_fetch(path, "beed.csv", BEED_ID)
    _require(df, BEED_FEATURES + [BEED_TARGET], "BEED")
    if df[BEED_FEATURES + [BEED_TARGET]].isna().any().any():
        raise ValueError("BEED: unexpected missing values.")
    y = df[BEED_TARGET].astype(int)
    if set(y.unique()) - {0, 1, 2, 3}:
        raise ValueError(f"BEED: unexpected labels {sorted(y.unique())}")
    return df[BEED_FEATURES].astype(float), y


def load_actg175_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Full ACTG 175 table (all 25 columns), schema-checked. Used by the EDA."""
    df = _read_or_fetch(path, "actg175.csv", ACTG_ID)
    needed = (ACTG_NUMERIC + ACTG_BINARY + list(ACTG_CATEGORICAL) + ACTG_SENSITIVE
              + [ACTG_TARGET])
    _require(df, needed, "ACTG175")
    return df


def load_actg175(path: str | Path | None = None, include_sensitive: bool = False
                 ) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) to predict CD4 at ~20 weeks (``cd420``) from BASELINE information only.

    Predictors: age, weight, Karnofsky score, prior antiretroviral days, baseline CD4/CD8,
    binary clinical history flags and the treatment arm (``trt``) and antiretroviral history
    stratum (``strat``), both one-hot encoded.
    Excluded on purpose: see ``ACTG_EXCLUDED`` (post-baseline outcomes and identifiers) and,
    unless ``include_sensitive``, race / gender / sexual activity / IV drug use.
    Rows with missing values in any used column are dropped.
    """
    df = load_actg175_raw(path)
    sensitive = ACTG_SENSITIVE if include_sensitive else []
    cols = ACTG_NUMERIC + ACTG_BINARY + sensitive
    X = df[cols].astype(float).copy()
    for col, levels in ACTG_CATEGORICAL.items():
        cat = pd.Categorical(df[col], categories=levels)
        X = pd.concat([X, pd.get_dummies(cat, prefix=col, dtype=float).set_axis(df.index)],
                      axis=1)
    y = df[ACTG_TARGET].astype(float)
    keep = X.notna().all(axis=1) & y.notna()
    return X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)


@dataclass(frozen=True)
class Split:
    """Disjoint index sets for train / validation / test."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def stratified_split(y, val_size=0.15, test_size=0.15, seed=42) -> Split:
    """Stratified random split (classification)."""
    idx = np.arange(len(y))
    rest, test = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(rest, test_size=rel_val, stratify=np.asarray(y)[rest],
                                  random_state=seed)
    return Split(np.sort(train), np.sort(val), np.sort(test))


def random_split(n, val_size=0.15, test_size=0.15, seed=42) -> Split:
    """Plain random split (valid here: ACTG 175 has one row per patient)."""
    idx = np.arange(n)
    rest, test = train_test_split(idx, test_size=test_size, random_state=seed)
    train, val = train_test_split(rest, test_size=val_size / (1.0 - test_size), random_state=seed)
    return Split(np.sort(train), np.sort(val), np.sort(test))

