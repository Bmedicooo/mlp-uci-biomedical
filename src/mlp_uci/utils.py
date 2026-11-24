"""Small helpers: reproducibility and I/O."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Keras/TensorFlow for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import keras

    keras.utils.set_random_seed(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def save_json(obj: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_jsonable)
