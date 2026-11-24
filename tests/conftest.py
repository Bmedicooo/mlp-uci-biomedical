"""Synthetic datasets with the SAME schema as the real UCI files (for fast, offline tests).

They are not real EEG / patient data: they only exercise the code paths.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def beed_csv(tmp_path_factory):
    rng = np.random.default_rng(0)
    n_per_class = 150
    frames = []
    for k in range(4):
        base = rng.normal(0, 1, (n_per_class, 16)) + k * 0.8
        df = pd.DataFrame(base, columns=[f"X{i}" for i in range(1, 17)])
        df["y"] = k
        frames.append(df)
    df = pd.concat(frames).sample(frac=1, random_state=0).reset_index(drop=True)
    path = tmp_path_factory.mktemp("data") / "BEED_Data.csv"
    df.to_csv(path, index=False)
    return path


ACTG_COLUMNS = ["pidnum", "cid", "time", "trt", "age", "wtkg", "hemo", "homo", "drugs", "karnof",
                "oprior", "z30", "zprior", "preanti", "race", "gender", "str2", "strat",
                "symptom", "treat", "offtrt", "cd40", "cd420", "cd80", "cd820"]


@pytest.fixture(scope="session")
def actg_csv(tmp_path_factory):
    """Synthetic table with the real ACTG 175 column names (not real patient data)."""
    rng = np.random.default_rng(2)
    n = 400
    trt = rng.integers(0, 4, n)
    cd40 = rng.normal(350, 120, n).clip(50, 900)
    cd80 = rng.normal(900, 300, n).clip(100, 3000)
    cd420 = 0.8 * cd40 + 25 * (trt > 0) + rng.normal(0, 40, n)
    df = pd.DataFrame({
        "pidnum": np.arange(n), "cid": rng.integers(0, 2, n), "time": rng.integers(100, 1200, n),
        "trt": trt, "age": rng.integers(18, 70, n), "wtkg": rng.normal(75, 12, n),
        "hemo": rng.integers(0, 2, n), "homo": rng.integers(0, 2, n),
        "drugs": rng.integers(0, 2, n), "karnof": rng.choice([70, 80, 90, 100], n),
        "oprior": rng.integers(0, 2, n), "z30": rng.integers(0, 2, n),
        "zprior": np.ones(n, dtype=int), "preanti": rng.integers(0, 1500, n),
        "race": rng.integers(0, 2, n), "gender": rng.integers(0, 2, n),
        "str2": rng.integers(0, 2, n), "strat": rng.integers(1, 4, n),
        "symptom": rng.integers(0, 2, n), "treat": (trt > 0).astype(int),
        "offtrt": rng.integers(0, 2, n), "cd40": cd40.round(), "cd420": cd420.round(),
        "cd80": cd80.round(), "cd820": (cd80 * 1.1).round(),
    })[ACTG_COLUMNS]
    path = tmp_path_factory.mktemp("data") / "actg175.csv"
    df.to_csv(path, index=False)
    return path
