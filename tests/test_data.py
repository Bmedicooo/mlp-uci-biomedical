import numpy as np
import pytest

from mlp_uci import data as D


def test_load_beed_schema(beed_csv):
    X, y = D.load_beed(beed_csv)
    assert list(X.columns) == D.BEED_FEATURES
    assert set(y.unique()) == {0, 1, 2, 3}


def test_load_beed_rejects_bad_schema(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="missing columns"):
        D.load_beed(p)


def _disjoint(s: D.Split, n: int):
    a, b, c = set(s.train), set(s.val), set(s.test)
    assert not (a & b) and not (a & c) and not (b & c)
    assert len(a | b | c) == n


def test_stratified_split_disjoint_and_stratified(beed_csv):
    _, y = D.load_beed(beed_csv)
    s = D.stratified_split(y.values, seed=1)
    _disjoint(s, len(y))
    for part in (s.train, s.val, s.test):
        counts = np.bincount(y.values[part], minlength=4)
        assert counts.max() - counts.min() <= 2          # classes stay balanced


def test_actg_excludes_post_baseline_and_sensitive_by_default(actg_csv):
    X, y = D.load_actg175(actg_csv)
    for col in D.ACTG_EXCLUDED:                      # outcomes / identifiers / redundant
        assert col not in X.columns
    for col in D.ACTG_SENSITIVE:
        assert col not in X.columns
    assert "cd420" not in X.columns and len(X) == len(y) == 400
    assert {"trt_0", "trt_1", "trt_2", "trt_3", "strat_1", "strat_2", "strat_3"} <= set(X.columns)
    assert X.notna().all().all()


def test_actg_include_sensitive_adds_columns(actg_csv):
    X, _ = D.load_actg175(actg_csv, include_sensitive=True)
    assert set(D.ACTG_SENSITIVE) <= set(X.columns)
    assert "cd820" not in X.columns                  # still excluded: measured after baseline


def test_actg_drops_rows_with_missing_target(tmp_path, actg_csv):
    import pandas as pd
    df = pd.read_csv(actg_csv)
    df.loc[:9, "cd420"] = np.nan
    p = tmp_path / "actg_na.csv"
    df.to_csv(p, index=False)
    X, y = D.load_actg175(p)
    assert len(X) == len(y) == 390


def test_actg_rejects_bad_schema(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="missing columns"):
        D.load_actg175(p)
