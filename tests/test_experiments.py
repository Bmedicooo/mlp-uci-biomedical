import json

import numpy as np

from mlp_uci.experiments import RunConfig, run_classification, run_regression

TINY_GRID = {"architecture": ["A_1x16", "B_1x128", "C_2x64_32"],
             "optimizer": ["adam", "sgd", "rmsprop"], "dropout": [0.0], "l2": [0.0]}


def test_classification_pipeline_end_to_end(beed_csv, tmp_path):
    cfg = RunConfig(out_dir=str(tmp_path / "clf"), data_path=str(beed_csv), n_seeds=2,
                    max_epochs=12, patience=4, grid=TINY_GRID)
    res = run_classification(cfg)
    assert res["test_mlp"]["accuracy"]["mean"] > 0.5      # synthetic classes are separable
    assert sum(res["split_sizes"].values()) == 600
    for f in ("metrics.json", "confusion_matrix.png", "learning_curves.png",
              "grid_search.csv", "channel_importance.png", "architectures.png",
              "grid_heatmap.png", "optimizer_curves.png"):
        assert (tmp_path / "clf" / f).exists()
    import pandas as pd
    grid = pd.read_csv(tmp_path / "clf" / "grid_search.csv")
    assert grid["optimizer"].nunique() == 3 and grid["architecture"].nunique() == 3
    assert set(res["search"]["by_optimizer"]) == {"adam", "sgd", "rmsprop"}
    json.loads((tmp_path / "clf" / "metrics.json").read_text())


def test_actg_regression_pipeline(actg_csv, tmp_path):
    cfg = RunConfig(out_dir=str(tmp_path / "actg"), data_path=str(actg_csv), n_seeds=2,
                    max_epochs=12, patience=4, grid=TINY_GRID)
    res = run_regression(cfg)
    assert res["split_mode"] == "random" and res["target"] == "cd420"
    assert "no_change (cd420 = cd40)" in res["baselines"]
    assert "cd820" not in res["features"] and "cid" not in res["features"]
    assert np.isfinite(res["test_mlp"]["rmse"]["mean"])
    for f in ("architectures.png", "grid_heatmap.png", "optimizer_curves.png", "metrics.json"):
        assert (tmp_path / "actg" / f).exists()


def test_actg_rejects_unknown_split(actg_csv, tmp_path):
    import pytest
    cfg = RunConfig(out_dir=str(tmp_path / "x"), data_path=str(actg_csv), split="group",
                    grid=TINY_GRID)
    with pytest.raises(ValueError, match="unknown split"):
        run_regression(cfg)


def test_eda_creates_figures_and_summary(actg_csv, tmp_path):
    from mlp_uci import data as D
    from mlp_uci.eda import run_eda_actg175
    res = run_eda_actg175(D.load_actg175_raw(actg_csv), tmp_path / "eda")
    assert res["n_patients"] == 400 and 0.5 < res["corr_cd40_cd420"] <= 1.0
    assert set(res["change_by_arm"]) == {"ZDV", "ZDV+ddI", "ZDV+Zal", "ddI"}
    for f in ("eda_target_distribution.png", "eda_cd40_vs_cd420.png",
              "eda_treatment_effect.png", "eda_correlations.png", "eda_summary.json"):
        assert (tmp_path / "eda" / f).exists()
