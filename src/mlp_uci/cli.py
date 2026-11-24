"""Command line interface: ``mlp-uci classify`` and ``mlp-uci regress``."""
from __future__ import annotations

import argparse
import json

from . import data as D
from .eda import run_eda_actg175
from .experiments import RunConfig, run_classification, run_regression


def _common(p: argparse.ArgumentParser, default_out: str) -> None:
    p.add_argument("--data-path", default=None,
                   help="Local CSV. If omitted, the dataset is fetched from UCI and cached.")
    p.add_argument("--out", default=default_out, help="Output directory for metrics and figures.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-seeds", type=int, default=5, help="Weight-init seeds for mean +/- std.")
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--patience", type=int, default=20)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="mlp-uci", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("classify", help="BEED EEG epilepsy classification")
    _common(c, "reports/beed_classification")

    r = sub.add_parser("regress", help="Regression: ACTG 175 CD4 at 20 weeks")
    _common(r, "reports/actg175_regression")
    r.add_argument("--dataset", choices=["actg175"], default="actg175")
    r.add_argument("--split", choices=["default", "random"], default="default",
                   help="ACTG 175 has one row per patient, so a random split is leakage-free.")
    r.add_argument("--include-sensitive", action="store_true",
                   help="actg175: also use race, gender, homo, drugs as predictors.")

    e = sub.add_parser("eda", help="Exploratory analysis of ACTG 175 (figures + summary)")
    e.add_argument("--data-path", default=None)
    e.add_argument("--out", default="reports/actg175_eda")

    a = ap.parse_args(argv)
    if a.cmd == "eda":
        res = run_eda_actg175(D.load_actg175_raw(a.data_path), a.out)
        print(json.dumps({k: res[k] for k in ("n_patients", "corr_cd40_cd420", "change_by_arm")},
                         indent=2))
        return
    cfg = RunConfig(out_dir=a.out, data_path=a.data_path, seed=a.seed, n_seeds=a.n_seeds,
                    max_epochs=a.max_epochs, batch_size=a.batch_size, patience=a.patience)
    if a.cmd == "classify":
        res = run_classification(cfg)
    else:
        cfg.split = a.split
        cfg.dataset, cfg.include_sensitive = a.dataset, a.include_sensitive
        res = run_regression(cfg)
    print(json.dumps({k: res[k] for k in ("test_mlp", "baselines", "best_hyperparameters")},
                     indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
