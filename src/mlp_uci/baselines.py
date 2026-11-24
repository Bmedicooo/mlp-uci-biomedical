"""Reference models: a shallow MLP only means something next to simple baselines."""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from .evaluate import classification_metrics, regression_metrics


def classification_baselines(X_train, y_train, X_test, y_test, seed=0) -> dict:
    models = {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
    }
    out = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        proba = m.predict_proba(X_test)
        full = np.zeros((len(X_test), int(max(np.max(y_train), np.max(y_test))) + 1))
        full[:, m.classes_] = proba
        res = classification_metrics(y_test, full)
        out[name] = {k: res[k] for k in ("accuracy", "macro_f1", "roc_auc_ovr")}
    return out


def regression_baselines(X_train, y_train, X_test, y_test, seed=0) -> dict:
    models = {
        "mean_predictor": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
    }
    out = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        out[name] = regression_metrics(y_test, m.predict(X_test))
    return out
