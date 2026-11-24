import numpy as np
import pytest

from mlp_uci.models import (
    ARCHITECTURES,
    MAX_HIDDEN_LAYERS,
    OPTIMIZER_NAMES,
    build_shallow_mlp,
    make_optimizer,
)
from mlp_uci.training import DEFAULT_GRID, fit


def _dense_count(model) -> int:
    return sum(layer.__class__.__name__ == "Dense" for layer in model.layers)


def test_one_hidden_layer_outputs_probabilities():
    m = build_shallow_mlp(16, "classification", n_classes=4, hidden_layers=(8,), dropout=0.1,
                          l2=1e-3)
    assert _dense_count(m) == 2                          # one hidden + output
    p = m.predict(np.zeros((3, 16)), verbose=0)
    assert p.shape == (3, 4) and np.allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_two_hidden_layers_allowed_three_rejected():
    m = build_shallow_mlp(10, "regression", hidden_layers=(8, 4))
    assert _dense_count(m) == 3
    assert m.predict(np.zeros((5, 10)), verbose=0).shape == (5, 1)
    with pytest.raises(ValueError, match="hidden layers"):
        build_shallow_mlp(10, "regression", hidden_layers=(8, 4, 2))
    with pytest.raises(ValueError):
        build_shallow_mlp(10, "regression", hidden_layers=())


def test_assignment_requirements_are_met_by_default():
    # at least three architectures, none deeper than two hidden layers, all different
    assert len(ARCHITECTURES) >= 3 and len(set(ARCHITECTURES.values())) == len(ARCHITECTURES)
    assert all(1 <= len(v) <= MAX_HIDDEN_LAYERS for v in ARCHITECTURES.values())
    assert any(len(v) == 2 for v in ARCHITECTURES.values())
    # at least three different optimizers are searched by default
    assert len(set(DEFAULT_GRID["optimizer"])) >= 3
    assert set(DEFAULT_GRID["optimizer"]) <= set(OPTIMIZER_NAMES)


@pytest.mark.parametrize("name", OPTIMIZER_NAMES)
def test_every_optimizer_trains(name):
    opt = make_optimizer(name)
    assert opt.__class__.__name__.lower() == name
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] > 0).astype(int)
    m = build_shallow_mlp(5, "classification", n_classes=2, hidden_layers=(16,), optimizer=name)
    h = fit(m, X[:150], y[:150], X[150:], y[150:], max_epochs=25, patience=10)
    assert h.history["loss"][-1] < h.history["loss"][0]


def test_invalid_arguments():
    with pytest.raises(ValueError):
        build_shallow_mlp(4, "ranking")
    with pytest.raises(ValueError):
        build_shallow_mlp(4, "classification", n_classes=1)
    with pytest.raises(ValueError, match="unknown optimizer"):
        make_optimizer("lbfgs")
