"""Shallow MLP builders (at most two hidden layers) in Keras."""
from __future__ import annotations

from collections.abc import Sequence

import keras
from keras import layers, regularizers

MAX_HIDDEN_LAYERS = 2

# Three architectures compared in the experiments (hidden layer sizes). Max. 2 hidden layers.
ARCHITECTURES: dict[str, tuple[int, ...]] = {
    "A_1x16": (16,),          # one small hidden layer
    "B_1x128": (128,),        # one wide hidden layer
    "C_2x64_32": (64, 32),    # two hidden layers
}

# Three optimizers compared in the experiments.
OPTIMIZER_NAMES = ("adam", "sgd", "rmsprop")
DEFAULT_LR = {"adam": 1e-3, "sgd": 1e-2, "rmsprop": 1e-3}


def make_optimizer(name: str, learning_rate: float | None = None) -> keras.optimizers.Optimizer:
    """Adam, SGD (with momentum 0.9) or RMSprop, each with a sensible default learning rate."""
    name = name.lower()
    if name not in OPTIMIZER_NAMES:
        raise ValueError(f"unknown optimizer {name!r}; choose from {OPTIMIZER_NAMES}")
    lr = DEFAULT_LR[name] if learning_rate is None else learning_rate
    if name == "adam":
        return keras.optimizers.Adam(lr)
    if name == "sgd":
        return keras.optimizers.SGD(lr, momentum=0.9)
    return keras.optimizers.RMSprop(lr)


def build_shallow_mlp(
    n_features: int,
    task: str,
    n_classes: int | None = None,
    hidden_layers: Sequence[int] | int = (64,),
    activation: str = "relu",
    dropout: float = 0.0,
    l2: float = 0.0,
    optimizer: str = "adam",
    learning_rate: float | None = None,
) -> keras.Model:
    """Input -> [Dense(h, activation, L2) -> Dropout?] x (1 or 2) -> output head.

    task="classification": softmax over ``n_classes`` + sparse categorical cross-entropy.
    task="regression": single linear unit + MSE (metrics: MAE, RMSE).
    ``hidden_layers`` is a tuple of layer widths with at most two entries (assignment limit).
    """
    if task not in {"classification", "regression"}:
        raise ValueError(f"unknown task {task!r}")
    sizes = (hidden_layers,) if isinstance(hidden_layers, int) else tuple(hidden_layers)
    if not 1 <= len(sizes) <= MAX_HIDDEN_LAYERS:
        raise ValueError(f"use 1 to {MAX_HIDDEN_LAYERS} hidden layers, got {len(sizes)}")

    reg = regularizers.l2(l2) if l2 > 0 else None
    inputs = keras.Input(shape=(n_features,))
    x = inputs
    for units in sizes:
        x = layers.Dense(units, activation=activation, kernel_regularizer=reg)(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)

    opt = make_optimizer(optimizer, learning_rate)
    if task == "classification":
        if not n_classes or n_classes < 2:
            raise ValueError("n_classes must be >= 2 for classification")
        outputs = layers.Dense(n_classes, activation="softmax")(x)
        model = keras.Model(inputs, outputs, name="mlp_clf")
        model.compile(optimizer=opt, loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
    else:
        outputs = layers.Dense(1)(x)
        model = keras.Model(inputs, outputs, name="mlp_reg")
        model.compile(optimizer=opt, loss="mse",
                      metrics=[keras.metrics.MeanAbsoluteError(name="mae"),
                               keras.metrics.RootMeanSquaredError(name="rmse")])
    return model
