# core/transformer_utils.py
"""
Small dual-head Transformer for OHLCV windows
    • d1 – next-day log-return
    • d5 – 5-day  log-return

Includes a walk-forward Hyperband tuner.
"""

from __future__ import annotations

import uuid
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import keras_tuner as kt
from sklearn.model_selection import TimeSeriesSplit


# ──────────────────────────────────────────────
#  Encoder block
# ──────────────────────────────────────────────
def _enc_block(x, *, d_model: int, heads: int, dff: int, drop: float):
    attn = layers.MultiHeadAttention(
        num_heads=heads, key_dim=d_model, dropout=drop
    )(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    ffn = layers.Dense(dff, activation="relu")(x)
    ffn = layers.Dense(d_model)(ffn)
    ffn = layers.Dropout(drop)(ffn)

    x = layers.Add()([x, ffn])
    return layers.LayerNormalization(epsilon=1e-6)(x)


# ──────────────────────────────────────────────
#  Model factory
# ──────────────────────────────────────────────
def build_transformer_model(hp: kt.HyperParameters, input_shape):
    nlayers = hp.Int("layers", 1, 3)
    d_model = hp.Int("d_model", 32, 128, step=32)
    heads   = hp.Choice("heads", [2, 4, 8])
    dff     = hp.Int("dff", 64, 256, step=64)
    drop    = hp.Float("dropout", 0.0, 0.3, step=0.1)
    lr      = hp.Float("lr", 1e-4, 1e-2, sampling="log")

    inp = layers.Input(shape=input_shape, name="window")  # (T, F)
    x   = layers.Dense(d_model, name="proj")(inp)

    for _ in range(nlayers):
        x = _enc_block(x, d_model=d_model, heads=heads, dff=dff, drop=drop)

    x = layers.GlobalAveragePooling1D(name="gap")(x)

    d1 = layers.Dense(1, name="d1")(x)
    d5 = layers.Dense(1, name="d5")(x)

    # unique model name -> prevents collisions when models are merged
    uid   = uuid.uuid4().hex[:8]
    model = models.Model(inp, [d1, d5], name=f"xf_dual_{uid}")

    model.compile(
        optimizer=optimizers.Adam(lr),
        loss      = dict(d1="mse", d5="mse"),
        loss_weights = dict(d1=1.0, d5=0.5),
        metrics   = dict(d1=["mae"], d5=["mae"]),
    )
    return model


# ──────────────────────────────────────────────
#  Walk-forward Hyperband tuner
# ──────────────────────────────────────────────
class _WFHyperband(kt.Hyperband):
    """Hyperband with walk-forward CV (no data leakage)."""

    def run_trial(
        self,
        trial,
        X,
        y,
        input_shape,
        n_splits: int = 3,
        epochs_per_fold: int = 15,
        **__,
    ):
        hp        = trial.hyperparameters
        splitter  = TimeSeriesSplit(n_splits=n_splits)
        losses    = []

        for tr_idx, val_idx in splitter.split(X):
            m = build_transformer_model(hp, input_shape)
            m.fit(
                X[tr_idx],
                {k: v[tr_idx] for k, v in y.items()},
                epochs     = epochs_per_fold,
                verbose    = 0,
                callbacks  = [tf.keras.callbacks.EarlyStopping(patience=3,
                                                               restore_best_weights=True)],
            )
            loss = m.evaluate(
                X[val_idx],
                {k: v[val_idx] for k, v in y.items()},
                verbose=0,
            )[0]
            losses.append(loss)

        mean_loss = float(np.mean(losses))
        self.oracle.update_trial(trial.trial_id, {"val_loss": mean_loss})
        return {"val_loss": mean_loss}


# ──────────────────────────────────────────────
#  Public helper
# ──────────────────────────────────────────────
def tune_transformer(
    X_train,
    y_train,
    input_shape,
    project_name: str = "xf",
    n_splits: int = 3,
):
    tuner = _WFHyperband(
        hypermodel   = lambda hp: build_transformer_model(hp, input_shape),
        objective    = kt.Objective("val_loss", "min"),
        max_epochs   = 30,
        factor       = 3,
        directory    = "tuning_logs",
        project_name = project_name,
        overwrite    = True,
    )

    tuner.search(
        X_train,
        y_train,
        input_shape      = input_shape,   # forwarded to run_trial
        n_splits         = n_splits,
        epochs_per_fold  = 15,
        verbose          = 1,
    )

    best_hp    = tuner.get_best_hyperparameters(1)[0]
    best_model = build_transformer_model(best_hp, input_shape)
    return best_model, best_hp
