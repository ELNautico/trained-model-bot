"""
train/core.py
• HParam-tuning (dual-head Transformer: d1 & d5)
• 3-seed ensemble (per-head Average)
• Recursively renames all layers → no name collisions
• Versioned saving / loading
• Price reconstruction (uses d1 head)
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from core.transformer_utils import tune_transformer, build_transformer_model


# ─────────────────────────────────────────────────────────────────────────────
#  Helper – give every layer a unique suffix
# ─────────────────────────────────────────────────────────────────────────────
def _rename_all(layer: tf.keras.layers.Layer, suffix: str) -> None:
    layer._name = f"{layer.name}_{suffix}"
    if isinstance(layer, tf.keras.Model):
        for sub in layer.layers:
            _rename_all(sub, suffix)


# ─────────────────────────────────────────────────────────────────────────────
#  Train, ensemble & save
# ─────────────────────────────────────────────────────────────────────────────
def train_and_save_model(
    X_train,
    y_train,
    input_shape,
    ticker: str,
    *,
    n_seeds: int = 3,
):
    print(f"🔧 Hyper-parameter tuning for {ticker} …")
    _, best_hp = tune_transformer(
        X_train, y_train, input_shape, project_name=f"xf_{ticker}"
    )

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tb_root = Path("logs") / "training" / ticker / ts

    # ── train one seed
    def _fit_seed(seed: int) -> tf.keras.Model:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)

        m = build_transformer_model(best_hp, input_shape)
        _rename_all(m, f"seed{seed}")            # *** unique names ***

        callbacks = [
            TensorBoard(log_dir=str(tb_root / f"seed_{seed}"), histogram_freq=1),
            EarlyStopping("val_d1_mae", mode="min", patience=5, restore_best_weights=True),
            ReduceLROnPlateau("val_d1_mae", mode="min", factor=0.5, patience=3, min_lr=1e-5),
        ]
        m.fit(
            X_train,
            y_train,
            epochs=50,
            validation_split=0.2,
            verbose=1,
            callbacks=callbacks,
        )
        return m

    print("📈 Training seeds …")
    seed_models = [_fit_seed(s) for s in range(n_seeds)]

    # ── build ensemble (d1 & d5 only)
    ens_inp = tf.keras.Input(shape=input_shape, name="ensemble_input")
    d1_list, d5_list = [], []

    for mdl in seed_models:
        d1_pred, d5_pred = mdl(ens_inp)
        d1_list.append(d1_pred)
        d5_list.append(d5_pred)

    avg_d1 = tf.keras.layers.Average(name="d1")(d1_list)
    avg_d5 = tf.keras.layers.Average(name="d5")(d5_list)

    ensemble = tf.keras.Model(ens_inp, [avg_d1, avg_d5], name="xf_ensemble")
    ensemble.compile(
        optimizer="adam",
        loss=dict(d1="mse", d5="mse"),
        metrics=dict(d1=["mae"], d5=["mae"]),
    )

    # ── save
    out_dir = Path("models") / ticker / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    ensemble.save(out_dir / "model.h5")
    with open(out_dir / "hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(best_hp.values, f, indent=4)

    # Save metadata: features, metrics, hyperparameters
    metadata = {
        "timestamp": ts,
        "ticker": ticker,
        "features": list(X_train.shape[1:]),
        "hyperparameters": best_hp.values,
        "training_args": {
            "n_seeds": n_seeds,
            "epochs": 50,
            "validation_split": 0.2
        },
        "metrics": {
            "final_train_loss": float(seed_models[0].history.history["loss"][-1]) if hasattr(seed_models[0], "history") else None,
            "final_val_loss": float(seed_models[0].history.history["val_loss"][-1]) if hasattr(seed_models[0], "history") else None
        }
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"📝 Saved metadata to {out_dir / 'metadata.json'}")
    print(f"✅ Saved ensemble to {out_dir}")
    return ensemble, None, best_hp


# ─────────────────────────────────────────────────────────────────────────────
#  Loader
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ticker: str, *, version_timestamp: str | None = None) -> tf.keras.Model:
    """
    Return the newest saved ensemble for `ticker` (or a specific timestamp
    if `version_timestamp` is given).  Models are re-compiled so we don’t rely on
    any legacy aliases such as 'mse'.
    """
    base = Path("models") / ticker
    if version_timestamp is None:
        versions = sorted(base.iterdir(), reverse=True)
        if not versions:
            raise FileNotFoundError(f"No model for {ticker}")
        mdl_path = versions[0] / "model.h5"
    else:
        mdl_path = base / version_timestamp / "model.h5"

    # ❶  load *un-compiled* to avoid deserialising old losses/metrics
    model = tf.keras.models.load_model(mdl_path, compile=False)

    # ❷  compile with explicit objects that work in TF-/Keras 3+
    model.compile(
        optimizer="adam",
        loss={"d1": tf.keras.losses.MeanSquaredError(),
              "d5": tf.keras.losses.MeanSquaredError()},
        metrics={"d1": [tf.keras.metrics.MeanAbsoluteError()],
                 "d5": [tf.keras.metrics.MeanAbsoluteError()]},
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Price reconstruction (d1 head)
# ─────────────────────────────────────────────────────────────────────────────
def predict_price(model, X_last, current_price: float):
    d1_pred, _ = model.predict(X_last, verbose=0)
    price_hat = current_price * np.exp(float(d1_pred[0, 0]))
    return price_hat, 0.0   # no sigma head
