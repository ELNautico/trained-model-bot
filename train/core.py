
from __future__ import annotations
import pandas as pd
from core.sensitivity import compute_feature_sensitivity
from core.features import get_feature_columns
"""
train/core.py
• HParam-tuning (dual-head Transformer: d1 & d5)
• 3-seed ensemble (per-head Average)
• Recursively renames all layers → no name collisions
• Versioned saving / loading
• Price reconstruction (uses d1 head)
"""

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


import re
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
    # Ensure enough samples for cross-validation
    min_samples = 2  # Number of folds for CV
    if X_train.shape[0] < min_samples:
        print(f"Skipping training: not enough samples for CV (required={min_samples}, got={X_train.shape[0]})")
        return None, None, None
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
    
    # NEW: Create ACTIVE symlink pointing to latest version
    model_base = Path("models") / ticker
    active_link = model_base / "ACTIVE"
    version_dir_name = ts  # timestamp from earlier
    
    # Remove old symlink if exists
    if active_link.exists() or active_link.is_symlink():
        active_link.unlink()
    
    # Create new symlink (relative, for portability)
    try:
        active_link.symlink_to(version_dir_name, target_is_directory=True)
        print(f"✅ Updated ACTIVE symlink → {version_dir_name}")
    except OSError as e:
        # Windows may require admin rights; fallback to marker file
        print(f"⚠️  Could not create symlink: {e}")
        print("📝 Writing ACTIVE marker file instead")
        active_link.write_text(version_dir_name, encoding="utf-8")

    # Feature sensitivity & pruning (save pruned features)
    from train.pipeline import save_pruned_features, build_sequences

    all_feats = get_feature_columns()
    if len(all_feats) > 10:
        last_seq = X_train[-1:][...]
        sens = compute_feature_sensitivity(ensemble, last_seq, all_feats)
        PRUNE_PCT = 30
        thresh = np.percentile(list(sens.values()), PRUNE_PCT)
        weak = [f for f, v in sens.items() if v <= thresh]
        if weak and len(weak) < len(all_feats) * 0.5:
            print(f"✂️  Pruning {len(weak)} weak features: {weak}")
            feats_pruned = [f for f in all_feats if f not in weak]
            win = 60 if X_train.shape[0] >= 600 else 30
            # Rebuild sequences with pruned features
            from train.pipeline import cached_download, enrich_features
            df_full = enrich_features(cached_download(ticker))
            X_tr_p, y_tr_p, _, _, _ = build_sequences(
                df_full, feats_pruned, window_size=win
            )
            # Retrain ensemble with pruned features
            pruned_seed_models = []
            for s in range(n_seeds):
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(s)
                m = build_transformer_model(best_hp, X_tr_p.shape[1:])
                _rename_all(m, f"pruned_seed{s}")
                m.fit(
                    X_tr_p,
                    [y_tr_p["d1"], y_tr_p["d5"]],
                    epochs=50,
                    validation_split=0.2,
                    verbose=1,
                )
                pruned_seed_models.append(m)

            ens_inp_p = tf.keras.Input(shape=X_tr_p.shape[1:], name="ensemble_input_pruned")
            d1_list_p, d5_list_p = [], []
            for mdl in pruned_seed_models:
                d1_pred, d5_pred = mdl(ens_inp_p)
                d1_list_p.append(d1_pred)
                d5_list_p.append(d5_pred)
            avg_d1_p = tf.keras.layers.Average(name="d1")(d1_list_p)
            avg_d5_p = tf.keras.layers.Average(name="d5")(d5_list_p)
            ensemble_pruned = tf.keras.Model(ens_inp_p, [avg_d1_p, avg_d5_p], name="xf_ensemble_pruned")
            ensemble_pruned.compile(
                optimizer="adam",
                loss=dict(d1="mse", d5="mse"),
                metrics=dict(d1=["mae"], d5=["mae"]),
            )
            # Save pruned features
            save_pruned_features(ticker, weak, out_dir)
            print(f"📝 Saved pruned features to {out_dir / 'pruned_features.json'} and LATEST_pruned_features.json")
    return ensemble, None, best_hp


# ─────────────────────────────────────────────────────────────────────────────
#  Loader
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ticker: str, *, version_timestamp: str | None = None) -> tf.keras.Model:
    """
    Load the ACTIVE model for a ticker (or specific version if provided).
    Falls back to latest timestamp if ACTIVE doesn't exist.
    """
    base = Path("models") / ticker
    
    if version_timestamp is None:
        # Try ACTIVE first
        active_link = base / "ACTIVE"
        
        if active_link.exists() or active_link.is_symlink():
            try:
                if active_link.is_symlink():
                    version_dir = active_link.resolve()
                    mdl_path = version_dir / "model.h5"
                    if mdl_path.exists():
                        print(f"📦 Loading ACTIVE model (symlink): {version_dir.name}")
                    else:
                        raise FileNotFoundError(f"Model file not found at {mdl_path}")
                else:
                    # Marker file fallback (for Windows)
                    version_name = active_link.read_text(encoding="utf-8").strip()
                    mdl_path = base / version_name / "model.h5"
                    if mdl_path.exists():
                        print(f"📦 Loading ACTIVE model (marker): {version_name}")
                    else:
                        raise FileNotFoundError(f"Model file not found at {mdl_path}")
            except Exception as e:
                print(f"⚠️  ACTIVE link exists but failed to load: {e}")
                # Fall through to timestamp search
                ts_pattern = re.compile(r"^\d{8}_\d{6}$")
                versions = [d for d in base.iterdir() if d.is_dir() and ts_pattern.match(d.name)]
                versions = sorted(versions, reverse=True)
                if not versions:
                    raise FileNotFoundError(f"No model for {ticker}")
                mdl_path = versions[0] / "model.h5"
                print(f"⚠️  Using latest fallback: {versions[0].name}")
        else:
            # No ACTIVE, use latest timestamp
            ts_pattern = re.compile(r"^\d{8}_\d{6}$")
            versions = [d for d in base.iterdir() if d.is_dir() and ts_pattern.match(d.name)]
            versions = sorted(versions, reverse=True)
            if not versions:
                raise FileNotFoundError(f"No model for {ticker}")
            mdl_path = versions[0] / "model.h5"
            print(f"⚠️  ACTIVE link missing; using latest: {versions[0].name}")
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
