import numpy as np
import keras_tuner as kt
from sklearn.model_selection import TimeSeriesSplit
from core.logging import ProgressCallback

from core.lstm_utils import estimate_huber_delta, build_lstm_model


def estimate_delta_from_data(y_train):
    """
    Reuse shared estimate for Huber δ.
    """
    return estimate_huber_delta(y_train)


def build_model(hp, input_shape, huber_delta=None):
    """
    Delegate to shared LSTM builder.
    """
    return build_lstm_model(hp, input_shape, huber_delta)


def _cv_loss(model_fn, X, y, n_splits=3, epochs=5):
    """
    Walk-forward cross-validation loss (mean val_loss).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    losses = []

    for train_idx, val_idx in tscv.split(X):
        m = model_fn()
        m.fit(X[train_idx], y[train_idx], epochs=epochs, verbose=0)
        val_loss = m.evaluate(X[val_idx], y[val_idx], verbose=0)[0]
        losses.append(val_loss)

    return np.mean(losses)


def tune_and_train_model(X_train, y_train, input_shape, project_name="lstm_model"):
    """
    Efficient walk-forward hyperparameter tuning + final training.
    """
    huber_delta = estimate_delta_from_data(y_train)

    # Custom objective via cross-validation
    def objective_fn(hp):
        return _cv_loss(
            lambda: build_lstm_model(hp, input_shape, huber_delta),
            X_train, y_train,
            n_splits=3, epochs=5
        )

    tuner = kt.Hyperband(
        hypermodel=lambda hp: build_lstm_model(hp, input_shape, huber_delta),
        objective=kt.Objective("val_loss", direction="min"),  # placeholder
        max_epochs=30,
        factor=3,
        directory='lstm_tuner',
        project_name=project_name,
        overwrite=True
    )

    # Monkey‐patch Hyperband to use our CV objective
    tuner.oracle._objective = kt.engine.objective.Objective("custom_loss", "min")
    tuner._evaluate_trial = lambda trial: objective_fn(trial.hyperparameters)

    tuner.search_space_summary()
    tuner.search(x=np.zeros((1, *input_shape)), y=np.zeros((1,)), epochs=1)

    best_hp = tuner.get_best_hyperparameters(1)[0]
    model = build_lstm_model(best_hp, input_shape, huber_delta)

    # Final fit on full data
    early_stop = kt.callbacks.EarlyStopping(
        monitor='loss', patience=5, restore_best_weights=True
    )
    reduce_lr = kt.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5
    )
    progress_cb = ProgressCallback()

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        callbacks=[early_stop, reduce_lr, progress_cb],
        verbose=1
    )

    return model, history, best_hp
