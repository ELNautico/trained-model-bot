import numpy as np
import tensorflow as tf
from tensorflow.keras import models as keras_models, layers as keras_layers, callbacks as keras_callbacks
from sklearn.model_selection import TimeSeriesSplit
import keras_tuner as kt

from core.logging import ProgressCallback


def estimate_delta_from_data(y_train):
    """Estimate a good Huber delta based on typical percentage changes."""
    median_abs_return = np.median(np.abs(y_train))
    return 5 * median_abs_return


def build_model(hp, input_shape, huber_delta=None):
    model = keras_models.Sequential()
    model.add(keras_layers.Input(shape=input_shape))

    units = hp.Int('units', 32, 128, step=32)
    dropout_rate = hp.Float('dropout_rate', 0.0, 0.5, step=0.1)
    model.add(keras_layers.LSTM(units, return_sequences=True))
    model.add(keras_layers.Dropout(dropout_rate))

    if hp.Boolean('second_layer'):
        units2 = hp.Int('units2', 32, 128, step=32)
        model.add(keras_layers.LSTM(units2))
    else:
        model.add(keras_layers.LSTM(units))

    model.add(keras_layers.Dropout(dropout_rate))
    model.add(keras_layers.Dense(1))

    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    loss_fn = tf.keras.losses.Huber(delta=huber_delta) if huber_delta else 'mse'

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['mae', 'mape']
    )
    return model


def _cv_loss(model_fn, X, y, n_splits=3, epochs=5):
    """
    Walk-forward cross-validation loss (mean validation loss across splits).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    losses = []

    for train_idx, val_idx in tscv.split(X):
        model = model_fn()
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model.fit(X_train, y_train, epochs=epochs, verbose=0)
        val_loss = model.evaluate(X_val, y_val, verbose=0)[0]
        losses.append(val_loss)

    return np.mean(losses)


def tune_and_train_model(X_train, y_train, input_shape, project_name="lstm_model"):
    """
    Efficient walk-forward hyperparameter tuning + full retraining.
    """
    huber_delta = estimate_delta_from_data(y_train)

    # Define objective using CV
    def objective_fn(hp):
        return _cv_loss(
            lambda: build_model(hp, input_shape, huber_delta),
            X_train, y_train,
            n_splits=3, epochs=5
        )

    tuner = kt.Hyperband(
        hypermodel=lambda hp: build_model(hp, input_shape, huber_delta),
        objective=kt.Objective("val_loss", direction="min"),  # Placeholder
        max_epochs=30,
        factor=3,
        directory='lstm_tuner',
        project_name=project_name,
        overwrite=True
    )

    # Monkey patch to replace internal scoring with our custom function
    tuner.oracle._objective = kt.engine.objective.Objective("custom", "min")
    tuner._evaluate_trial = lambda trial: objective_fn(trial.hyperparameters)

    tuner.search_space_summary()

    # Run tuning (we won't use actual data, our _cv_loss handles training)
    tuner.search(x=np.zeros((1, *input_shape)), y=np.zeros((1,)), epochs=1)

    best_hp = tuner.get_best_hyperparameters(1)[0]
    model = build_model(best_hp, input_shape, huber_delta)

    # Final training with full data
    early_stop = keras_callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = keras_callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
    progress_cb = ProgressCallback()

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        callbacks=[early_stop, reduce_lr, progress_cb],
        verbose=1
    )

    return model, history, best_hp
