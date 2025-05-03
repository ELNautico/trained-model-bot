import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import numpy as np


def estimate_huber_delta(y_train):
    """
    Estimate a robust delta for the Huber loss function
    based on the distribution of training targets.
    """
    median_abs = np.median(np.abs(y_train - np.median(y_train)))
    return 5 * median_abs


def build_model(hp, input_shape, huber_delta=None):
    """
    Builds a compiled LSTM model using hyperparameters from keras-tuner.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    units = hp.Int('units', min_value=32, max_value=128, step=32)
    dropout = hp.Float('dropout', 0.0, 0.5, step=0.1)

    model.add(layers.LSTM(units, return_sequences=True))
    model.add(layers.Dropout(dropout))

    if hp.Boolean('use_second_lstm'):
        units2 = hp.Int('units2', 32, 128, step=32)
        model.add(layers.LSTM(units2))
        model.add(layers.Dropout(dropout))
    else:
        model.add(layers.LSTM(units))
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(1))

    lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    loss = tf.keras.losses.Huber(delta=huber_delta) if huber_delta else 'mse'

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=['mae', 'mape']
    )
    return model


def tune_model(X_train, y_train, input_shape, project_name):
    """
    Tunes hyperparameters using KerasTuner's Hyperband search.
    """
    huber_delta = estimate_huber_delta(y_train)

    tuner = kt.Hyperband(
        lambda hp: build_model(hp, input_shape, huber_delta),
        objective='val_loss',
        max_epochs=30,
        factor=3,
        executions_per_trial=2,
        directory='tuning_logs',
        project_name=project_name,
        overwrite=True
    )

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

    tuner.search(
        X_train, y_train,
        validation_split=0.2,
        epochs=30,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = build_model(best_hp, input_shape, huber_delta)
    return best_model, best_hp
