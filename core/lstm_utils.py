import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers, optimizers, losses


def estimate_huber_delta(y_train: np.ndarray) -> float:
    """
    Estimate a robust delta for the Huber loss function
    based on the distribution of training targets.
    """
    # median absolute deviation
    med = np.median(y_train)
    mad = np.median(np.abs(y_train - med))
    return 5 * mad


def build_lstm_model(hp, input_shape, huber_delta=None):
    """
    Build a compiled LSTM model parameterized by a KerasTuner HyperParameters object.
    Hyperparameter names:
      - units            : number of LSTM units in first layer
      - dropout_rate     : dropout rate after each LSTM
      - use_second_lstm  : whether to add a second LSTM layer
      - units2           : if second LSTM, its units
      - learning_rate    : Adam learning rate (log sampling)
    """
    model = Sequential([layers.Input(shape=input_shape)])

    # shared hyperparameter names
    units = hp.Int("units", 32, 128, step=32)
    dropout_rate = hp.Float("dropout_rate", 0.0, 0.5, step=0.1)

    # first LSTM + dropout
    model.add(layers.LSTM(units, return_sequences=True))
    model.add(layers.Dropout(dropout_rate))

    # optional second LSTM
    if hp.Boolean("use_second_lstm"):
        units2 = hp.Int("units2", 32, 128, step=32)
        model.add(layers.LSTM(units2))
        model.add(layers.Dropout(dropout_rate))
    else:
        # reuse `units` if no second layer
        model.add(layers.LSTM(units))
        model.add(layers.Dropout(dropout_rate))

    # final dense
    model.add(layers.Dense(1))

    # compile
    lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
    loss_fn = losses.Huber(delta=huber_delta) if huber_delta else "mse"
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
        metrics=["mae", "mape"],
    )
    return model
