import numpy as np
import keras_tuner as kt
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from pathlib import Path
from datetime import datetime

from core.lstm_utils import estimate_huber_delta, build_lstm_model


class TimeSeriesHyperband(kt.Hyperband):
    def run_trial(self, trial, X, y, **kwargs):
        hp = trial.hyperparameters
        huber_delta = estimate_huber_delta(y)

        # build model factory
        def mk_model():
            return build_lstm_model(hp, X.shape[1:], huber_delta)

        # cross‐validation
        tscv = TimeSeriesSplit(n_splits=3)
        losses = []
        for train_idx, val_idx in tscv.split(X):
            m = mk_model()
            m.fit(
                X[train_idx], y[train_idx],
                epochs=10, verbose=0
            )
            losses.append(
                m.evaluate(X[val_idx], y[val_idx], verbose=0)[0]
            )

    mean_loss = float(np.mean(losses))
    # report to tuner under the objective name
    self.oracle.update_trial(trial.trial_id, {"loss": mean_loss})


def tune_model(X_train, y_train, input_shape, project_name):
    huber_delta = estimate_huber_delta(y_train)

    # prepare a unique TensorBoard logdir for this tuning run
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tb_logdir = Path("logs") / "tuner" / project_name / timestamp
    tb_logdir.mkdir(parents=True, exist_ok=True)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=str(tb_logdir))

    tuner = TimeSeriesHyperband(
        hypermodel=lambda hp: build_lstm_model(hp, input_shape, huber_delta),
        objective=kt.Objective("val_loss", "min"),
        max_epochs=30,
        factor=3,
        directory="tuning_logs",
        project_name=project_name,
        overwrite=True,
    )

    # pass TensorBoard callback into search
    tuner.search(
        X_train,
        y_train,
        verbose=1,
        callbacks=[tb_callback]
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = build_lstm_model(best_hp, input_shape, huber_delta)
    return best_model, best_hp


def bayesian_tune(X_train, y_train, input_shape, project_name, max_trials=20, init_points=5):
    """
    Bayesian search with TensorBoard logging.
    """
    huber_delta = estimate_huber_delta(y_train)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tb_logdir = Path("logs") / "tuner" / f"{project_name}_bayes" / timestamp
    tb_logdir.mkdir(parents=True, exist_ok=True)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=str(tb_logdir))

    tuner = kt.BayesianOptimization(
        hypermodel=lambda hp: build_lstm_model(hp, input_shape, huber_delta),
        objective=kt.Objective("loss", "min"),
        max_trials=max_trials,
        num_initial_points=init_points,
        directory="tuning_logs",
        project_name=project_name + "_bayes",
        overwrite=True,
    )

    tuner.search(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=30,
        verbose=1,
        callbacks=[tb_callback]
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = build_lstm_model(best_hp, input_shape, huber_delta)
    return best_model, best_hp
