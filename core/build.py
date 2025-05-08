import numpy as np
import keras_tuner as kt
from sklearn.model_selection import TimeSeriesSplit

from core.lstm_utils import estimate_huber_delta, build_lstm_model


class TimeSeriesHyperband(kt.Hyperband):
    def run_trial(self, trial, X, y, **kwargs):
        hp = trial.hyperparameters
        huber_delta = estimate_huber_delta(y)

        # build model factory
        def mk_model():
            return build_lstm_model(hp, X.shape[1:], huber_delta)

        tscv = TimeSeriesSplit(n_splits=3)
        losses = []
        for train_idx, val_idx in tscv.split(X):
            m = mk_model()
            m.fit(X[train_idx], y[train_idx], epochs=10, verbose=0)
            losses.append(m.evaluate(X[val_idx], y[val_idx], verbose=0)[0])

        mean_loss = float(np.mean(losses))
        # report under the name your objective expects:
        self.oracle.update_trial(trial.trial_id, {"val_loss": mean_loss})

def tune_model(X_train, y_train, input_shape, project_name):
    huber_delta = estimate_huber_delta(y_train)
    tuner = TimeSeriesHyperband(
        hypermodel=lambda hp: build_lstm_model(hp, input_shape, huber_delta),
        objective=kt.Objective("val_loss", "min"),
        max_epochs=30, factor=3,
        directory="tuning_logs", project_name=project_name,
        overwrite=True,
    )
    # Pass X and y, our run_trial handles all CV
    tuner.search(X_train, y_train, verbose=1)
    best_hp = tuner.get_best_hyperparameters(1)[0]
    return build_lstm_model(best_hp, input_shape, huber_delta), best_hp

# --- Optional: Bayesian search variant for even faster convergence --- #

def bayesian_tune(X_train, y_train, input_shape, project_name, max_trials=20, init_points=5):
    """
    Tune hyperparameters with BayesianOptimization.
    Falls back to using built-in validation_split, but you could
    similarly subclass the Bayesian optimizer to inject CV.
    """
    huber_delta = estimate_huber_delta(y_train)

    tuner = kt.BayesianOptimization(
        hypermodel=lambda hp: build_lstm_model(hp, input_shape, huber_delta),
        objective=kt.Objective("val_loss", "min"),
        max_trials=max_trials,
        num_initial_points=init_points,
        directory="tuning_logs",
        project_name=project_name + "_bayes",
        overwrite=True,
    )

    tuner.search(
        X_train, y_train,
        validation_split=0.2,
        epochs=30,
        verbose=1,
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = build_lstm_model(best_hp, input_shape, huber_delta)
    return best_model, best_hp
