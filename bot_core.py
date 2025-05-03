import os
import time
import logging
import yfinance as yf
import pandas as pd
import numpy as np
import math
from pathlib import Path

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

import requests
from bs4 import BeautifulSoup

import tensorflow as tf
import tensorflow.keras.models as keras_models
import tensorflow.keras.layers as keras_layers
import tensorflow.keras.callbacks as keras_callbacks
from tensorflow.keras.losses import Huber
from sklearn.mixture import GaussianMixture
import keras_tuner as kt

from storage import init_db, get_recent_errors

# ------------------------------
# Attempt to import FearAndGreedIndex; if not available, define a fallback.
# ------------------------------
def get_fear_and_greed_index():
    try:
        url = "https://edition.cnn.com/markets/fear-and-greed"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            raise ValueError(f"HTTP {response.status_code} error when fetching Fear and Greed Index")

        soup = BeautifulSoup(response.text, 'html.parser')
        gauge = soup.find('div', class_='FearGreedIndicator__Dial-value')
        status = soup.find('div', class_='FearGreedIndicator__Dial-status')

        if gauge and status:
            value = int(gauge.text.strip())
            status_text = status.text.strip()
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
            return value, status_text, timestamp
        else:
            raise ValueError("Could not find Fear & Greed elements on page")

    except Exception as e:
        logging.error(f"Error fetching Fear and Greed index: {e}")
        # Fallback to safe values
        return "N/A", "N/A", datetime.now().strftime('%Y-%m-%d %H:%M')

# ------------------------------
# Logging Configuration
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[logging.FileHandler("trading_log.txt"), logging.StreamHandler()]
)

def setup_training_logger(ticker: str) -> logging.Logger:
    """
    Create a separate logger for each training session.
    Output will go into logs/{ticker}_train_{timestamp}.txt
    """
    # Create logs/ directory if missing
    os.makedirs("logs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"logs/{ticker}_train_{timestamp}.txt"

    # Setup file handler for this training session
    file_handler = logging.FileHandler(filename, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Create a dedicated logger
    logger = logging.getLogger(f"training_{ticker}_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # (Optional) also show info on console (useful if you want to see during training)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    logger.propagate = False  # prevent double printing to console

    return logger

# ------------------------------
# Additional Technical Indicators
# ------------------------------
def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    delta = data['Close'].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    data['RSI_14'] = 100 - 100/(1 + gain.rolling(14).mean()/(loss.rolling(14).mean()+1e-10))
    data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
    data['ATR'] = data['High'].rolling(14).max() - data['Low'].rolling(14).min()
    data['ROC'] = data['Close'].pct_change(10)
    data.dropna(inplace=True)
    return data

# ------------------------------
# Walk-Forward Cross-Validation Tuner
# ------------------------------
def _cv_loss(model_fn, X, y, tscv):
    errs = []
    for train_idx, val_idx in tscv.split(X):
        m = model_fn()
        Xtr, ytr = X[train_idx], y[train_idx]
        Xv, yv = X[val_idx], y[val_idx]
        m.fit(Xtr, ytr, epochs=10, verbose=0)
        errs.append(m.evaluate(Xv, yv, verbose=0)[0])
    return np.mean(errs)

def walk_forward_tuner(X: np.ndarray, y: np.ndarray, input_shape: tuple, project_name: str):
    tscv = TimeSeriesSplit(n_splits=5)
    huber_delta = estimate_delta_from_data(y)

    tuner = kt.Hyperband(
        lambda hp: build_model(hp, input_shape, huber_delta=huber_delta),
        objective=lambda m, _x, _y: _cv_loss(m, X, y, tscv),
        max_epochs=30,
        factor=3,
        directory='tuner',
        project_name=project_name
    )
    tuner.search(x=None, y=None)
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = build_model(best_hp, input_shape, huber_delta=huber_delta)
    return model, best_hp

# ------------------------------
# Regular Retraining Placeholder
# ------------------------------
def should_retrain(max_error_threshold: float = 2.0, window: int = 7) -> bool:
    """
    Check recent evaluation errors; retrain if avg absolute pct_error > threshold.
    """
    init_db()
    errors = get_recent_errors(window)
    if not errors:
        return True
    avg_err = np.mean(np.abs(errors))
    logging.info(f"Avg pct_error over last {window} days: {avg_err:.2f}%")
    return avg_err > max_error_threshold

# ------------------------------
# Model Interpretability: Feature Sensitivity
# ------------------------------
def compute_feature_sensitivity(model, sample, scaler, feature_names):
    """
    Compute sensitivity analysis using TensorFlow's GradientTape.
    Returns a dictionary mapping each feature name to the mean absolute gradient.
    """
    sample_tensor = tf.convert_to_tensor(sample, dtype=tf.float32)
    sample_tensor = tf.Variable(sample_tensor)
    with tf.GradientTape() as tape:
        prediction = model(sample_tensor)
    grads = tape.gradient(prediction, sample_tensor)
    avg_abs_grads = tf.reduce_mean(tf.abs(grads), axis=1).numpy().flatten()
    sensitivity = dict(zip(feature_names, avg_abs_grads))
    return sensitivity

# ------------------------------
# Custom Callback for Progress Indicator with Timing
# ------------------------------
class ProgressCallback(keras_callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []
        self.total_epochs = self.params.get("epochs", 0)
        logging.info(f"Starting training for {self.total_epochs} epochs...")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.epoch_start_time
        self.epoch_times.append(elapsed)
        avg_epoch_time = np.mean(self.epoch_times)
        epochs_remaining = self.total_epochs - (epoch + 1)
        eta = epochs_remaining * avg_epoch_time

        loss = logs.get("loss", 0.0)
        val_loss = logs.get("val_loss", 0.0)
        mape = logs.get("mape", None)
        val_mape = logs.get("val_mape", None)

        # Detect if MAPE is unrealistic
        if mape is not None and abs(mape) > 1000:
            mape = None
        if val_mape is not None and abs(val_mape) > 1000:
            val_mape = None

        log_msg = (
            f"Epoch {epoch + 1:>2}/{self.total_epochs}: "
            f"loss = {loss:.4f}, val_loss = {val_loss:.4f}, "
            f"epoch time = {elapsed:.2f}s, ETA = {eta:.2f}s"
        )

        if mape is not None and val_mape is not None:
            log_msg += f", mape = {mape:.1f}, val_mape = {val_mape:.1f}"

        logging.info(log_msg)

# ------------------------------
# Robust Daily Data Download with Retries
# ------------------------------
def download_daily_data(ticker, retries=3, delay=5):
    """
    Downloads daily data for the given ticker spanning 30 years.
    Removes duplicate rows and retries if no data is returned.
    """
    from datetime import datetime, timedelta

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 30)
    attempt = 0
    while attempt < retries:
        data = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1d",
            timeout=30
        )
        if not data.empty:
            data = data[~data.index.duplicated(keep='last')]
            return data
        else:
            logging.warning(f"Attempt {attempt+1} for ticker {ticker} returned no data. Retrying in {delay} seconds...")
            time.sleep(delay)
            attempt += 1
    raise ValueError(f"No daily data returned for {ticker} after {retries} attempts.")

# ------------------------------------------------------------------
# Prepare data when you want to use *all* rows for training
# ------------------------------------------------------------------
def prepare_data_full(data: pd.DataFrame, window_size: int = 60):
    """
    Adds indicators (basic + richer), scales on the full dataframe, and returns sequences X, y, scaler.
    No train/test split – used for final live retraining.
    """
    data = add_technical_indicators(data)
    data = add_richer_features(data)

    feature_cols = [
        'Open','High','Low','Close','Volume',
        'SMA_20','SMA_50','RSI_14','MACD','ATR','ROC',
        'BB_mid','BB_upper','BB_lower',
        'Stoch_%K','Stoch_%D',
        'OBV','VWAP','CMF','Volatility_Regime'
    ]
    df = data[feature_cols].copy().ffill()

    if len(df) < window_size + 1:
        raise ValueError("Not enough data to form a single sequence.")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i - window_size:i, :])
        # 🟢 Use **actual price** as target
        y.append(df.iloc[i]['Close'])

    return np.asarray(X), np.asarray(y), scaler

# ------------------------------
# Baseline Forecast Model
# ------------------------------
def baseline_forecast(data):
    """Generates a baseline forecast using the 20-day SMA."""
    return data['Close'].iloc[-1].item()

# Makes Huber loss function much more adaptive to actual data behavior
def estimate_delta_from_data(y_train):
    """
    Estimate a good Huber delta based on typical percentage changes.
    """
    median_abs_return = np.median(np.abs(y_train))
    delta = 5 * median_abs_return
    return delta

# ------------------------------
# Model Building and Hyperparameter Tuning
# ------------------------------
def build_model(hp, input_shape, huber_delta=None):
    model = keras_models.Sequential()
    model.add(keras_layers.Input(shape=input_shape))
    units = hp.Int('units', min_value=32, max_value=128, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)

    model.add(keras_layers.LSTM(units, return_sequences=True))
    model.add(keras_layers.Dropout(dropout_rate))

    if hp.Boolean('second_layer'):
        units2 = hp.Int('units2', min_value=32, max_value=128, step=32)
        model.add(keras_layers.LSTM(units2))
        model.add(keras_layers.Dropout(dropout_rate))
    else:
        model.add(keras_layers.LSTM(units))
        model.add(keras_layers.Dropout(dropout_rate))

    model.add(keras_layers.Dense(1))

    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')

    # ⚡ Here dynamically switch the loss:
    if huber_delta is not None:
        loss_fn = tf.keras.losses.Huber(delta=huber_delta)
    else:
        loss_fn = 'mse'

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['mae', 'mape']
    )
    return model

def tune_and_train_model(X_train, y_train, input_shape, project_name="lstm_model"):
    # NEW: per-ticker training log
    log_filename = f"logs/training_{project_name}.log"
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    logging.getLogger().addHandler(file_handler)

    try:
        # everything else unchanged...
        huber_delta = estimate_delta_from_data(y_train)

        tuner = kt.Hyperband(
            lambda hp: build_model(hp, input_shape, huber_delta=huber_delta),
            objective='val_loss',
            max_epochs=30,
            factor=3,
            directory='lstm_tuner',
            project_name=project_name,
            overwrite=True,
            executions_per_trial=2
        )

        early_stop = keras_callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        reduce_lr = keras_callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        )
        progress_cb = ProgressCallback()

        logging.info(f" Tuning {project_name} with shape {input_shape} and {len(X_train)} samples")
        tuner.search(
            X_train, y_train,
            epochs=30,
            validation_split=0.2,
            callbacks=[early_stop, reduce_lr, progress_cb],
            verbose=1
        )

        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        logging.info(f" Best HP: {best_hp.values}")

        model = build_model(best_hp, input_shape, huber_delta=huber_delta)
        logging.info(" Training final model...")
        history = model.fit(
            X_train, y_train,
            epochs=30,
            validation_split=0.2,
            callbacks=[early_stop, reduce_lr, progress_cb],
            verbose=1
        )
        return model, history, best_hp

    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()

# ------------------------------
# Additional Risk Control
# ------------------------------
def adjust_leverage_for_volatility(leverage, volatility, volatility_threshold=0.02):
    vol = float(volatility)
    if vol > volatility_threshold:
        adjusted = leverage * (volatility_threshold / vol)
        return max(2, round(adjusted))
    else:
        return leverage

# ------------------------------
# Certificate Recommendation
# ------------------------------
def calculate_optimal_leverage(current_price, predicted_price, cert_type, knockout_buffer=0.10):
    current_price = float(current_price)
    predicted_price = float(predicted_price)
    pct_change = (predicted_price / current_price - 1) * 100
    if abs(pct_change) < 0.01:
        return 2
    scaling_factor = 50
    candidate = round(scaling_factor / abs(pct_change))
    candidate = max(2, min(candidate, 40))

    if cert_type == "Call":
        knockout_barrier = current_price * (1 - knockout_buffer)
        profit_margin = predicted_price - current_price
        knockout_gap = current_price - knockout_barrier
    elif cert_type == "Put":
        knockout_barrier = current_price * (1 + knockout_buffer)
        profit_margin = current_price - predicted_price
        knockout_gap = knockout_barrier - current_price
    else:
        return candidate

    risk_ratio = profit_margin / knockout_gap if knockout_gap != 0 else 0
    threshold_low, threshold_high = 1.0, 2.0
    if risk_ratio < threshold_low:
        adjusted_candidate = 2
    elif risk_ratio < threshold_high:
        factor = (risk_ratio - threshold_low) / (threshold_high - threshold_low)
        adjusted_candidate = 2 + factor * (candidate - 2)
    else:
        adjusted_candidate = candidate

    allowed_levels = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40]
    optimal = min(allowed_levels, key=lambda x: abs(x - adjusted_candidate))
    return optimal

def generate_certificate(current_price, predicted_price, volatility):
    current_price = float(current_price)
    predicted_price = float(predicted_price)
    if predicted_price > current_price:
        cert_type = "Call"
        cert_desc = "Call Certificate (Bullish)"
    elif predicted_price < current_price:
        cert_type = "Put"
        cert_desc = "Put Certificate (Bearish)"
    else:
        cert_type = "Neutral"
        cert_desc = "Neutral"

    base_leverage = calculate_optimal_leverage(current_price, predicted_price, cert_type)
    adjusted_leverage = adjust_leverage_for_volatility(base_leverage, volatility)
    return f"{cert_desc} with recommended leverage {adjusted_leverage}x"

# ------------------------------
# Trading Signal and Risk Management Functions
# ------------------------------
def determine_trade_signal(predicted_pct_change, trend_signal, current_RSI, volume, min_threshold=0.5):
    """
    Determines a trade signal based on multiple criteria.
    Returns a tuple: (trade_decision, confidence) => "Buy", "Sell", or "No Trade".
    """
    abs_change = abs(predicted_pct_change)
    if abs_change < min_threshold:
        return "No Trade", 0

    confidence = min(100, abs_change * 10)
    if (predicted_pct_change > 0) and (trend_signal == "Bullish") and (current_RSI < 70):
        return "Buy", confidence
    elif (predicted_pct_change < 0) and (trend_signal == "Bearish") and (current_RSI > 30):
        return "Sell", confidence
    else:
        return "No Trade", confidence

def determine_position_size(account_balance, risk_per_trade, volatility):
    base_position = account_balance * risk_per_trade
    vol = float(volatility) if volatility > 0 else 0.01
    position_size = base_position / vol
    return position_size

# ------------------------------
# Backtesting Function
# ------------------------------
def backtest_strategy(model, X_test, scaler, data, window_size, train_size):
    """
    Evaluates directional accuracy and realistic cumulative return.
    Handles clipping and avoids numerical overflow.
    """
    predicted = model.predict(X_test, verbose=0).flatten()

    # Rescale to original Close prices
    close_min = scaler.data_min_[3]
    close_max = scaler.data_max_[3]
    predicted_prices = predicted * (close_max - close_min) + close_min

    split_index = len(data) - (len(X_test) + window_size)
    test_indices = np.arange(split_index + window_size, len(data))

    actual_prices = data['Close'].iloc[test_indices].values
    prev_prices = data['Close'].iloc[test_indices - 1].values

    actual_returns = (actual_prices / prev_prices) - 1.0
    predicted_returns = (predicted_prices / actual_prices) - 1.0

    # Directional accuracy
    directional_accuracy = np.mean(np.sign(predicted_returns) == np.sign(actual_returns))

    # Strategy: long if model predicts up, else short
    raw_strategy_returns = np.where(predicted_returns > 0, actual_returns, -actual_returns)

    # Clip returns to [-99%, +99%] to prevent log(0) or exp(huge)
    raw_strategy_returns = np.clip(raw_strategy_returns, -0.99, 0.99)
    log_returns = np.log1p(raw_strategy_returns)
    cumulative_return = np.expm1(np.sum(log_returns))

    return directional_accuracy, cumulative_return

# ------------------------------
# Main Trading Function: train_predict_for_ticker
# ------------------------------

import matplotlib.pyplot as plt

def plot_predictions(predicted_prices, actual_prices):
    plt.figure(figsize=(12, 5))
    plt.plot(actual_prices, label="Actual", linewidth=2)
    plt.plot(predicted_prices, label="Predicted", linewidth=2)
    plt.title("Model Predictions vs Actual Close Prices")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def train_predict_for_ticker(ticker, use_ensemble=True, account_balance=100000, risk_per_trade=0.01):
    """
    Full pipeline for training and predicting for a ticker.
    Logs each training into a dedicated file under 'logs/' folder.
    """
    logger = setup_training_logger(ticker)
    logger.info(f" Starting training for {ticker}...")

    try:
        data = download_daily_data(ticker)
        logger.info(f" Downloaded {len(data)} rows for {ticker}")
    except Exception as e:
        logger.error(f" Error downloading data for {ticker}: {e}")
        raise

    # Adaptive window size
    n_rows = len(data)
    if n_rows >= 600:
        window_size = 60
    elif n_rows >= 400:
        window_size = 45
    else:
        window_size = max(30, n_rows // 10)
    logger.info(f" Using window_size={window_size} based on {n_rows} rows.")

    try:
        X_train, y_train, X_test, y_test, scaler, train_size = prepare_data_and_split_with_richer(
            data,
            window_size=window_size,
            test_ratio=0.2 if n_rows >= 500 else 0.1
        )
        logger.info(f" Prepared training/testing sequences. Train size: {len(X_train)}, Test size: {len(X_test)}")
        if len(X_train) < 10:
            logger.error(f" Not enough training samples ({len(X_train)}). Aborting {ticker}.")
            raise ValueError(f"Not enough training samples for {ticker}.")
    except Exception as e:
        logger.error(f" Error preparing data for {ticker}: {e}")
        raise

    input_shape = (X_train.shape[1], X_train.shape[2])
    from pathlib import Path
    model_path = Path("models") / f"{ticker}_best_model.h5"
    if not os.path.exists(model_path):
        logger.info(" No model found. Training from scratch.")
        retrain = True
    else:
        retrain = False

    if retrain:
        try:
            logger.info(" Retraining model from scratch...")
            result = tune_and_train_model(X_train, y_train, input_shape, project_name=f"lstm_model_{ticker}")

            if result is None or len(result) != 3:
                raise ValueError("tune_and_train_model did not return 3 values (model, history, best_hp)")

            model, history, best_hp = result
            model.save(model_path)
            logger.info(f" Model saved to {model_path}")
        except Exception as e:
            logger.error(f" Error training model for {ticker}: {e}")
            raise
    else:
        if not os.path.exists(model_path):
            logger.error(f" Pre-trained model not found at {model_path}.")
            raise FileNotFoundError(f"No model found at {model_path}.")
        try:
            model = tf.keras.models.load_model(model_path)
            logger.info(f" Loaded pre-trained model for {ticker}.")
        except (OSError, IOError) as e:
            logger.error(f" Model loading error for {ticker}: {e}")
            raise

    current_price = float(data['Close'].iloc[-1].item())
    last_seq = X_test[-1:]
    pred_scaled = model.predict(last_seq, verbose=0)[0][0]
    cmin, cmax = scaler.data_min_[3], scaler.data_max_[3]
    lstm_pred = pred_scaled * (cmax - cmin) + cmin
    base_pred = current_price
    raw_pred = (0.9 * lstm_pred + 0.1 * base_pred) if use_ensemble else lstm_pred

    hist_vol = float(data['Close'].pct_change().dropna().std())
    clamp_pct = min(hist_vol * 2, 0.05)
    pmin, pmax = current_price * (1 - clamp_pct), current_price * (1 + clamp_pct)
    predicted_price = float(np.clip(raw_pred, pmin, pmax))

    ts_utc = data.index[-1]
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=ZoneInfo("UTC"))
    ts_local = ts_utc.astimezone(ZoneInfo("Europe/Vienna"))
    ts_str = ts_local.strftime("%Y-%m-%d %H:%M %Z")

    last_close = float(data['Close'].iloc[-2]) if len(data) >= 2 else current_price
    next_close_dt = datetime.combine(
        ts_local.date() + timedelta(days=1),
        datetime.strptime("17:30", "%H:%M").time()
    ).replace(tzinfo=ZoneInfo("Europe/Vienna"))
    next_close_str = next_close_dt.strftime("%Y-%m-%d %H:%M %Z")

    pct_change = (predicted_price / current_price - 1) * 100
    errors = get_recent_errors(30)
    sigma = float(np.std(errors)) if errors else hist_vol * 100
    if sigma > 0:
        z = abs(pct_change) / sigma
        confidence = 0.5 * (1 + math.erf(z / math.sqrt(2))) * 100
    else:
        confidence = 50.0
    confidence = float(round(confidence, 1))

    ind = add_technical_indicators(data.copy()).iloc[-1]
    sma20, sma50, rsi = float(ind['SMA_20']), float(ind['SMA_50']), float(ind['RSI_14'])
    trend = "Bullish" if sma20 > sma50 else "Bearish"
    volume = float(data['Volume'].iloc[-1])
    decision, _ = determine_trade_signal(pct_change, trend, rsi, volume)

    pos_size = determine_position_size(account_balance, risk_per_trade, hist_vol)

    fgi_val, fgi_status, fgi_ts = get_fear_and_greed_index()

    predicted = model.predict(X_test, verbose=0).flatten()
    predicted_prices = predicted * (scaler.data_max_[3] - scaler.data_min_[3]) + scaler.data_min_[3]
    test_start = len(data) - (len(X_test) + window_size)
    actual_prices = data['Close'].iloc[test_start + window_size:].values

    plot_predictions(predicted_prices, actual_prices)

    try:
        sensitivity = compute_feature_sensitivity(
            model,
            X_test[-1:],
            scaler,
            [
                'Open','High','Low','Close','Volume',
                'SMA_20','SMA_50','RSI_14','MACD','ATR','ROC',
                'BB_mid','BB_upper','BB_lower',
                'Stoch_%K','Stoch_%D',
                'OBV','VWAP','CMF','Volatility_Regime'
            ]
        )
    except Exception as e:
        logger.error(f" Sensitivity analysis failed: {e}")
        sensitivity = None

    logger.info(f" Directional Accuracy: {acc:.2%}, Cumulative Return: {cum_ret:.2%}")
    logger.info(f" Predicted next close price: {predicted_price:.2f} ({pct_change:.2f}%)")
    logger.info(f" Confidence: {confidence:.1f}%, Trade Signal: {decision}")

    return {
        "Last Closing Price": last_close,
        "Current Price": current_price,
        "Current Timestamp": ts_str,
        "Predicted Price for Close": predicted_price,
        "Predicted Close Time": next_close_str,
        "Trend Signal": trend,
        "Predicted % Change": pct_change,
        "Trade Decision": decision,
        "Signal Confidence": confidence,
        "Position Size": pos_size,
        "Volatility": hist_vol,
        "Directional Accuracy": acc,
        "Cumulative Return": cum_ret,
        "Fear and Greed Index": fgi_val,
        "Fear and Greed Status": fgi_status,
        "Fear and Greed Timestamp": fgi_ts,
        "Feature Sensitivity": sensitivity
    }, data

# ------------------------------
# Model Monitoring & Retraining (with Rolling Window)
# ------------------------------
def monitor_and_update_model(ticker, iterations=1, update_interval_seconds=10,
                             window_size=60, account_balance=100000, risk_per_trade=0.01):
    """
    Simulates a monitoring loop that periodically:
      - Downloads the latest daily data.
      - Retrains/updates the model using a rolling window (currently just re-trains from scratch).
      - Computes updated predictions and performance metrics.
      - Logs and prints updated predictions.

    For testing, the loop runs for a fixed number of iterations with a short sleep interval.
    """
    for i in range(iterations):
        try:
            logging.info(f"Update iteration {i + 1} for ticker {ticker}...")
            results, _ = train_predict_for_ticker(
                ticker,
                use_ensemble=True,
                account_balance=account_balance,
                risk_per_trade=risk_per_trade
            )
            logging.info(f"Updated results for {ticker}:")
            for key, value in results.items():
                logging.info(f"  {key}: {value}")

            print(f"\nUpdated Results for {ticker} (Iteration {i + 1}):")
            for key, value in results.items():
                print(f"  {key}: {value}")

        except Exception as e:
            logging.error(f"Error in update iteration {i + 1} for {ticker}: {e}")

        logging.info(f"Sleeping until next update for {ticker} (waiting {update_interval_seconds} seconds)...\n")
        time.sleep(update_interval_seconds)

# ------------------------------
# Richer Feature Set & Alternative Data
# ------------------------------
def add_richer_features(data: pd.DataFrame, n_regimes: int = 3) -> pd.DataFrame:
    """
    Enhances the DataFrame with additional technical indicators and volatility regimes.
    Fixes issues with zero/NaN volume and prevents failures from empty GMM inputs.
    """
    df = data.copy()

    # --------- Safe handling for Volume ----------
    if 'Volume' not in df.columns:
        df['Volume'] = 1e3  # fallback dummy if Volume missing

    # Replace zero/NaN with fallback (minimum positive or 1e3)
    df['Volume'] = df['Volume'].replace(0, np.nan)
    min_positive = df['Volume'][df['Volume'] > 0].min()
    fallback = min_positive if isinstance(min_positive, (int, float)) and min_positive > 0 else 1e3
    df['Volume'] = df['Volume'].fillna(fallback)

    # --------- Bollinger Bands ----------
    df['BB_mid'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']

    # --------- Stochastic Oscillator ----------
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_%K'] = (df['Close'] - low_14) / (high_14 - low_14 + 1e-10)
    df['Stoch_%D'] = df['Stoch_%K'].rolling(window=3).mean()

    # --------- OBV ----------
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # --------- VWAP ----------
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()

    # --------- CMF ----------
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
    mf_volume = mf_multiplier * df['Volume']
    df['CMF'] = mf_volume.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()

    # --------- Volatility Regime ----------
    returns = df['Close'].pct_change().fillna(0)
    rolling_vol = returns.rolling(window=10).std().fillna(0).values.reshape(-1, 1)

    if np.count_nonzero(rolling_vol > 0) > 50:
        gm = GaussianMixture(n_components=n_regimes, random_state=42)
        df['Volatility_Regime'] = gm.fit_predict(rolling_vol)
    else:
        df['Volatility_Regime'] = 0  # fallback: single regime

    # --------- Final Cleanup ----------
    before = len(df)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.dropna(inplace=True)
    after = len(df)
    dropped = before - after
    drop_pct = round((dropped / before * 100), 1) if before > 0 else 0.0
    logging.info(f"[Cleaner] Dropped {dropped} rows ({drop_pct}%) due to NaNs.")
    print("Final DataFrame shape:", df.shape)
    print("Remaining NaNs:", df.isna().sum().sum())
    return df


# Integration into existing prepare pipeline:

def prepare_data_and_split_with_richer(data: pd.DataFrame, window_size: int = 60, test_ratio: float = 0.2):
    """
    Adds richer features, splits, scales and sequences.
    Now auto-handles small datasets more gracefully.
    """
    data = add_technical_indicators(data)
    df_features = add_richer_features(data)

    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'ATR', 'ROC',
        'BB_mid', 'BB_upper', 'BB_lower',
        'Stoch_%K', 'Stoch_%D',
        'OBV', 'VWAP', 'CMF', 'Volatility_Regime'
    ]
    df_model = df_features[feature_cols].copy().ffill()

    if len(df_model) < window_size + 5:
        raise ValueError(f"Not enough data to create sequences. Only {len(df_model)} rows after processing.")

    split_idx = int((1 - test_ratio) * len(df_model))
    train_df = df_model.iloc[:split_idx]
    test_df = df_model.iloc[split_idx:]

    scaler = MinMaxScaler()
    scaler.fit(train_df)
    scaled_train = scaler.transform(train_df)
    scaled_test = scaler.transform(test_df)

    def build_seq(arr, original_df):
        X, y = [], []
        for i in range(window_size, len(arr)):
            X.append(arr[i-window_size:i, :])
            # 🟢 Use **actual price** from the unscaled DataFrame
            y.append(original_df.iloc[i]['Close'])
        return np.array(X), np.array(y)

    X_train, y_train = build_seq(scaled_train, train_df)
    X_test, y_test = build_seq(scaled_test, test_df)
    train_size = len(X_train)

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Train/Test split resulted in empty sets. Reduce window_size or increase dataset.")

    return X_train, y_train, X_test, y_test, scaler, train_size

# ------------------------------
# Main Trading Pipeline (with Monitoring and Retraining)
# ------------------------------
def main():
    indices = {
        "S&P 500": "^GSPC",
        "ATX": "^ATX",
        "DAX": "^GDAXI",
        "NASDAQ": "^IXIC"
    }
    # For demonstration, run the monitoring loop for one iteration per index
    for index_name, ticker in indices.items():
        logging.info(f"Starting monitoring and retraining loop for {index_name} ({ticker})...")
        monitor_and_update_model(ticker, iterations=1, update_interval_seconds=5)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(f"Fatal error in main execution: {e}")
