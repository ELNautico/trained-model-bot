import os
import time
import logging
import yfinance as yf
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo  # For timezone conversion
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

import tensorflow as tf
import tensorflow.keras.models as keras_models
import tensorflow.keras.layers as keras_layers
import tensorflow.keras.callbacks as keras_callbacks
import keras_tuner as kt

from storage import init_db, get_recent_errors

# ------------------------------
# Attempt to import FearAndGreedIndex; if not available, define a fallback.
# ------------------------------
try:
    from fear_and_greed import FearAndGreedIndex
except ImportError as e:
    logging.error(f"Error importing FearAndGreedIndex: {e}")
    class FearAndGreedIndex:
        @property
        def current_value(self):
            return "N/A"
        @property
        def current_status(self):
            return "N/A"
        @property
        def timestamp(self):
            return "N/A"


# ------------------------------
# Logging Configuration
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[logging.FileHandler("trading_log.txt"), logging.StreamHandler()]
)

# ------------------------------
# Adaptive Retraining Logic
# ------------------------------
def should_retrain(max_error_threshold: float = 2.0, window: int = 7) -> bool:
    """
    Pull the last `window` pct_error values from the evaluation table;
    retrain if the average absolute error exceeds `max_error_threshold`%.
    """
    init_db()
    errors = get_recent_errors(window)  # List[float]
    if not errors:
        # no history → train once
        return True
    avg_err = float(np.mean(np.abs(errors)))
    logging.info(f"Avg |%_error| over last {window} days = {avg_err:.2f}%")
    return avg_err > max_error_threshold

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

    def build_model(hp):
        model = keras_models.Sequential([
            keras_layers.Input(shape=input_shape),
            keras_layers.LSTM(hp.Int('units', 32, 128, step=32), return_sequences=True),
            keras_layers.Dropout(hp.Float('drop', 0.0, 0.5, step=0.1)),
            keras_layers.LSTM(hp.Int('units2', 32, 128, step=32)) if hp.Boolean('second_layer') else keras_layers.LSTM(hp.Int('units', 32, 128, step=32)),
            keras_layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    tuner = kt.Hyperband(
        build_model,
        objective=lambda m, _x, _y: _cv_loss(m, X, y, tscv),
        max_epochs=30,
        factor=3,
        directory='tuner',
        project_name=project_name
    )
    tuner.search(x=None, y=None)
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = build_model(best_hp)
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
        logging.info(
            f"Epoch {epoch + 1:>2}/{self.total_epochs}: "
            f"loss = {loss:.4f}, val_loss = {val_loss:.4f}, "
            f"epoch time = {elapsed:.2f}s, ETA = {eta:.2f}s"
        )

# ------------------------------
# Technical Indicators
# ------------------------------
def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['SMA_50'] = data['Close'].rolling(50).mean()
    delta = data['Close'].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    data['RSI_14'] = 100 - 100/(1 + gain.rolling(14).mean()/(loss.rolling(14).mean()+1e-10))
    # MACD, ATR, ROC
    data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
    data['ATR'] = data['High'].rolling(14).max() - data['Low'].rolling(14).min()
    data['ROC'] = data['Close'].pct_change(10)
    data.dropna(inplace=True)
    return data

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

# ------------------------------
# Prepare and Split Data (avoiding data leakage)
# ------------------------------
def prepare_data_and_split(data, window_size=60, test_ratio=0.2):
    """
    1) Adds technical indicators.
    2) Splits into train/test subsets (80/20 by default).
    3) Fits MinMaxScaler on the training subset only, then transforms both.
    4) Creates windowed sequences for LSTM training/testing.

    Returns:
        X_train, y_train, X_test, y_test, scaler, train_size
    """
    # Add technical indicators
    data = add_technical_indicators(data)

    # Define our features (columns)
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI_14']
    df = data[feature_cols].copy()
    df.ffill(inplace=True)  # forward fill to handle any missing

    # Basic check for enough data
    if len(df) < window_size + 1:
        raise ValueError("Not enough data to form a single sequence.")

    # Split index
    split_index = int((1 - test_ratio) * len(df))
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    # Fit scaler on training portion only
    scaler = MinMaxScaler()
    scaler.fit(train_df)

    # Transform train and test
    scaled_train = scaler.transform(train_df)
    scaled_test = scaler.transform(test_df)

    # Utility to build sequences from a scaled array
    def build_sequences(scaled_array):
        X_list, y_list = [], []
        for i in range(window_size, len(scaled_array)):
            # features from [i-window_size : i], predict close at index i
            X_list.append(scaled_array[i - window_size:i, :])
            y_list.append(scaled_array[i, 3])  # index 3 => 'Close'
        return np.array(X_list), np.array(y_list)

    X_train, y_train = build_sequences(scaled_train)
    X_test, y_test = build_sequences(scaled_test)

    train_size = len(X_train)  # for reference in backtesting
    return X_train, y_train, X_test, y_test, scaler, train_size

# ------------------------------------------------------------------
# Prepare data when you want to use *all* rows for training
# ------------------------------------------------------------------
def prepare_data_full(data: pd.DataFrame, window_size: int = 60):
    """
    Adds indicators, scales on the full dataframe, and returns sequences X, y, scaler.
    No train/test split – used for final live retraining.
    """
    data = add_technical_indicators(data)
    feature_cols = ['Open', 'High', 'Low', 'Close',
                    'Volume', 'SMA_20', 'SMA_50', 'RSI_14']
    df = data[feature_cols].copy().ffill()

    if len(df) < window_size + 1:
        raise ValueError("Not enough data to form a single sequence.")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i - window_size:i, :])
        y.append(scaled[i, 3])          # Close
    return np.asarray(X), np.asarray(y), scaler

# ------------------------------
# Baseline Forecast Model
# ------------------------------
def baseline_forecast(data):
    """Generates a baseline forecast using the 20-day SMA."""
    return float(data['Close'].iloc[-1])

# ------------------------------
# Model Building and Hyperparameter Tuning
# ------------------------------
def build_model(hp, input_shape):
    model = keras_models.Sequential()
    model.add(keras_layers.Input(shape=input_shape))
    units = hp.Int('units', min_value=32, max_value=128, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)

    # First LSTM layer
    model.add(keras_layers.LSTM(units, return_sequences=True))
    model.add(keras_layers.Dropout(dropout_rate))

    # Optional second LSTM layer
    if hp.Boolean('second_layer'):
        units2 = hp.Int('units2', min_value=32, max_value=128, step=32)
        model.add(keras_layers.LSTM(units2))
        model.add(keras_layers.Dropout(dropout_rate))
    else:
        model.add(keras_layers.LSTM(units))
        model.add(keras_layers.Dropout(dropout_rate))

    # Final Dense layer for regression
    model.add(keras_layers.Dense(1))

    # Hyperparameter for learning rate
    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')
    # Compile with additional metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mape']
    )
    return model

def tune_and_train_model(X_train, y_train, input_shape, project_name="lstm_model"):
    tuner = kt.Hyperband(
        lambda hp: build_model(hp, input_shape),
        objective='val_loss',
        max_epochs=30,
        factor=3,
        directory='lstm_tuner',
        project_name=project_name
    )
    early_stop = keras_callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    progress_cb = ProgressCallback()

    logging.info("Starting hyperparameter tuning...")
    tuner.search(
        X_train, y_train,
        epochs=30,
        validation_split=0.2,
        callbacks=[early_stop, progress_cb],
        verbose=1
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    logging.info(f"Best hyperparameters found: {best_hp.values}")

    # Build final model with best hyperparameters
    model = build_model(best_hp, input_shape)
    logging.info("Training final model with best hyperparameters...")

    history = model.fit(
        X_train, y_train,
        epochs=30,
        validation_split=0.2,
        callbacks=[early_stop, progress_cb],
        verbose=1
    )
    return model, history, best_hp

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
    Calculates directional accuracy and a simplified cumulative return metric
    using the predicted sign vs. actual sign for each day in the test set.
    """
    # Predict on test set
    predictions = model.predict(X_test).flatten()

    # Inverse transform for 'Close' only
    close_min = scaler.data_min_[3]
    close_max = scaler.data_max_[3]
    predicted_prices = predictions * (close_max - close_min) + close_min

    # The test set data in the original "data" goes from (split_index + window_size) to the end
    # => We can re-construct the actual indices used for the test portion
    test_end = len(data)
    # The "train_size" is the number of sequences in the training portion.
    # Each sequence is 'window_size' in length, so the raw training data covers up to `split_index`.
    # If you want a more precise link to the original data indices:
    split_index = len(data) - (len(X_test) + window_size)
    test_indices = np.arange(split_index + window_size, test_end)

    test_actual = data['Close'].iloc[test_indices].values

    # Day-by-day returns
    predicted_returns = (predicted_prices / test_actual) - 1.0
    previous_prices = data['Close'].iloc[test_indices - 1].values
    actual_returns = (test_actual / previous_prices) - 1.0

    directional_accuracy = np.mean(np.sign(predicted_returns) == np.sign(actual_returns))

    # Simple PnL: go long if predicted return > 0, else short
    strategy_returns = np.where(predicted_returns > 0, actual_returns, -actual_returns)
    cumulative_return = np.prod(1 + strategy_returns) - 1

    return directional_accuracy, cumulative_return

# ------------------------------
# Fetch Fear and Greed Index
# ------------------------------
def get_fear_and_greed_index():
    try:
        fgi = FearAndGreedIndex()
        current_value = fgi.current_value
        current_status = fgi.current_status
        timestamp = fgi.timestamp
        return current_value, current_status, timestamp
    except Exception as e:
        logging.error(f"Error fetching Fear and Greed index: {e}")
        return None, None, None

# ------------------------------
# Main Trading Function: train_predict_for_ticker
# ------------------------------
def train_predict_for_ticker(ticker, use_ensemble=True, account_balance=100000, risk_per_trade=0.01):
    """
    Downloads daily data (30 years), prepares data with a 60-day window, trains (or loads) an LSTM,
    and produces outputs including trade signals, position sizing, backtesting metrics,
    and feature sensitivity (with a real confidence measure).
    """
    # 1) Download data
    try:
        data = download_daily_data(ticker)
    except Exception as e:
        logging.error(f"Error downloading data for {ticker}: {e}")
        raise

    # 2) Prepare data
    window_size = 60
    try:
        X_train, y_train, X_test, y_test, scaler, train_size = prepare_data_and_split(
            data, window_size=window_size, test_ratio=0.2
        )
        if len(X_train) < 10:
            raise ValueError("Not enough training samples to train the model.")
    except Exception as e:
        logging.error(f"Error preparing data for {ticker}: {e}")
        raise

    input_shape = (X_train.shape[1], X_train.shape[2])

    # 3) Train or load model
    model_path = f"{ticker}_best_model.h5"
    if should_retrain():
        try:
            model, history, best_hp = tune_and_train_model(
                X_train, y_train, input_shape, project_name=f"lstm_model_{ticker}"
            )
            model.save(model_path)
        except Exception as e:
            logging.error(f"Error training model for {ticker}: {e}")
            raise
    else:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot find model at {model_path}")
        model = tf.keras.models.load_model(model_path)

    # 4) Raw forecast
    current_price = float(data['Close'].iloc[-1])
    last_seq      = X_test[-1:].copy()
    pred_scaled   = model.predict(last_seq, verbose=0)[0][0]

    #  └─ inverse scale
    cmin, cmax   = scaler.data_min_[3], scaler.data_max_[3]
    lstm_pred    = pred_scaled * (cmax - cmin) + cmin
    base_pred    = current_price
    raw_pred     = (0.9*lstm_pred + 0.1*base_pred) if use_ensemble else lstm_pred

    #  └─ clamp by ±2σ, max ±5%
    hist_vol     = float(data['Close'].pct_change().dropna().std())
    clamp_pct    = min(hist_vol*2, 0.05)
    pmin, pmax   = current_price*(1-clamp_pct), current_price*(1+clamp_pct)
    predicted_price = float(np.clip(raw_pred, pmin, pmax))

    # 5) Timestamps
    ts_utc   = data.index[-1]
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=ZoneInfo("UTC"))
    ts_local = ts_utc.astimezone(ZoneInfo("Europe/Vienna"))
    ts_str   = ts_local.strftime("%Y-%m-%d %H:%M %Z")

    # 6) Yesterday close & next-close timestamp
    last_close = float(data['Close'].iloc[-2]) if len(data)>=2 else current_price
    next_close_dt = datetime.combine(
        ts_local.date()+timedelta(days=1),
        datetime.strptime("17:30","%H:%M").time()
    ).replace(tzinfo=ZoneInfo("Europe/Vienna"))
    next_close_str = next_close_dt.strftime("%Y-%m-%d %H:%M %Z")

    # 7) Real confidence: P(sign correct) under N(0,σ²) error model
    pct_change = (predicted_price/current_price - 1)*100
    errors = get_recent_errors(30)          # last 30 days pct_error
    sigma = float(np.std(errors)) if errors else hist_vol*100
    if sigma > 0:
        z = abs(pct_change)/sigma
        # P(actual and predicted share the same sign)
        confidence = 0.5*(1 + math.erf(z/math.sqrt(2))) * 100
    else:
        confidence = 50.0
    confidence = float(round(confidence,1))

    # 8) Indicators & signal
    ind = add_technical_indicators(data.copy()).iloc[-1]
    sma20, sma50, rsi = float(ind['SMA_20']), float(ind['SMA_50']), float(ind['RSI_14'])
    trend = "Bullish" if sma20> sma50 else "Bearish"
    volume = float(data['Volume'].iloc[-1])
    cert, decision = generate_certificate(current_price, predicted_price, hist_vol), None
    decision, _ = determine_trade_signal(pct_change, trend, rsi, volume)  # only for Buy/Sell/No Trade

    # 9) Position sizing
    pos_size = determine_position_size(account_balance, risk_per_trade, hist_vol)

    # 10) Fear & Greed
    fgi_val, fgi_status, fgi_ts = get_fear_and_greed_index()

    # 11) Backtest
    acc, cum_ret = backtest_strategy(model, X_test, scaler, data, window_size, train_size)

    # 12) Feature sensitivity
    try:
        sensitivity = compute_feature_sensitivity(
            model, X_test[-1:], scaler,
            ['Open','High','Low','Close','Volume','SMA_20','SMA_50','RSI_14']
        )
    except Exception as e:
        logging.error(f"Sensitivity error: {e}")
        sensitivity = None

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
