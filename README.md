# Trained Model – Research Backtests & Signal Bot

This repository contains a **research-grade backtest harness** for a calibrated, triple-barrier-based signal model, plus supporting training and (optional) Telegram alerting utilities.

## Important disclaimer
This project is for **research and education**. It is **not** financial advice, and it should not be used to make real-money trading decisions without substantial additional work (robust validation, risk controls, monitoring, and compliance considerations).

## What’s inside

### 1) `backtest_signals.py` (walk-forward backtest)
Runs a long-only FLAT/LONG state machine with:

- Time-series-safe **walk-forward retraining** (retrain every *N* bars).
- A configurable **training lookback** window (e.g., 1200–2000 bars).
- Entry decisions at **Close[t]**, consistent with the labeling assumption.
- Exits via:
  - **Triple barrier** (stop/target) using next-day OHLC (conservative same-bar rule: stop wins).
  - **Time stop** at horizon.
  - Optional **model-based exits** (EV / p(stop) gates) at close.

Outputs:
- Printed summaries for **FULL** and optional **HOLDOUT** windows.
- CSVs written to `backtests/`:
  - `<TICKER>_trades[_<TAG>].csv`
  - `<TICKER>_equity[_<TAG>].csv`

### 2) `sweep_backtests.py` (parameter sweep / grid search)
Automates running many configurations across:
- `entry_min_ev`
- `retrain_every`
- `lookback`

For each run it computes BOTH:
- **FULL** metrics (entire span)
- **HOLDOUT** metrics (evaluation window you define)

It writes a consolidated CSV you can sort/filter in Excel or pandas, and can optionally re-run the **top-K** configs to save their detailed trades/equity CSVs.

### 3) Supporting modules
- `signals/` – model training, labeling, configuration (`SignalConfig`), feature column inference.
- `core/` – feature engineering and helper logic.
- `train/` – forecasting pipeline and model utilities (separate from the signal backtest harness).
- `bot_listener.py` – optional Telegram alerting (requires Telegram credentials).

## Setup

### Prerequisites
- Python 3.10+ recommended
- A Twelve Data API key (required for data download in `train/pipeline.py`)

### Install
```bash
python -m venv .venv
.
.venv\Scripts\activate        # Windows PowerShell
# source .venv/bin/activate    # macOS/Linux

pip install -U pip
pip install -r requirements.txt
```

### Configure credentials
Create `config.toml` from the template:
```bash
copy config.example.toml config.toml   # Windows
# cp config.example.toml config.toml   # macOS/Linux
```

Do not commit `config.toml`. It is ignored via `.gitignore`.

## Usage

### Single backtest
```bash
python backtest_signals.py TSLA \
  --start 2018-01-01 --end 2025-12-31 \
  --entry-min-ev 0.20 --retrain-every 80 --lookback 1500 \
  --metrics-start 2025-01-01 --metrics-end 2025-12-31 \
  --tag FINAL_2025_ev020_r80_lb1500
```

### Parameter sweep (grid search)
```bash
python sweep_backtests.py TSLA \
  --start 2018-01-01 --end 2025-12-31 \
  --metrics-start 2024-01-01 --metrics-end 2025-12-31 \
  --entry-min-ev 0.16:0.26:0.02 \
  --retrain-every 60,80,100 \
  --lookback 1200,1500,2000 \
  --objective holdout_sharpe \
  --min-holdout-trades 20 \
  --save-top-k 5 \
  --tag TSLA_grid_2024_2025
```

## Interpreting key knobs

- **`entry_min_ev`**: higher reduces trade count and (often) increases average trade quality, but can overfit if the holdout sample becomes too small.
- **`retrain_every`**: smaller retrains more frequently (more compute, potentially more responsive); larger retrains less often (more stable, potentially slower to adapt).
- **`lookback`**: larger provides more history (stability), smaller focuses on recent regime (adaptability). There is usually a bias/variance trade.

## Common pitfalls (and why we care)

- **Overfitting via parameter sweeps**: once you pick the “best” config on a given holdout, it is no longer a pristine out-of-sample test. Treat it as *model selection*, then validate again on a *fresh* period or other tickers.
- **Small-sample illusions**: very high Sharpe/profit factor with very few trades is usually not robust. Use `--min-holdout-trades` in sweeps.
- **Data adjustment**: some Twelve Data client versions do not support adjusted OHLCV; the code falls back to unadjusted with a warning. That can materially impact split-heavy tickers.

## Repo hygiene

- Generated artifacts (backtest CSVs, caches, local DB) are intentionally excluded via `.gitignore`.
- Keep `backtests/` empty in source control; it is an output folder.

## Suggested next steps

1. **Robust validation**
   - Evaluate on multiple tickers and multiple non-overlapping holdouts.
   - Add a final “lockbox” period (never used for tuning).

2. **Execution realism**
   - Add slippage models, spread, and (optionally) next-open entry/exit variants.
   - Stress test with higher costs.

3. **Stability analysis**
   - Sensitivity of results to small changes in `entry_min_ev`.
   - Rolling-window performance and drawdown analysis.

4. **Operationalization**
   - Paper trade only at first.
   - Add monitoring/alerts for data feed failures and model drift.
