import os
import json
import numpy as np
import logging
from pathlib import Path
from datetime import datetime


def save_json(data: dict, path: str):
    """Save a dictionary to a JSON file."""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logging.info(f"✅ Saved JSON to {path}")
    except Exception as e:
        logging.error(f"❌ Failed to save JSON to {path}: {e}")


def load_json(path: str) -> dict:
    """Load a JSON file if it exists."""
    if not os.path.exists(path):
        logging.warning(f"⚠️ JSON file not found: {path}")
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"❌ Failed to load JSON from {path}: {e}")
        return {}


def ensure_dir(path: str):
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def timestamp_now(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Returns current timestamp as formatted string."""
    return datetime.now().strftime(fmt)


def rmse(y_true, y_pred):
    """Root Mean Square Error."""
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def directional_accuracy(y_true, y_pred):
    """Percentage of correct direction forecasts."""
    return np.mean(np.sign(y_true) == np.sign(y_pred))
