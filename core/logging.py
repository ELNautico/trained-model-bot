import logging
import os
import time
import numpy as np
try:
    import psutil
except ImportError:
    psutil = None
import tensorflow as tf
from logging.handlers import TimedRotatingFileHandler
from tensorflow.keras.callbacks import Callback
from alert import send


def configure_logging(log_file="logs/trading_log.txt"):
    """
    Configure root logger with a timed rotating file handler and console output.
    Rotates logs at midnight and keeps 7 days of backups.
    """
    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Timed rotating file handler: rotate at midnight, keep 7 backups
    handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        backupCount=7,
        encoding="utf-8"
    )
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    handler.setFormatter(formatter)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    # Apply handlers
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler, console]
    )


class ProgressCallback(Callback):
    """
    Keras callback that logs epoch progress, estimates ETA, and optionally sends notifications.
    Also captures CPU, memory, and (if available) GPU usage.
    """
    def __init__(self, label=None, notify_every_n_epochs=1):
        super().__init__()
        self.label = label
        self.notify_every_n_epochs = notify_every_n_epochs

    def on_train_begin(self, logs=None):
        self.epoch_times = []
        self.total_epochs = self.params.get("epochs", 0)
        logging.info(f"🏁 Starting training for {self.total_epochs} epochs...")
        if self.label:
            send(f"🏁 Started training for {self.label} ({self.total_epochs} epochs)")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.epoch_start
        self.epoch_times.append(duration)
        avg = np.mean(self.epoch_times)
        remaining = avg * (self.total_epochs - epoch - 1)

        # Safely retrieve losses
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
        val_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"

        # Resource usage
        cpu_info = None
        mem_info = None
        if psutil:
            try:
                cpu_info = psutil.cpu_percent()
                mem_info = psutil.virtual_memory().percent
            except Exception:
                cpu_info = None
                mem_info = None

        # GPU usage
        gpu_info = ""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # Requires TensorFlow 2.3+
                mem_stats = tf.config.experimental.get_memory_info(gpus[0].name)
                used_mb = mem_stats['peak'] / (1024 ** 2)
                gpu_info = f" | GPU peak_mem: {used_mb:.1f}MB"
        except Exception:
            gpu_info = ""

        # Build log message
        msg = (
            f"Epoch {epoch+1}/{self.total_epochs} - "
            f"Time: {duration:.2f}s - ETA: {remaining:.2f}s | "
            f"Loss: {loss_str} | Val Loss: {val_str}"
        )
        if cpu_info is not None and mem_info is not None:
            msg += f" | CPU: {cpu_info:.1f}% | Mem: {mem_info:.1f}%"
        msg += gpu_info

        logging.info(msg)

        # Optional notification
        if self.label and (epoch + 1) % self.notify_every_n_epochs == 0:
            notify_msg = (
                f"🧪 {self.label} Epoch {epoch+1}/{self.total_epochs}\n"
                f"Loss: {loss_str}, Val Loss: {val_str}, ETA: {remaining:.1f}s"
            )
            if cpu_info is not None and mem_info is not None:
                notify_msg += f"\nCPU: {cpu_info:.1f}%, Mem: {mem_info:.1f}%{gpu_info}"
            send(notify_msg)
