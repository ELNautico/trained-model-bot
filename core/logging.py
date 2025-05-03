import logging
import os
import time
import numpy as np
from tensorflow.keras.callbacks import Callback
from alert import send  # ensure alert.send(message: str) is defined

def configure_logging(log_file="logs/trading_log.txt"):
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

class ProgressCallback(Callback):
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

        msg = f"Epoch {epoch+1}/{self.total_epochs} - Time: {duration:.2f}s - ETA: {remaining:.2f}s"
        if logs:
            loss = logs.get("loss", 0.0)
            val = logs.get("val_loss", 0.0)
            msg += f" | Loss: {loss:.4f} | Val Loss: {val:.4f}"
        logging.info(msg)

        if self.label and (epoch + 1) % self.notify_every_n_epochs == 0:
            send(f"🧪 {self.label} Epoch {epoch+1}/{self.total_epochs}\n"
                 f"Loss: {loss:.4f}, Val Loss: {val:.4f}, ETA: {remaining:.1f}s")
