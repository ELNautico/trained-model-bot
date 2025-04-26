import subprocess, logging
from pathlib import Path
import toml

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

# --- load config ---
_cfg = toml.load(Path(__file__).with_name("config.toml"))
TOKEN = _cfg["telegram"]["bot_token"]
AUTHORIZED_CHAT_ID = str(_cfg["telegram"]["chat_id"])

# --- paths ---
PYTHON = r"C:\Users\Paul\Coding\Trained model\.venv\Scripts\python.exe"
JOBS   = r"C:\Users\Paul\Coding\Trained model\jobs.py"

logging.basicConfig(level=logging.INFO)

def restricted(func):
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if str(update.effective_chat.id) != AUTHORIZED_CHAT_ID:
            logging.warning("Unauthorized access denied for chat %s", update.effective_chat.id)
            return
        await func(update, context)
    return wrapped

@restricted
async def forecast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🕑 Starting daily forecast…")
    subprocess.Popen([PYTHON, JOBS, "forecast"])

@restricted
async def evaluate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🕑 Running end-of-day evaluation…")
    subprocess.Popen([PYTHON, JOBS, "evaluate"])

@restricted
async def retrain(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🕑 Retraining all models… this may take a while.")
    subprocess.Popen([PYTHON, JOBS, "retrain"])

def main():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("forecast", forecast))
    app.add_handler(CommandHandler("evaluate", evaluate))
    app.add_handler(CommandHandler("retrain", retrain))
    app.run_polling()

if __name__ == "__main__":
    main()
