import subprocess
import logging
from pathlib import Path
import toml
from storage import init_db

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

from storage import add_to_watchlist, remove_from_watchlist, get_watchlist

# --- load config ---
_cfg = toml.load(Path(__file__).with_name("config.toml"))
TOKEN = _cfg["telegram"]["bot_token"]
AUTHORIZED_CHAT_ID = str(_cfg["telegram"]["chat_id"])

# --- paths ---
PYTHON = r"C:\Users\Paul\Coding\Trained model\.venv\Scripts\python.exe"
JOBS = r"C:\Users\Paul\Coding\Trained model\jobs.py"

logging.basicConfig(level=logging.INFO)

# --- restricted access ---
def restricted(func):
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if str(update.effective_chat.id) != AUTHORIZED_CHAT_ID:
            logging.warning("Unauthorized access denied for chat %s", update.effective_chat.id)
            return
        await func(update, context)
    return wrapped

# --- command handlers ---
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

@restricted
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📬 Sending help instructions…")
    subprocess.Popen([PYTHON, JOBS, "help"])

@restricted
async def add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❗ Usage: /add TICKER")
        return
    ticker = context.args[0].upper()
    add_to_watchlist(ticker)
    await update.message.reply_text(f"✅ Added `{ticker}` to your watchlist.", parse_mode="Markdown")

@restricted
async def remove(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❗ Usage: /remove TICKER")
        return
    ticker = context.args[0].upper()
    remove_from_watchlist(ticker)
    await update.message.reply_text(f"🗑️ Removed `{ticker}` from your watchlist.", parse_mode="Markdown")

@restricted
async def list_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tickers = get_watchlist()
    if not tickers:
        await update.message.reply_text("📭 Your watchlist is currently empty.")
    else:
        formatted = "\n".join(f"• `{t}`" for t in tickers)
        await update.message.reply_text(f"📋 *Your Watchlist:*\n{formatted}", parse_mode="Markdown")

# --- app setup ---
def main():
    init_db()
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("forecast", forecast))
    app.add_handler(CommandHandler("evaluate", evaluate))
    app.add_handler(CommandHandler("retrain", retrain))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("add", add))
    app.add_handler(CommandHandler("remove", remove))
    app.add_handler(CommandHandler("watchlist", list_watchlist))
    app.run_polling()

if __name__ == "__main__":
    main()
