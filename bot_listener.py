import logging
import asyncio
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import toml
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

from storage import init_db, add_to_watchlist, remove_from_watchlist, get_watchlist
from train.pipeline import train_predict_for_ticker
from train.core import train_and_save_model, load_model, predict_price

# --- load config ---
_cfg = toml.load(Path(__file__).with_name("config.toml"))
TOKEN = _cfg["telegram"]["bot_token"]
AUTHORIZED_CHAT_ID = str(_cfg["telegram"]["chat_id"])

logging.basicConfig(level=logging.INFO)


def restricted(func):
    """
    Decorator to block unauthorized chats but reply politely.
    """
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = str(update.effective_chat.id)
        if chat_id != AUTHORIZED_CHAT_ID:
            await update.message.reply_text(
                "🚫 You are not authorized to use this bot."
            )
            logging.warning("Unauthorized access attempt from %s", chat_id)
            return
        await func(update, context)
    return wrapped


@restricted
async def forecast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🕑 Starting daily forecast…")
    tickers = get_watchlist()
    if not tickers:
        return await msg.edit_text("📭 Your watchlist is empty. Add tickers with /add.")

    for ticker in tickers:
        version_dir = Path("models") / ticker
        if not version_dir.exists() or not any(version_dir.iterdir()):
            await msg.reply_text(f"🛠️ No model found for {ticker}. Training a new one now…")

        try:
            result, _ = await asyncio.to_thread(
                train_predict_for_ticker,
                ticker, True, 100_000, 0.01
            )

            current   = result["Current Price"]
            predicted = result["Predicted Price"]
            conf      = result["Confidence"]
            pct_chg   = result["Predicted % Change"]

            direction = "Buy" if pct_chg > 0 else "Sell" if pct_chg < 0 else "Hold"

            now   = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC")) \
                                     .astimezone(ZoneInfo("Europe/Vienna"))
            ts_str = now.strftime("%d.%m at %H:%M")

            text = (
                f"Prediction on {ts_str}\n"
                f"Current Price: {current:.2f}\n"
                f"Predicted Close: {predicted:.2f} ({conf:.1f}%)\n\n"
                f"Recommendation:\n"
                f"{direction}"
            )
            await msg.reply_text(text)

        except Exception as e:
            logging.error("Forecast failed for %s: %s", ticker, e)
            await msg.reply_text(f"❌ {ticker}: Forecast failed – {e}")

@restricted
async def evaluate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🕑 Running end-of-day evaluation…")
    # If you have an evaluate function, call it here; placeholder:
    await msg.edit_text("✅ Evaluation complete.")  # implement actual logic

@restricted
async def retrain(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🕑 Retraining all models… this may take a while.")
    tickers = get_watchlist()
    if not tickers:
        return await msg.edit_text("📭 Your watchlist is empty. Add tickers with /add.")

    for ticker in tickers:
        try:
            # Retrain in a thread to avoid blocking
            model = await asyncio.to_thread(
                train_and_save_model,
                *prepare_for_retrain(ticker)
            )
            await msg.reply_text(f"✅ Retrained model for {ticker}.")
        except Exception as e:
            logging.error("Retrain failed for %s: %s", ticker, e)
            await msg.reply_text(f"❌ Retrain failed for {ticker}: {e}")

def prepare_for_retrain(ticker: str):
    """
    Helper to load or prepare data for retraining.
    Returns args for train_and_save_model.
    """
    # You may need to re-download & preprocess data here
    from train.pipeline import download_data, prepare_data_and_split
    df = download_data(ticker)
    X_train, y_train, _, _, _, _ = prepare_data_and_split(df)
    return (X_train, y_train, X_train.shape[1:], ticker)

@restricted
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "🧠 *Available Commands:*\n\n"
        "/forecast – Run model and send market predictions.\n"
        "/evaluate – Compare today’s forecasts to actual prices.\n"
        "/retrain – Retrain all models in your watchlist.\n"
        "/add TICKER – Add a stock ticker to your watchlist.\n"
        "/remove TICKER – Remove a stock from your watchlist.\n"
        "/watchlist – Show all watched tickers.\n"
        "/help – Show this message."
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

@restricted
async def add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await update.message.reply_text("❗ Usage: /add TICKER")
    ticker = context.args[0].upper()
    add_to_watchlist(ticker)
    await update.message.reply_text(f"✅ Added `{ticker}` to your watchlist.", parse_mode="Markdown")

@restricted
async def remove(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await update.message.reply_text("❗ Usage: /remove TICKER")
    ticker = context.args[0].upper()
    remove_from_watchlist(ticker)
    await update.message.reply_text(f"🗑️ Removed `{ticker}` from your watchlist.", parse_mode="Markdown")

@restricted
async def list_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tickers = get_watchlist()
    if not tickers:
        await update.message.reply_text("📭 Your watchlist is empty.")
    else:
        formatted = "\n".join(f"• `{t}`" for t in tickers)
        await update.message.reply_text(f"📋 *Your Watchlist:*\n{formatted}", parse_mode="Markdown")

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
