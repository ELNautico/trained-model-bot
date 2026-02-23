import logging
import asyncio
from pathlib import Path

import toml
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

from core.storage import (
    init_db,
    add_to_watchlist,
    remove_from_watchlist,
    get_watchlist,
    list_positions,
    list_paper_positions,
)
from .jobs import signal_job, evaluate_job, positions_job, paper_positions_job, help_job

# --- load config ---
_cfg_path = Path(__file__).with_name("config.toml")
if not _cfg_path.exists():
    _cfg_path = Path(__file__).parent.parent / "config.toml"
_cfg = toml.load(_cfg_path)
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
            await update.message.reply_text("🚫 You are not authorized to use this bot.")
            logging.warning("Unauthorized access attempt from %s", chat_id)
            return
        await func(update, context)
    return wrapped


@restricted
async def forecast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Runs the NEW signal engine (BUY/WAIT/HOLD/SELL).
    Kept as /forecast for backward compatibility with your bot UX.
    """
    msg = await update.message.reply_text("🕑 Running signal engine for your watchlist…")

    try:
        await asyncio.to_thread(signal_job, False)
        await msg.edit_text("✅ Signal run complete. Check messages above for details.")
    except Exception as e:
        logging.error("Signal job failed: %s", e)
        await msg.edit_text(f"❌ Signal job failed – {e}")


@restricted
async def paper(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Run a full paper-trading signal cycle (no real positions opened)."""
    msg = await update.message.reply_text("🧪 Running paper-trading signal engine…")
    try:
        await asyncio.to_thread(signal_job, False, True)
        await msg.edit_text("✅ Paper signal run complete. Check messages above.")
    except Exception as e:
        logging.error("Paper signal job failed: %s", e)
        await msg.edit_text(f"❌ Paper signal job failed – {e}")


@restricted
async def evaluate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🕑 Running end-of-day exit checks…")
    try:
        await asyncio.to_thread(evaluate_job)
        await msg.edit_text("✅ EOD checks complete.")
    except Exception as e:
        logging.error("Evaluation failed: %s", e)
        await msg.edit_text(f"❌ Evaluation failed – {e}")


@restricted
async def evaluate_paper(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Run EOD exit checks for paper positions."""
    msg = await update.message.reply_text("🕑 Running paper EOD exit checks…")
    try:
        await asyncio.to_thread(evaluate_job, True)
        await msg.edit_text("✅ Paper EOD checks complete.")
    except Exception as e:
        logging.error("Paper evaluation failed: %s", e)
        await msg.edit_text(f"❌ Paper evaluation failed – {e}")


@restricted
async def positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    init_db()
    pos = list_positions()
    if not pos:
        return await update.message.reply_text("Position report: FLAT (no open positions).")

    lines = ["Open positions:"]
    for p in pos:
        lines.append(
            f"- {p['ticker']}: {p['state']} | shares={p['shares']} | "
            f"entry={float(p['entry_px']):.2f} | stop={float(p['stop_px']):.2f} | "
            f"target={float(p['target_px']):.2f} | hold={int(p['hold_days'])}/{int(p['horizon_days'])}"
        )
    await update.message.reply_text("\n".join(lines))


@restricted
async def paper_positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show open paper (simulated) positions."""
    init_db()
    pos = list_paper_positions()
    if not pos:
        return await update.message.reply_text("[PAPER] No open paper positions.")

    lines = ["[PAPER] Open positions:"]
    for p in pos:
        lines.append(
            f"- {p['ticker']}: {p['state']} | shares={p['shares']} | "
            f"entry={float(p['entry_px']):.2f} | stop={float(p['stop_px']):.2f} | "
            f"target={float(p['target_px']):.2f} | hold={int(p['hold_days'])}/{int(p['horizon_days'])}"
        )
    await update.message.reply_text("\n".join(lines))


@restricted
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "Available Commands:\n\n"
        "/forecast        – Run signals (BUY/WAIT/HOLD/SELL) for all watchlist tickers.\n"
        "/paper           – Paper-trading run (same signals, no real positions).\n"
        "/evaluate        – Run EOD exit checks (stop/target/time-stop).\n"
        "/evaluate_paper  – EOD checks for paper positions.\n"
        "/positions       – Show open real positions.\n"
        "/paper_positions – Show open paper positions.\n"
        "/add TICKER      – Add a stock ticker to your watchlist.\n"
        "/remove TICKER   – Remove a stock from your watchlist.\n"
        "/watchlist       – Show your current watchlist.\n"
        "/help            – Show this message.\n\n"
        "Note: This bot provides model-driven signals, not investment advice."
    )
    await update.message.reply_text(help_text)


@restricted
async def add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await update.message.reply_text("Usage: /add TICKER")
    ticker = context.args[0].upper()
    add_to_watchlist(ticker)
    await update.message.reply_text(f"✅ Added {ticker} to your watchlist.")


@restricted
async def remove(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await update.message.reply_text("Usage: /remove TICKER")
    ticker = context.args[0].upper()
    remove_from_watchlist(ticker)
    await update.message.reply_text(f"🗑️ Removed {ticker} from your watchlist.")


@restricted
async def list_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tickers = get_watchlist()
    if not tickers:
        await update.message.reply_text("📭 Your watchlist is empty.")
    else:
        formatted = "\n".join(f"• {t}" for t in tickers)
        await update.message.reply_text(f"📋 Your Watchlist:\n{formatted}")


def main():
    init_db()
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("forecast", forecast))
    app.add_handler(CommandHandler("paper", paper))
    app.add_handler(CommandHandler("evaluate", evaluate))
    app.add_handler(CommandHandler("evaluate_paper", evaluate_paper))
    app.add_handler(CommandHandler("positions", positions))
    app.add_handler(CommandHandler("paper_positions", paper_positions))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("add", add))
    app.add_handler(CommandHandler("remove", remove))
    app.add_handler(CommandHandler("watchlist", list_watchlist))

    app.run_polling()


if __name__ == "__main__":
    main()
