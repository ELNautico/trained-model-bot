# alert.py
import requests, toml, pathlib, logging

_cfg = toml.load(pathlib.Path(__file__).with_name("config.toml"))
TOKEN   = _cfg["telegram"]["bot_token"]
CHAT_ID = _cfg["telegram"]["chat_id"]

def send(text: str):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": text})
    except Exception as exc:
        logging.error("Telegram failed: %s", exc)
