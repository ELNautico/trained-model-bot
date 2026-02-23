import requests, toml, pathlib, logging
import requests
from requests.adapters import HTTPAdapter, Retry

_session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500,502,503,504])
_session.mount("https://", HTTPAdapter(max_retries=retries))

_cfg_path = pathlib.Path(__file__).with_name("config.toml")
if not _cfg_path.exists():
    _cfg_path = pathlib.Path(__file__).parent.parent / "config.toml"
_cfg = toml.load(_cfg_path)
TOKEN   = _cfg["telegram"]["bot_token"]
CHAT_ID = _cfg["telegram"]["chat_id"]

def send(text: str):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        _session.post(url, json={"chat_id": CHAT_ID, "text": text}, timeout=5)
    except Exception as exc:
        logging.error("Telegram failed: %s", exc)
