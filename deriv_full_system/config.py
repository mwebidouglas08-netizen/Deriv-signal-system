"""
config.py
─────────────────────────────────────────────────────────────────────────────
Configuration loaded from environment variables (Railway-compatible).
Set these in Railway → your project → Variables tab.

For local development create a .env file and run:
    pip install python-dotenv
─────────────────────────────────────────────────────────────────────────────
"""

import os

def _env(key, default=None, required=False):
    val = os.environ.get(key, default)
    if required and not val:
        raise EnvironmentError(
            f"Required env var '{key}' is not set.\n"
            f"Set it in Railway → Variables, or in your local .env file."
        )
    return val

def _float(key, default): return float(_env(key, default))
def _int(key, default):   return int(_env(key, default))

# ── DERIV ─────────────────────────────────────────────────────────────────────
DERIV_API_TOKEN = _env("DERIV_API_TOKEN", required=True)

# ── TELEGRAM ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN", default="")
TELEGRAM_CHAT_ID   = _env("TELEGRAM_CHAT_ID",   default="")

# ── SYMBOLS ───────────────────────────────────────────────────────────────────
_raw_symbols = _env("SYMBOLS", "R_75,R_100,1HZ100V,BOOM1000,CRASH1000")

_LABEL_MAP = {
    "R_10":      "Volatility 10 Index",
    "R_25":      "Volatility 25 Index",
    "R_50":      "Volatility 50 Index",
    "R_75":      "Volatility 75 Index",
    "R_100":     "Volatility 100 Index",
    "1HZ10V":    "Volatility 10 (1s) Index",
    "1HZ25V":    "Volatility 25 (1s) Index",
    "1HZ50V":    "Volatility 50 (1s) Index",
    "1HZ75V":    "Volatility 75 (1s) Index",
    "1HZ100V":   "Volatility 100 (1s) Index",
    "BOOM300":   "Boom 300 Index",
    "BOOM500":   "Boom 500 Index",
    "BOOM1000":  "Boom 1000 Index",
    "CRASH300":  "Crash 300 Index",
    "CRASH500":  "Crash 500 Index",
    "CRASH1000": "Crash 1000 Index",
}

SYMBOLS = {
    s.strip(): _LABEL_MAP.get(s.strip(), s.strip())
    for s in _raw_symbols.split(",") if s.strip()
}

# ── SIGNAL ENGINE ─────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = _float("CONFIDENCE_THRESHOLD", 0.68)
COOLDOWN_SECONDS     = _int("COOLDOWN_SECONDS", 180)
TIMEFRAME            = _env("TIMEFRAME", "1m")

# ── RISK MANAGEMENT ───────────────────────────────────────────────────────────
RISK_PCT    = _float("RISK_PCT",    1.0)
RR_RATIO    = _float("RR_RATIO",   2.0)
ATR_SL_MULT = _float("ATR_SL_MULT", 1.5)

# ── MODEL ─────────────────────────────────────────────────────────────────────
MODEL_SAVE_PATH     = _env("MODEL_SAVE_PATH", "/tmp/model.pkl")
MODEL_SAVE_INTERVAL = _int("MODEL_SAVE_INTERVAL", 300)

# ── LOGGING ───────────────────────────────────────────────────────────────────
LOG_LEVEL = _env("LOG_LEVEL", "INFO")
LOG_FILE  = _env("LOG_FILE", None)
