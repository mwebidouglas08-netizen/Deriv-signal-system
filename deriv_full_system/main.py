"""
main.py  —  Deriv AI Signal System (orchestrator)
─────────────────────────────────────────────────────────────────────────────
Run locally:   python main.py
Deploy:        Railway auto-runs this via Procfile
─────────────────────────────────────────────────────────────────────────────
"""

import asyncio
import logging
import os
import signal
import sys
import time

# Load .env for local dev (ignored if python-dotenv not installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import config
from data_collector import DataBus, DerivCollector
from model import DirectionModel
from notifier import TelegramNotifier
from signal_engine import MultiSymbolEngine, RiskManager

# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging():
    handlers = [logging.StreamHandler(sys.stdout)]
    if config.LOG_FILE:
        handlers.append(logging.FileHandler(config.LOG_FILE))
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

logger = logging.getLogger("main")

# ── Model saver task ──────────────────────────────────────────────────────────
async def model_saver(model: DirectionModel, interval: int, path: str):
    while True:
        await asyncio.sleep(interval)
        try:
            model.save(path)
            logger.debug("Model saved → %s", path)
        except Exception as exc:
            logger.warning("Model save failed: %s", exc)

# ── Entry point ───────────────────────────────────────────────────────────────
async def main():
    setup_logging()

    logger.info("=" * 60)
    logger.info("  Deriv AI Signal System — starting")
    logger.info("  Symbols    : %s", list(config.SYMBOLS.keys()))
    logger.info("  Confidence : %.0f%%", config.CONFIDENCE_THRESHOLD * 100)
    logger.info("  RR ratio   : 1:%.1f  |  Risk/trade: %.1f%%",
                config.RR_RATIO, config.RISK_PCT)
    logger.info("  Telegram   : %s",
                "enabled" if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID
                else "disabled (console only)")
    logger.info("=" * 60)

    # Shared objects
    bus          = DataBus()
    model        = DirectionModel()
    signal_queue = asyncio.Queue()

    # Load persisted model if available
    if os.path.exists(config.MODEL_SAVE_PATH):
        try:
            model.load(config.MODEL_SAVE_PATH)
            logger.info("Loaded persisted model from %s", config.MODEL_SAVE_PATH)
        except Exception as exc:
            logger.warning("Could not load model (%s) — using heuristic weights", exc)

    # Risk manager
    risk_manager = RiskManager(
        atr_sl_mult=config.ATR_SL_MULT,
        rr_ratio=config.RR_RATIO,
        risk_pct=config.RISK_PCT,
    )

    # Signal engine (registers as DataBus listener)
    _engine = MultiSymbolEngine(
        symbols=config.SYMBOLS,
        bus=bus,
        model=model,
        risk_manager=risk_manager,
        signal_queue=signal_queue,
        conf_threshold=config.CONFIDENCE_THRESHOLD,
        cooldown_secs=config.COOLDOWN_SECONDS,
    )

    # Notifier
    notifier = TelegramNotifier(
        bot_token=config.TELEGRAM_BOT_TOKEN,
        chat_id=config.TELEGRAM_CHAT_ID,
        signal_queue=signal_queue,
        symbols=list(config.SYMBOLS.keys()),
    )

    # Data collector
    collector = DerivCollector(
        api_token=config.DERIV_API_TOKEN,
        symbols=list(config.SYMBOLS.keys()),
        bus=bus,
    )

    # Graceful shutdown
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _shutdown(sig_name):
        logger.info("Received %s — shutting down ...", sig_name)
        stop_event.set()

    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(
                getattr(signal, sig_name),
                lambda s=sig_name: _shutdown(s),
            )
        except (NotImplementedError, AttributeError):
            pass  # Windows

    # Launch tasks
    tasks = [
        asyncio.create_task(collector.run(),  name="collector"),
        asyncio.create_task(notifier.run(),   name="notifier"),
        asyncio.create_task(
            model_saver(model, config.MODEL_SAVE_INTERVAL, config.MODEL_SAVE_PATH),
            name="model_saver",
        ),
    ]

    logger.info("All tasks running — system is live")

    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Stopping tasks ...")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            model.save(config.MODEL_SAVE_PATH)
        except Exception:
            pass
        logger.info("Shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
