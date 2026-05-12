"""
notifier.py
─────────────────────────────────────────────────────────────────────────────
Telegram bot notifier.

Features
  • Sends rich formatted signal alerts with all parameters
  • /start    — welcome message + instructions
  • /status   — current system status + last signal per symbol
  • /signals  — last 5 signals
  • /help     — list all commands
  • Duplicate-signal guard (same symbol + direction within 60 s = skip)
  • Async — runs alongside the data collector in the same event loop
─────────────────────────────────────────────────────────────────────────────
"""

import asyncio
import logging
import time
from collections import deque
from typing import Dict, Optional

from signal_engine import Signal

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Emoji / formatting constants
# ──────────────────────────────────────────────────────────────────────────────

EMOJI = {
    "BUY":   "📈",
    "SELL":  "📉",
    "INFO":  "ℹ️",
    "WARN":  "⚠️",
    "OK":    "✅",
    "CLOCK": "⏱",
    "CONF":  "🎯",
    "SL":    "🛑",
    "TP":    "💰",
    "ENTRY": "🔖",
    "REASON":"💡",
    "MODEL": "🤖",
    "RR":    "⚖️",
}


def _conf_bar(confidence: float, width: int = 10) -> str:
    """▓▓▓▓▓▓░░░░  70%"""
    filled = round(confidence * width)
    return "▓" * filled + "░" * (width - filled) + f"  {confidence * 100:.0f}%"


def format_signal(sig: Signal) -> str:
    """Return a Markdown-safe Telegram message for a trading signal."""
    bull = sig.direction == "BUY"
    arrow = "🟢 BUY" if bull else "🔴 SELL"

    lines = [
        f"{EMOJI[sig.direction]} *{arrow}*  \\|  {sig.symbol_label}",
        "",
        f"{EMOJI['CONF']} Confidence:  `{_conf_bar(sig.confidence)}`",
        f"{EMOJI['MODEL']} Model: `{sig.model_src.upper()}`",
        "",
        f"{EMOJI['ENTRY']} Entry:       `{sig.entry_price:.5f}`",
        f"{EMOJI['SL']}    Stop loss:  `{sig.stop_loss:.5f}`",
        f"{EMOJI['TP']}    Take profit:`{sig.take_profit:.5f}`",
        f"{EMOJI['RR']}  Risk\\:Reward: `1 : {sig.rr_actual:.1f}`",
        "",
    ]

    # indicator snapshot
    ind_parts = []
    if sig.rsi is not None:
        ind_parts.append(f"RSI `{sig.rsi:.1f}`")
    if sig.macd_line is not None:
        ind_parts.append(f"MACD `{sig.macd_line:+.5f}`")
    if sig.ema9 is not None and sig.ema21 is not None:
        cross = "↑" if sig.ema9 > sig.ema21 else "↓"
        ind_parts.append(f"EMA9/21 {cross}")
    if ind_parts:
        lines.append("📊 Indicators: " + " \\| ".join(ind_parts))
        lines.append("")

    lines.append(f"{EMOJI['REASON']} Reason: _{sig.reason}_")
    lines.append(f"{EMOJI['CLOCK']} Timeframe: `{sig.timeframe}`")
    lines.append("")
    lines.append(
        "⚠️ _This is a signal, not financial advice\\. "
        "Always manage your own risk\\._"
    )

    return "\n".join(lines)


def format_status(
    symbols: list,
    last_signals: Dict[str, Optional[Signal]],
    start_time: float,
    total_signals: int,
) -> str:
    uptime_m = int((time.time() - start_time) / 60)
    lines = [
        "📡 *System Status*",
        "",
        f"⏱ Uptime: `{uptime_m} min`",
        f"📊 Total signals sent: `{total_signals}`",
        f"📌 Monitoring: `{', '.join(symbols)}`",
        "",
        "*Last signal per symbol:*",
    ]
    for sym in symbols:
        s = last_signals.get(sym)
        if s:
            age_m = int((time.time() - s.timestamp) / 60)
            lines.append(
                f"  • {s.symbol_label}: "
                f"{'🟢' if s.direction=='BUY' else '🔴'} {s.direction} "
                f"`{s.confidence*100:.0f}%` — {age_m} min ago"
            )
        else:
            lines.append(f"  • {sym}: _no signal yet_")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Telegram notifier
# ──────────────────────────────────────────────────────────────────────────────

class TelegramNotifier:
    """
    Consumes signals from an asyncio.Queue and sends them to a Telegram chat.
    Also handles inbound commands from the bot.

    Usage:
        notifier = TelegramNotifier(
            bot_token="...",
            chat_id="...",
            signal_queue=queue,
            symbols=["R_75", "R_100"],
        )
        await notifier.run()
    """

    def __init__(
        self,
        bot_token:    str,
        chat_id:      str,
        signal_queue: asyncio.Queue,
        symbols:      list,
    ):
        self.bot_token    = bot_token
        self.chat_id      = chat_id
        self.queue        = signal_queue
        self.symbols      = symbols

        self._start_time   = time.time()
        self._total_sent   = 0
        self._last_signals: Dict[str, Optional[Signal]] = {s: None for s in symbols}
        self._history:      deque = deque(maxlen=20)

        # duplicate guard: (symbol, direction) → last sent epoch
        self._dedup: Dict[str, float] = {}
        self._dedup_window = 60  # seconds

        # lazy-import telegram so the module loads without the library
        try:
            from telegram import Bot, Update
            from telegram.ext import (
                Application, CommandHandler, ContextTypes
            )
            self._tg_available = True
        except ImportError:
            logger.warning(
                "python-telegram-bot not installed — Telegram notifications disabled. "
                "Run: pip install python-telegram-bot"
            )
            self._tg_available = False

    # ── public entry point ────────────────────────────────────────────────────
    async def run(self):
        if not self._tg_available:
            await self._run_console_fallback()
            return

        from telegram import Bot
        from telegram.ext import Application, CommandHandler

        app = (
            Application.builder()
            .token(self.bot_token)
            .build()
        )

        # register commands
        app.add_handler(CommandHandler("start",   self._cmd_start))
        app.add_handler(CommandHandler("status",  self._cmd_status))
        app.add_handler(CommandHandler("signals", self._cmd_signals))
        app.add_handler(CommandHandler("help",    self._cmd_help))

        self._bot = app.bot

        # start polling in background
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)

        logger.info("Telegram bot started — polling for commands")

        # send startup message
        try:
            await self._send(
                "🚀 *Deriv AI Signal System started\\!*\n\n"
                f"Monitoring: `{', '.join(self.symbols)}`\n"
                "Use /help to see available commands\\.",
                parse_mode="MarkdownV2",
            )
        except Exception as exc:
            logger.warning("Could not send startup message: %s", exc)

        # main signal-dispatch loop
        try:
            while True:
                try:
                    signal: Signal = await asyncio.wait_for(
                        self.queue.get(), timeout=5.0
                    )
                    await self._dispatch(signal)
                    self.queue.task_done()
                except asyncio.TimeoutError:
                    pass
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.error("Notifier loop error: %s", exc)
        finally:
            await app.updater.stop()
            await app.stop()
            await app.shutdown()

    # ── signal dispatch ───────────────────────────────────────────────────────
    async def _dispatch(self, sig: Signal):
        # duplicate check
        key = f"{sig.symbol}:{sig.direction}"
        last = self._dedup.get(key, 0)
        if time.time() - last < self._dedup_window:
            logger.info("Duplicate suppressed: %s %s", sig.symbol, sig.direction)
            return

        self._dedup[key] = time.time()
        self._last_signals[sig.symbol] = sig
        self._history.appendleft(sig)
        self._total_sent += 1

        text = format_signal(sig)
        try:
            await self._send(text, parse_mode="MarkdownV2")
            logger.info("Telegram signal sent: %s %s %.1f%%",
                        sig.symbol, sig.direction, sig.confidence * 100)
        except Exception as exc:
            logger.error("Failed to send Telegram message: %s", exc)
            # fallback: print to console
            print("\n" + "═" * 50)
            print(f"[SIGNAL] {sig.symbol} {sig.direction} {sig.confidence*100:.0f}%")
            print(f"  Entry: {sig.entry_price:.5f}  SL: {sig.stop_loss:.5f}  TP: {sig.take_profit:.5f}")
            print(f"  Reason: {sig.reason}")
            print("═" * 50 + "\n")

    async def _send(self, text: str, parse_mode: str = "MarkdownV2"):
        if not hasattr(self, "_bot"):
            print(text)
            return
        await self._bot.send_message(
            chat_id=self.chat_id,
            text=text,
            parse_mode=parse_mode,
        )

    # ── command handlers ──────────────────────────────────────────────────────
    async def _cmd_start(self, update, context):
        text = (
            "👋 *Welcome to Deriv AI Signal Bot\\!*\n\n"
            "I send real\\-time BUY/SELL signals for Deriv synthetic indices "
            "powered by technical indicators and a machine learning model\\.\n\n"
            "Use /help to see all commands\\."
        )
        await update.message.reply_text(text, parse_mode="MarkdownV2")

    async def _cmd_status(self, update, context):
        text = format_status(
            self.symbols,
            self._last_signals,
            self._start_time,
            self._total_sent,
        )
        await update.message.reply_text(text, parse_mode="Markdown")

    async def _cmd_signals(self, update, context):
        if not self._history:
            await update.message.reply_text("No signals sent yet\\.", parse_mode="MarkdownV2")
            return
        last5 = list(self._history)[:5]
        for sig in last5:
            await update.message.reply_text(
                format_signal(sig), parse_mode="MarkdownV2"
            )
            await asyncio.sleep(0.3)  # avoid Telegram flood limits

    async def _cmd_help(self, update, context):
        text = (
            "📋 *Available commands*\n\n"
            "/start — welcome message\n"
            "/status — system uptime and last signal per symbol\n"
            "/signals — last 5 signals sent\n"
            "/help — this message\n\n"
            "⚠️ _Signals are not financial advice\\. Trade at your own risk\\._"
        )
        await update.message.reply_text(text, parse_mode="MarkdownV2")

    # ── console fallback (no telegram library) ────────────────────────────────
    async def _run_console_fallback(self):
        logger.info("Running in console-output mode (Telegram not available)")
        while True:
            try:
                signal: Signal = await asyncio.wait_for(
                    self.queue.get(), timeout=5.0
                )
                self._print_signal(signal)
                self.queue.task_done()
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                break

    def _print_signal(self, sig: Signal):
        sep = "═" * 60
        bull = sig.direction == "BUY"
        print(f"\n{sep}")
        print(f"  {'🟢 BUY' if bull else '🔴 SELL'}  |  {sig.symbol_label}")
        print(f"  Confidence : {sig.confidence*100:.1f}%  [{sig.model_src}]")
        print(f"  Entry      : {sig.entry_price:.5f}")
        print(f"  Stop loss  : {sig.stop_loss:.5f}")
        print(f"  Take profit: {sig.take_profit:.5f}")
        print(f"  RR ratio   : 1:{sig.rr_actual:.1f}")
        if sig.rsi:   print(f"  RSI        : {sig.rsi:.1f}")
        if sig.macd_line: print(f"  MACD       : {sig.macd_line:+.5f}")
        print(f"  Reason     : {sig.reason}")
        print(f"  Timeframe  : {sig.timeframe}")
        print(sep + "\n")
