"""
data_collector.py
─────────────────────────────────────────────────────────────────────────────
Deriv WebSocket API integration.
  • Authorises with your API token
  • Subscribes to synthetic-index tick streams
  • Assembles ticks into OHLC candles (1 min, 5 min)
  • Exposes a DataBus that every other module reads from
  • Reconnects automatically on any drop
─────────────────────────────────────────────────────────────────────────────
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DERIV_WS_URL    = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
PING_INTERVAL   = 25        # seconds — keep-alive ping
MAX_TICKS       = 5_000     # rolling tick window per symbol
MAX_CANDLES     = 500       # max candles kept per timeframe per symbol
RECONNECT_DELAY = 5         # seconds before reconnect attempt

TIMEFRAMES = {
    "1m":  60,
    "5m":  300,
}

# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Tick:
    symbol:    str
    epoch:     int          # unix timestamp (seconds)
    price:     float
    received:  float = field(default_factory=time.time)


@dataclass
class Candle:
    symbol:    str
    timeframe: str
    epoch:     int          # candle start (unix seconds)
    open:      float
    high:      float
    low:       float
    close:     float
    tick_count: int = 1
    closed:    bool = False  # True once the next candle starts


class DataBus:
    """
    Thread-safe (asyncio) in-memory store shared by all modules.

    Modules read:
        bus.ticks[symbol]               → deque of recent Tick objects
        bus.candles[symbol][timeframe]  → list of completed Candle objects
        bus.live_candle[symbol][tf]     → current open Candle (may be None)
        bus.latest_price(symbol)        → float | None
    """

    def __init__(self):
        self.ticks:       Dict[str, deque]          = defaultdict(lambda: deque(maxlen=MAX_TICKS))
        self.candles:     Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        self.live_candle: Dict[str, Dict[str, Optional[Candle]]] = \
            defaultdict(lambda: defaultdict(lambda: None))
        self._listeners:  List[Callable]            = []

    def latest_price(self, symbol: str) -> Optional[float]:
        if not self.ticks[symbol]:
            return None
        return self.ticks[symbol][-1].price

    def add_tick(self, tick: Tick):
        self.ticks[tick.symbol].append(tick)
        self._build_candles(tick)
        for cb in self._listeners:
            try:
                cb(tick)
            except Exception as exc:
                logger.warning("Listener error: %s", exc)

    def subscribe(self, callback: Callable):
        """Register a function(tick: Tick) called on every new tick."""
        self._listeners.append(callback)

    # ── internal candle builder ────────────────────────────────────────────────
    def _build_candles(self, tick: Tick):
        for tf, seconds in TIMEFRAMES.items():
            slot = (tick.epoch // seconds) * seconds
            live = self.live_candle[tick.symbol][tf]

            if live is None or live.epoch != slot:
                # close the previous candle
                if live is not None:
                    live.closed = True
                    self.candles[tick.symbol][tf].append(live)
                    # trim to max length
                    if len(self.candles[tick.symbol][tf]) > MAX_CANDLES:
                        self.candles[tick.symbol][tf].pop(0)

                # open a new candle
                self.live_candle[tick.symbol][tf] = Candle(
                    symbol=tick.symbol, timeframe=tf, epoch=slot,
                    open=tick.price, high=tick.price,
                    low=tick.price,  close=tick.price,
                )
            else:
                # update current candle
                live.high  = max(live.high, tick.price)
                live.low   = min(live.low,  tick.price)
                live.close = tick.price
                live.tick_count += 1

    def closed_candles(self, symbol: str, tf: str, n: int = 200) -> List[Candle]:
        """Return the last *n* fully-closed candles for a symbol + timeframe."""
        return self.candles[symbol][tf][-n:]

    def all_candles(self, symbol: str, tf: str, n: int = 200) -> List[Candle]:
        """Closed candles + the current live candle appended (if any)."""
        closed = self.closed_candles(symbol, tf, n)
        live   = self.live_candle[symbol][tf]
        return closed + ([live] if live else [])


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket collector
# ──────────────────────────────────────────────────────────────────────────────

class DerivCollector:
    """
    Manages one long-lived WebSocket connection to Deriv.
    Call `await collector.run()` — it loops forever, reconnecting on drops.
    """

    def __init__(self, api_token: str, symbols: List[str], bus: DataBus):
        self.api_token = api_token
        self.symbols   = symbols
        self.bus       = bus
        self._ws       = None
        self._running  = False
        self._authorised = False

    # ── public API ─────────────────────────────────────────────────────────────
    async def run(self):
        """Entry point — reconnects forever."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_stream()
            except Exception as exc:
                logger.error("Collector crashed: %s — reconnecting in %ds", exc, RECONNECT_DELAY)
            if self._running:
                await asyncio.sleep(RECONNECT_DELAY)

    async def stop(self):
        self._running = False
        if self._ws:
            await self._ws.close()

    # ── internals ──────────────────────────────────────────────────────────────
    async def _connect_and_stream(self):
        logger.info("Connecting to Deriv WebSocket …")
        async with websockets.connect(
            DERIV_WS_URL,
            ping_interval=None,     # we manage pings ourselves
            max_size=2**20,
        ) as ws:
            self._ws       = ws
            self._authorised = False
            logger.info("WebSocket connected — authorising …")

            await self._send(ws, {"authorize": self.api_token})

            ping_task = asyncio.create_task(self._ping_loop(ws))
            try:
                async for raw in ws:
                    await self._handle(ws, raw)
            except ConnectionClosed as exc:
                logger.warning("WebSocket closed: %s", exc)
            finally:
                ping_task.cancel()
                self._authorised = False

    async def _send(self, ws, payload: dict):
        await ws.send(json.dumps(payload))

    async def _ping_loop(self, ws):
        while True:
            await asyncio.sleep(PING_INTERVAL)
            try:
                await self._send(ws, {"ping": 1})
            except Exception:
                break

    async def _handle(self, ws, raw: str):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        if "error" in msg:
            err = msg["error"]
            logger.error("Deriv API error [%s]: %s", err.get("code"), err.get("message"))
            return

        mt = msg.get("msg_type")

        if mt == "authorize":
            info = msg["authorize"]
            logger.info(
                "Authorised as %s | balance: %.2f %s",
                info.get("loginid"), info.get("balance", 0), info.get("currency", "")
            )
            self._authorised = True
            # subscribe to all symbols
            for sym in self.symbols:
                await self._send(ws, {"ticks": sym, "subscribe": 1})
                logger.info("Subscribed to %s", sym)

        elif mt == "tick":
            t = msg["tick"]
            tick = Tick(
                symbol=t["symbol"],
                epoch=int(t["epoch"]),
                price=float(t["quote"]),
            )
            self.bus.add_tick(tick)
            logger.debug("TICK %s  %.5f", tick.symbol, tick.price)

        elif mt == "ping":
            pass  # keep-alive ack

        else:
            logger.debug("Unhandled msg_type: %s", mt)
