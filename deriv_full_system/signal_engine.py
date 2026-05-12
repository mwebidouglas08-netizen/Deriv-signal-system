"""
signal_engine.py
─────────────────────────────────────────────────────────────────────────────
Signal generation pipeline:
  1. Pull candles from DataBus
  2. Run feature_engine()
  3. Run DirectionModel.predict()
  4. Apply signal filters (confidence, volatility, choppiness, cooldown)
  5. Compute risk parameters via RiskManager
  6. Emit a Signal object (or None)
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from data_collector import DataBus, Tick
from model import DirectionModel, Features, feature_engine

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration defaults (overridden from config.py at startup)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONF_THRESHOLD   = 0.68    # minimum model confidence
DEFAULT_RISK_PCT         = 1.0     # % of account at risk per trade
DEFAULT_RR_RATIO         = 2.0     # minimum reward:risk
DEFAULT_ATR_SL_MULT      = 1.5     # stop-loss = ATR * this
DEFAULT_COOLDOWN_SECS    = 180     # seconds between signals per symbol
DEFAULT_MIN_BB_WIDTH     = 0.0008  # skip flat (width < this fraction of price)
DEFAULT_MIN_TREND_STR    = 0.005   # skip choppy (trend-strength < this)
DEFAULT_MAX_RSI_NEUTRAL  = (45, 55)  # skip when RSI inside this range AND trend weak


# ──────────────────────────────────────────────────────────────────────────────
# Signal dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    # identity
    signal_id:   str
    timestamp:   float = field(default_factory=time.time)

    # symbol
    symbol:      str = ""
    symbol_label: str = ""

    # direction
    direction:   str = ""         # "BUY" | "SELL"
    confidence:  float = 0.0      # 0-1
    model_src:   str = ""         # "sklearn" | "heuristic"

    # prices
    entry_price: float = 0.0
    stop_loss:   float = 0.0
    take_profit: float = 0.0

    # risk
    atr:         float = 0.0
    sl_dist:     float = 0.0
    tp_dist:     float = 0.0
    rr_actual:   float = 0.0
    risk_pct:    float = 0.0

    # indicators snapshot
    rsi:         Optional[float] = None
    macd_line:   Optional[float] = None
    ema9:        Optional[float] = None
    ema21:       Optional[float] = None
    bb_upper:    Optional[float] = None
    bb_lower:    Optional[float] = None

    # human-readable explanation
    reason:      str = ""
    timeframe:   str = "1m"

    def to_dict(self) -> dict:
        return {
            "symbol":       self.symbol,
            "signal":       self.direction,
            "confidence":   round(self.confidence, 4),
            "entry_price":  round(self.entry_price, 5),
            "stop_loss":    round(self.stop_loss, 5),
            "take_profit":  round(self.take_profit, 5),
            "rr_ratio":     round(self.rr_actual, 2),
            "atr":          round(self.atr, 5),
            "rsi":          round(self.rsi, 2) if self.rsi else None,
            "reason":       self.reason,
            "model":        self.model_src,
            "timeframe":    self.timeframe,
            "timestamp":    self.timestamp,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Risk manager
# ──────────────────────────────────────────────────────────────────────────────

class RiskManager:
    def __init__(
        self,
        atr_sl_mult: float = DEFAULT_ATR_SL_MULT,
        rr_ratio:    float = DEFAULT_RR_RATIO,
        risk_pct:    float = DEFAULT_RISK_PCT,
    ):
        self.atr_sl_mult = atr_sl_mult
        self.rr_ratio    = rr_ratio
        self.risk_pct    = risk_pct

    def calculate(
        self, direction: str, entry: float, atr: float
    ) -> Optional[Dict]:
        """
        Returns {"sl", "tp", "sl_dist", "tp_dist", "rr"} or None if ATR invalid.
        """
        if atr <= 0:
            return None

        sl_dist = atr * self.atr_sl_mult
        tp_dist = sl_dist * self.rr_ratio

        if direction == "BUY":
            sl = max(0.0, entry - sl_dist)
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = max(0.0, entry - tp_dist)

        rr = tp_dist / sl_dist if sl_dist > 0 else 0.0

        return {
            "sl":      sl,
            "tp":      tp,
            "sl_dist": sl_dist,
            "tp_dist": tp_dist,
            "rr":      rr,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Signal engine
# ──────────────────────────────────────────────────────────────────────────────

class SignalEngine:
    """
    One instance per symbol.  Call `evaluate(symbol)` on every tick (or
    every N ticks).  Returns a Signal or None.
    """

    def __init__(
        self,
        symbol:         str,
        symbol_label:   str,
        bus:            DataBus,
        model:          DirectionModel,
        risk_manager:   RiskManager,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        cooldown_secs:  int   = DEFAULT_COOLDOWN_SECS,
        timeframe:      str   = "1m",
    ):
        self.symbol         = symbol
        self.symbol_label   = symbol_label
        self.bus            = bus
        self.model          = model
        self.risk_manager   = risk_manager
        self.conf_threshold = conf_threshold
        self.cooldown_secs  = cooldown_secs
        self.timeframe      = timeframe

        self._last_signal_time:  float = 0.0
        self._signal_count:      int   = 0
        self._pending_features:  Optional[Features] = None  # for online update
        self._pending_price:     float = 0.0
        self._outcome_check_at:  float = 0.0      # epoch to check outcome

    # ── public ─────────────────────────────────────────────────────────────────
    def evaluate(self) -> Optional[Signal]:
        """
        Run full pipeline for this symbol.
        Called on every tick — returns Signal or None.
        """
        self._maybe_update_model()

        candles = self.bus.all_candles(self.symbol, self.timeframe)
        if len(candles) < 30:
            logger.debug("%s: not enough candles yet (%d)", self.symbol, len(candles))
            return None

        # ── cooldown ────────────────────────────────────────────────────────
        now = time.time()
        remaining = self.cooldown_secs - (now - self._last_signal_time)
        if remaining > 0:
            logger.debug("%s: cooldown %.0fs remaining", self.symbol, remaining)
            return None

        # ── features ────────────────────────────────────────────────────────
        feats = feature_engine(candles)
        if not feats.valid:
            logger.debug("%s: features not valid yet", self.symbol)
            return None

        raw = feats.raw
        price = raw["price"]

        # ── market quality filters ───────────────────────────────────────────
        skip, skip_reason = self._market_filters(raw, feats)
        if skip:
            logger.info("%s: SKIP — %s", self.symbol, skip_reason)
            return None

        # ── model prediction ─────────────────────────────────────────────────
        pred = self.model.predict(feats)
        if pred is None:
            return None

        conf = pred["confidence"]
        if conf < self.conf_threshold:
            logger.debug(
                "%s: confidence %.1f%% below threshold %.1f%%",
                self.symbol, conf * 100, self.conf_threshold * 100,
            )
            return None

        direction = "BUY" if pred["direction"] == "UP" else "SELL"

        # ── risk parameters ──────────────────────────────────────────────────
        rp = self.risk_manager.calculate(direction, price, raw["atr"])
        if rp is None:
            logger.warning("%s: risk calculation failed", self.symbol)
            return None

        if rp["rr"] < self.risk_manager.rr_ratio - 0.05:
            logger.info("%s: RR %.2f < %.2f — skip", self.symbol, rp["rr"], self.risk_manager.rr_ratio)
            return None

        # ── build reason string ──────────────────────────────────────────────
        reason = self._build_reason(direction, raw, feats)

        # ── assemble signal ──────────────────────────────────────────────────
        self._signal_count += 1
        sig_id = f"{self.symbol}-{int(now)}-{self._signal_count}"

        signal = Signal(
            signal_id    = sig_id,
            symbol       = self.symbol,
            symbol_label = self.symbol_label,
            direction    = direction,
            confidence   = conf,
            model_src    = pred["model"],
            entry_price  = price,
            stop_loss    = rp["sl"],
            take_profit  = rp["tp"],
            atr          = raw["atr"],
            sl_dist      = rp["sl_dist"],
            tp_dist      = rp["tp_dist"],
            rr_actual    = rp["rr"],
            risk_pct     = self.risk_manager.risk_pct,
            rsi          = raw.get("rsi"),
            macd_line    = raw.get("macd_line"),
            ema9         = raw.get("ema9"),
            ema21        = raw.get("ema21"),
            bb_upper     = raw.get("bb_upper"),
            bb_lower     = raw.get("bb_lower"),
            reason       = reason,
            timeframe    = self.timeframe,
        )

        self._last_signal_time   = now
        self._pending_features   = feats
        self._pending_price      = price
        self._outcome_check_at   = now + 60  # check outcome in 60 seconds

        logger.info(
            "SIGNAL ▶ %s %s | conf=%.1f%% | entry=%.5f SL=%.5f TP=%.5f | %s",
            self.symbol, direction, conf * 100,
            price, rp["sl"], rp["tp"], reason,
        )
        return signal

    # ── private helpers ────────────────────────────────────────────────────────

    def _market_filters(self, raw: dict, feats: Features) -> tuple:
        """Returns (should_skip: bool, reason: str)"""
        # Low volatility (flat market)
        bb_width_pct = raw.get("bb_upper", 0) - raw.get("bb_lower", 0)
        if raw.get("price", 1) > 0:
            bb_frac = bb_width_pct / raw["price"]
        else:
            bb_frac = 0
        if bb_frac < DEFAULT_MIN_BB_WIDTH:
            return True, f"BB width {bb_frac:.5f} < min {DEFAULT_MIN_BB_WIDTH}"

        # Choppy/sideways: RSI neutral + weak trend
        rsi = raw.get("rsi", 50)
        ts  = feats.vector[FEATURE_NAMES_IDX("trend_strength")] if hasattr(feats, "vector") else 0
        ts  = float(abs(raw.get("ema9", 1) - raw.get("ema21", 1)) / max(raw.get("price", 1), 1e-9))
        lo, hi = DEFAULT_MAX_RSI_NEUTRAL
        if lo < rsi < hi and ts < DEFAULT_MIN_TREND_STR:
            return True, f"Sideways market (RSI={rsi:.1f}, trend={ts:.5f})"

        return False, ""

    def _build_reason(self, direction: str, raw: dict, feats: Features) -> str:
        parts = []
        rsi  = raw.get("rsi", 50)
        macd = raw.get("macd_line", 0)
        ema9  = raw.get("ema9", 0)
        ema21 = raw.get("ema21", 0)
        bb_pos = feats.vector[5] if feats.valid else 0.5  # index 5 = bb_pos

        if direction == "BUY":
            if rsi < 35:  parts.append(f"RSI oversold ({rsi:.0f})")
            if macd > 0:  parts.append("MACD bullish")
            if ema9 > ema21: parts.append("EMA9 > EMA21 ↑")
            if bb_pos < 0.25: parts.append("Price near BB lower band")
        else:
            if rsi > 65:  parts.append(f"RSI overbought ({rsi:.0f})")
            if macd < 0:  parts.append("MACD bearish")
            if ema9 < ema21: parts.append("EMA9 < EMA21 ↓")
            if bb_pos > 0.75: parts.append("Price near BB upper band")

        return " + ".join(parts) if parts else "Model convergence"

    def _maybe_update_model(self):
        """Check if a pending prediction outcome can be observed and update model."""
        if self._pending_features is None:
            return
        if time.time() < self._outcome_check_at:
            return

        current_price = self.bus.latest_price(self.symbol)
        if current_price is None:
            return

        outcome = 1 if current_price > self._pending_price else 0
        self.model.update(self._pending_features, outcome)
        logger.debug(
            "%s: model updated — outcome=%s (%.5f → %.5f)",
            self.symbol, "UP" if outcome else "DOWN",
            self._pending_price, current_price,
        )
        self._pending_features = None
        self._pending_price    = 0.0
        self._outcome_check_at = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Helper to index feature names
# ──────────────────────────────────────────────────────────────────────────────
from model import FEATURE_NAMES  # noqa: E402

def FEATURE_NAMES_IDX(name: str) -> int:
    return FEATURE_NAMES.index(name)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-symbol orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class MultiSymbolEngine:
    """
    Wraps one SignalEngine per symbol.
    Call `on_tick(tick)` — registered as a DataBus listener.
    Collected signals are put into the asyncio queue for the notifier.
    """

    def __init__(
        self,
        symbols:      Dict[str, str],   # {code: label}
        bus:          DataBus,
        model:        DirectionModel,
        risk_manager: RiskManager,
        signal_queue: "asyncio.Queue",
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        cooldown_secs:  int   = DEFAULT_COOLDOWN_SECS,
    ):
        self._queue = signal_queue
        self._engines: Dict[str, SignalEngine] = {}
        self._tick_counter: Dict[str, int] = {}

        for code, label in symbols.items():
            self._engines[code] = SignalEngine(
                symbol         = code,
                symbol_label   = label,
                bus            = bus,
                model          = model,
                risk_manager   = risk_manager,
                conf_threshold = conf_threshold,
                cooldown_secs  = cooldown_secs,
            )
            self._tick_counter[code] = 0

        bus.subscribe(self._on_tick)
        logger.info("MultiSymbolEngine ready for: %s", list(symbols.keys()))

    def _on_tick(self, tick: Tick):
        code = tick.symbol
        if code not in self._engines:
            return

        self._tick_counter[code] += 1
        # evaluate every 5th tick to reduce CPU load
        if self._tick_counter[code] % 5 != 0:
            return

        engine = self._engines[code]
        signal = engine.evaluate()
        if signal is not None:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.call_soon_threadsafe(self._queue.put_nowait, signal)
            except Exception as exc:
                logger.warning("Failed to queue signal: %s", exc)
