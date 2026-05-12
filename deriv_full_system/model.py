"""
model.py
─────────────────────────────────────────────────────────────────────────────
Feature engineering + AI model for Deriv synthetic indices.

Pipeline
  1.  feature_engine(candles)  →  feature vector (numpy array)
  2.  DirectionModel.predict(features) → probability UP, direction, confidence
  3.  DirectionModel.update(features, outcome) — online partial-fit

Model: Logistic Regression (scikit-learn) trained with SGD so it can be
       updated in real-time as new labelled outcomes arrive.

We also ship hand-crafted heuristic weights as a cold-start fallback so the
system emits signals from minute one without any prior training data.
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Feature names (must stay in this exact order)
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "rsi_norm",          # (RSI - 50) / 50           ∈ [-1, 1]
    "macd_sign",         # sign(MACD line)            ∈ {-1, 1}
    "macd_hist_norm",    # histogram / price          ∈ [-1, 1] (clipped)
    "ema_cross",         # sign(EMA9 - EMA21)         ∈ {-1, 1}
    "ema_slope_norm",    # EMA9 slope / price * 100   ∈ [-1, 1]
    "bb_pos",            # (price - lower) / band     ∈ [0, 1]
    "bb_width_norm",     # band width / price * 100   normalised
    "atr_norm",          # ATR / price * 1000         normalised
    "momentum_norm",     # 10-bar momentum / price    ∈ [-1, 1]
    "velocity_norm",     # 5-bar rate-of-change %     ∈ [-1, 1]
    "trend_strength",    # |EMA9 - EMA21| / price * 1000
    "vol_spike",         # ATR vs 20-bar avg ATR ratio
]

N_FEATURES = len(FEATURE_NAMES)


# ──────────────────────────────────────────────────────────────────────────────
# Technical indicator helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    e = sum(values[:period]) / period
    for v in values[period:]:
        e = v * k + e * (1 - k)
    return e


def _rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        delta = closes[-period - 1 + i] - closes[-period - 1 + i - 1]
        if delta > 0:
            gains += delta
        else:
            losses -= delta
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100 - 100 / (1 + rs)


def _macd(closes: List[float]) -> Optional[Dict]:
    if len(closes) < 26:
        return None
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    if ema12 is None or ema26 is None:
        return None
    macd_line = ema12 - ema26
    # approximate signal line
    signal_line = macd_line * 0.88
    return {
        "macd":      macd_line,
        "signal":    signal_line,
        "histogram": macd_line - signal_line,
    }


def _bollinger(closes: List[float], period: int = 20, mult: float = 2.0) -> Optional[Dict]:
    if len(closes) < period:
        return None
    sl = closes[-period:]
    ma = sum(sl) / period
    sd = math.sqrt(sum((x - ma) ** 2 for x in sl) / period)
    return {
        "upper":  ma + mult * sd,
        "middle": ma,
        "lower":  ma - mult * sd,
        "width":  4 * mult * sd,
    }


def _atr(candles, period: int = 14) -> Optional[float]:
    if len(candles) < period + 1:
        return None
    recent = candles[-(period + 1):]
    trs = []
    for i in range(1, len(recent)):
        h, l, pc = recent[i].high, recent[i].low, recent[i - 1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs) / len(trs)


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Features:
    vector:  np.ndarray          # shape (N_FEATURES,)
    raw:     Dict[str, float]    # human-readable indicator values
    valid:   bool                # False if not enough data yet


def feature_engine(candles) -> Features:
    """
    Build a feature vector from a list of Candle objects (oldest → newest).
    Requires at least 30 closed candles for a valid feature set.
    """
    empty = Features(
        vector=np.zeros(N_FEATURES, dtype=np.float32),
        raw={},
        valid=False,
    )

    if len(candles) < 30:
        return empty

    closes = [c.close for c in candles]
    price  = closes[-1]

    if price == 0:
        return empty

    # ── indicators ────────────────────────────────────────────────────────────
    rsi  = _rsi(closes)
    macd = _macd(closes)
    bb   = _bollinger(closes)
    atr  = _atr(candles)
    ema9  = _ema(closes, 9)
    ema21 = _ema(closes, 21)
    ema9_prev = _ema(closes[:-1], 9) if len(closes) > 9 else None

    if any(v is None for v in [rsi, macd, bb, atr, ema9, ema21]):
        return empty

    # ── raw dict ──────────────────────────────────────────────────────────────
    raw: Dict[str, float] = {
        "rsi":       rsi,
        "macd_line": macd["macd"],
        "macd_hist": macd["histogram"],
        "ema9":      ema9,
        "ema21":     ema21,
        "bb_upper":  bb["upper"],
        "bb_lower":  bb["lower"],
        "bb_middle": bb["middle"],
        "atr":       atr,
        "price":     price,
    }

    # ── compute momentum & velocity ───────────────────────────────────────────
    momentum = (closes[-1] - closes[-11]) if len(closes) >= 11 else 0.0
    velocity = ((closes[-1] - closes[-6]) / closes[-6] * 100) if len(closes) >= 6 else 0.0

    raw["momentum"] = momentum
    raw["velocity"] = velocity

    # ── ATR history for vol-spike ratio ───────────────────────────────────────
    if len(candles) >= 35:
        avg_atr_20 = sum(
            (c.high - c.low) for c in candles[-20:]
        ) / 20
        vol_spike = (atr / avg_atr_20) if avg_atr_20 > 0 else 1.0
    else:
        vol_spike = 1.0

    raw["vol_spike"] = vol_spike

    # ── BB position ───────────────────────────────────────────────────────────
    band = bb["upper"] - bb["lower"]
    bb_pos = ((price - bb["lower"]) / band) if band > 0 else 0.5

    # ── trend strength ────────────────────────────────────────────────────────
    trend_strength = abs(ema9 - ema21) / price * 1000

    # ── EMA slope ─────────────────────────────────────────────────────────────
    ema_slope = ((ema9 - ema9_prev) / price * 100) if ema9_prev else 0.0

    # ── assemble feature vector ───────────────────────────────────────────────
    v = np.array([
        _clamp((rsi - 50) / 50),                          # rsi_norm
        float(np.sign(macd["macd"])),                     # macd_sign
        _clamp(macd["histogram"] / price * 1000),         # macd_hist_norm
        float(np.sign(ema9 - ema21)),                     # ema_cross
        _clamp(ema_slope),                                 # ema_slope_norm
        _clamp(bb_pos, 0.0, 1.0),                         # bb_pos
        _clamp(band / price * 100),                       # bb_width_norm
        _clamp(atr / price * 1000),                       # atr_norm
        _clamp(momentum / price * 100),                   # momentum_norm
        _clamp(velocity / 5),                             # velocity_norm
        _clamp(trend_strength / 5),                       # trend_strength
        _clamp(vol_spike - 1.0),                          # vol_spike (centred at 0)
    ], dtype=np.float32)

    return Features(vector=v, raw=raw, valid=True)


# ──────────────────────────────────────────────────────────────────────────────
# Logistic regression model
# ──────────────────────────────────────────────────────────────────────────────

# Hand-tuned initial weights for synthetic-index behaviour (cold-start)
HEURISTIC_WEIGHTS = np.array([
    -2.10,   # rsi_norm       : oversold → buy pressure
     1.80,   # macd_sign      : positive → bullish
     1.20,   # macd_hist_norm : histogram above zero → momentum up
     2.40,   # ema_cross      : fast > slow → uptrend
     1.00,   # ema_slope_norm : rising fast EMA → up
    -1.60,   # bb_pos         : low position → bounce likely
    -0.40,   # bb_width_norm  : wider band → less reliable mean-reversion
     0.00,   # atr_norm       : volatility neutral by default
     1.50,   # momentum_norm  : positive momentum → up
     1.20,   # velocity_norm  : price accelerating up → up
     0.90,   # trend_strength : stronger trend → trust the direction more
     0.50,   # vol_spike      : spike → momentum continuation on synthetic
], dtype=np.float32)

HEURISTIC_BIAS = -0.05   # slight sell bias (synthetic indices trend down slightly)


class DirectionModel:
    """
    Logistic regression with:
      • heuristic cold-start weights
      • online SGD updates as outcomes arrive
      • sklearn SGDClassifier as the trainable backend
    """

    def __init__(self, learning_rate: float = 0.05):
        self.lr          = learning_rate
        self.weights     = HEURISTIC_WEIGHTS.copy()
        self.bias        = HEURISTIC_BIAS
        self.trained     = False          # True once sklearn model is fit
        self._X_buf:  List[np.ndarray] = []
        self._y_buf:  List[int]        = []
        self._buf_max = 500             # train sklearn once we have this many

        # lazy-import sklearn so the module loads even without it installed
        try:
            from sklearn.linear_model import SGDClassifier
            self._clf = SGDClassifier(
                loss="log_loss",
                learning_rate="constant",
                eta0=learning_rate,
                max_iter=1,
                warm_start=True,
                random_state=42,
            )
        except ImportError:
            logger.warning("scikit-learn not installed — using heuristic weights only")
            self._clf = None

    # ── prediction ────────────────────────────────────────────────────────────
    def predict(self, features: Features) -> Optional[Dict]:
        """
        Returns:
          {
            "prob_up":    float,   # 0-1
            "direction":  "UP" | "DOWN",
            "confidence": float,   # 0-1
            "model":      "sklearn" | "heuristic"
          }
        or None if features are invalid.
        """
        if not features.valid:
            return None

        x = features.vector

        if self.trained and self._clf is not None:
            prob_up = float(self._clf.predict_proba([x])[0][1])
            src = "sklearn"
        else:
            z       = float(np.dot(self.weights, x) + self.bias)
            prob_up = 1.0 / (1.0 + math.exp(-z))
            src     = "heuristic"

        direction  = "UP"   if prob_up >= 0.5 else "DOWN"
        confidence = prob_up if prob_up >= 0.5 else 1.0 - prob_up

        return {
            "prob_up":    prob_up,
            "direction":  direction,
            "confidence": confidence,
            "model":      src,
        }

    # ── online update ─────────────────────────────────────────────────────────
    def update(self, features: Features, outcome: int):
        """
        Call this after observing actual market direction.
        outcome = 1 (price went UP) or 0 (price went DOWN).
        """
        if not features.valid or self._clf is None:
            return

        self._X_buf.append(features.vector)
        self._y_buf.append(outcome)

        # partial-fit every 10 new samples
        if len(self._X_buf) % 10 == 0:
            X = np.array(self._X_buf[-200:])
            y = np.array(self._y_buf[-200:])
            try:
                self._clf.partial_fit(X, y, classes=[0, 1])
                self.trained = True
                logger.debug("Model updated — %d training samples", len(self._X_buf))
            except Exception as exc:
                logger.warning("Model update failed: %s", exc)

        # trim buffer
        if len(self._X_buf) > self._buf_max:
            self._X_buf = self._X_buf[-self._buf_max:]
            self._y_buf = self._y_buf[-self._buf_max:]

    # ── bulk training ─────────────────────────────────────────────────────────
    def train_batch(self, X: np.ndarray, y: np.ndarray):
        """Train on a prepared (X, y) dataset — for backtesting pre-training."""
        if self._clf is None:
            return
        if len(X) < 10:
            logger.warning("Not enough training samples (%d)", len(X))
            return
        self._clf.fit(X, y)
        self.trained = True
        logger.info("Batch training complete on %d samples", len(X))

    # ── serialisation ─────────────────────────────────────────────────────────
    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "weights":  self.weights,
                "bias":     self.bias,
                "trained":  self.trained,
                "X_buf":    self._X_buf,
                "y_buf":    self._y_buf,
                "clf":      self._clf,
            }, f)
        logger.info("Model saved to %s", path)

    def load(self, path: str):
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.weights  = d["weights"]
        self.bias     = d["bias"]
        self.trained  = d["trained"]
        self._X_buf   = d["X_buf"]
        self._y_buf   = d["y_buf"]
        self._clf     = d["clf"]
        logger.info("Model loaded from %s (trained=%s)", path, self.trained)
