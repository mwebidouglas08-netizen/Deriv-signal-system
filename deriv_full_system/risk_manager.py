"""
risk_manager.py
─────────────────────────────────────────────────────────────────────────────
Standalone risk management module.
Calculates position sizing, stop-loss, take-profit based on ATR.
─────────────────────────────────────────────────────────────────────────────
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RiskResult:
    direction:   str
    entry:       float
    stop_loss:   float
    take_profit: float
    sl_distance: float
    tp_distance: float
    rr_ratio:    float
    risk_pct:    float
    atr:         float
    valid:       bool
    reason:      str = ""


class RiskManager:
    """
    ATR-based risk manager.

    stop_loss   = entry ± (ATR × atr_sl_mult)
    take_profit = entry ± (sl_distance × rr_ratio)
    """

    def __init__(
        self,
        atr_sl_mult: float = 1.5,
        rr_ratio:    float = 2.0,
        risk_pct:    float = 1.0,
        min_rr:      float = 1.8,     # reject signals below this RR
    ):
        self.atr_sl_mult = atr_sl_mult
        self.rr_ratio    = rr_ratio
        self.risk_pct    = risk_pct
        self.min_rr      = min_rr

    def calculate(
        self,
        direction: str,
        entry:     float,
        atr:       float,
    ) -> RiskResult:
        invalid = RiskResult(
            direction=direction, entry=entry, stop_loss=0, take_profit=0,
            sl_distance=0, tp_distance=0, rr_ratio=0, risk_pct=self.risk_pct,
            atr=atr, valid=False,
        )

        if atr <= 0:
            invalid.reason = "ATR is zero or negative"
            return invalid
        if entry <= 0:
            invalid.reason = "Entry price is zero or negative"
            return invalid

        sl_dist = atr * self.atr_sl_mult
        tp_dist = sl_dist * self.rr_ratio
        rr      = tp_dist / sl_dist if sl_dist > 0 else 0

        if direction == "BUY":
            sl = max(0.0, entry - sl_dist)
            tp = entry + tp_dist
        elif direction == "SELL":
            sl = entry + sl_dist
            tp = max(0.0, entry - tp_dist)
        else:
            invalid.reason = f"Unknown direction: {direction}"
            return invalid

        if rr < self.min_rr:
            invalid.reason = f"RR {rr:.2f} below minimum {self.min_rr}"
            return invalid

        return RiskResult(
            direction=direction,
            entry=entry,
            stop_loss=round(sl, 5),
            take_profit=round(tp, 5),
            sl_distance=round(sl_dist, 5),
            tp_distance=round(tp_dist, 5),
            rr_ratio=round(rr, 2),
            risk_pct=self.risk_pct,
            atr=round(atr, 5),
            valid=True,
        )

    def position_size(self, balance: float, entry: float, sl: float) -> float:
        """
        Suggested lot/stake size based on account balance and risk %.
        Returns the dollar amount to risk (for reference — not executed).
        """
        if balance <= 0 or entry <= 0:
            return 0.0
        risk_amount = balance * (self.risk_pct / 100)
        sl_dist     = abs(entry - sl)
        if sl_dist == 0:
            return 0.0
        return round(risk_amount, 2)
