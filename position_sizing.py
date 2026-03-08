"""
Position Sizing — Expected Value & Kelly Criterion
Based on QR-PM-2026-0041 Doc 1, Equation (4)

Handwritten note: "NEVER full Kelly on 5min markets"
→ Fractional Kelly (25%) is default for short-horizon markets
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class TradeSignal:
    """A fully-evaluated trade opportunity"""
    market_id: str
    question: str
    direction: str          # "YES" or "NO"
    p_hat: float            # Agent's estimated probability
    p_market: float         # Current market price
    ev: float               # Expected value per dollar
    kelly_fraction: float   # Optimal fraction of bankroll
    recommended_size: float # In USDC
    confidence: str         # "HIGH" / "MEDIUM" / "LOW"
    edge_bps: int           # Edge in basis points


class PositionSizer:
    """
    Expected Value and Kelly position sizing.

    Eq. (4): EV = p̂·(1-p) - (1-p̂)·p = p̂ - p
    
    Kelly fraction: f* = (p̂ - p) / ((1-p) · p)
    (derived from Kelly criterion for binary bets)
    
    Note: Uses FRACTIONAL Kelly per handwritten annotation.
    Default kelly_multiplier = 0.25 for short-duration markets.
    """

    def __init__(
        self,
        bankroll: float = 1000.0,
        kelly_multiplier: float = 0.25,   # 25% Kelly — "NEVER full Kelly on 5min markets"
        min_edge: float = 0.04,           # Minimum 4% edge to trade (covers fees + slippage)
        max_position_pct: float = 0.20,   # Never more than 20% of bankroll on one trade
        min_position_usdc: float = 5.0,   # Minimum trade size
    ):
        self.bankroll = bankroll
        self.kelly_multiplier = kelly_multiplier
        self.min_edge = min_edge
        self.max_position_pct = max_position_pct
        self.min_position_usdc = min_position_usdc

    def expected_value(self, p_hat: float, p_market: float) -> float:
        """
        Eq. (4): EV = p̂·(1-p) - (1-p̂)·p = p̂ - p
        
        This is per-dollar EV on a YES bet at market price p.
        """
        return p_hat - p_market

    def kelly_fraction(self, p_hat: float, p_market: float) -> float:
        """
        Kelly optimal fraction for binary market:
        f* = edge / odds = (p̂ - p) / (1 - p)
        
        Then multiplied by kelly_multiplier (fractional Kelly).
        """
        eps = 1e-8
        edge = p_hat - p_market
        if edge <= 0:
            return 0.0
        # For YES bet: odds = (1 - p_market) / p_market, net_odds = 1/p_market - 1
        net_odds = (1 - p_market) / max(p_market, eps)
        f_star = edge / (1 - p_market + eps)  # Kelly formula for binary
        return min(f_star * self.kelly_multiplier, self.max_position_pct)

    def size_trade(
        self,
        market_id: str,
        question: str,
        p_hat: float,
        p_market: float,
    ) -> Optional[TradeSignal]:
        """
        Full position sizing pipeline.
        Returns None if edge is below threshold (no trade).
        """
        # Determine direction
        if p_hat > p_market:
            direction = "YES"
            ev = self.expected_value(p_hat, p_market)
            kelly = self.kelly_fraction(p_hat, p_market)
        else:
            # Flip: sell YES = buy NO
            direction = "NO"
            p_hat_no = 1 - p_hat
            p_market_no = 1 - p_market
            ev = self.expected_value(p_hat_no, p_market_no)
            kelly = self.kelly_fraction(p_hat_no, p_market_no)

        # Gate on minimum edge
        if abs(ev) < self.min_edge:
            return None

        # Size the position
        raw_size = self.bankroll * kelly
        size = max(min(raw_size, self.bankroll * self.max_position_pct), self.min_position_usdc)

        edge_bps = int(abs(ev) * 10000)
        confidence = "HIGH" if abs(ev) > 0.10 else ("MEDIUM" if abs(ev) > 0.06 else "LOW")

        return TradeSignal(
            market_id=market_id,
            question=question,
            direction=direction,
            p_hat=round(p_hat, 4),
            p_market=round(p_market, 4),
            ev=round(ev, 4),
            kelly_fraction=round(kelly, 4),
            recommended_size=round(size, 2),
            confidence=confidence,
            edge_bps=edge_bps,
        )

    def update_bankroll(self, pnl: float):
        """Update bankroll after trade resolution"""
        self.bankroll = max(self.bankroll + pnl, 0)
