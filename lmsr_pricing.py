"""
Logarithmic Market Scoring Rule (LMSR)
Based on QR-PM-2026-0041 — Doc 2, Equations (1)-(4)
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


class LMSRMarket:
    """
    LMSR cost-function-based market maker.

    C(q) = b * ln( Σ e^(qi/b) )          — Eq. (1)
    p_i(q) = e^(qi/b) / Σ e^(qj/b)       — Eq. (3)  [softmax]
    L_max = b * ln(n)                      — Eq. (2)
    Trade cost = C(q + δ) - C(q)          — Eq. (4)
    """

    def __init__(self, n_outcomes: int = 2, b: float = 100.0):
        """
        Args:
            n_outcomes: Number of mutually exclusive outcomes (2 for binary)
            b: Liquidity parameter. Larger b = more liquidity, higher max loss.
               Polymarket typically uses b ~ 100-1000 USDC equivalent
        """
        assert n_outcomes >= 2
        assert b > 0
        self.n = n_outcomes
        self.b = b
        self.q = np.zeros(n_outcomes)  # outstanding quantity vector

    # ──────────────────────────────────────────────
    # Core LMSR functions (Doc 2 equations)
    # ──────────────────────────────────────────────

    def cost(self, q: np.ndarray = None) -> float:
        """
        Eq. (1): C(q) = b * ln( Σ e^(qi/b) )
        Uses log-sum-exp for numerical stability.
        """
        q = q if q is not None else self.q
        return self.b * np.logaddexp.reduce(q / self.b)

    def prices(self, q: np.ndarray = None) -> np.ndarray:
        """
        Eq. (3): p_i(q) = softmax(q / b)
        Returns probability vector summing to 1.
        """
        q = q if q is not None else self.q
        x = q / self.b
        x_shifted = x - x.max()  # numerical stability
        exp_x = np.exp(x_shifted)
        return exp_x / exp_x.sum()

    def price(self, outcome_idx: int) -> float:
        """Current market price for a specific outcome"""
        return float(self.prices()[outcome_idx])

    def max_market_maker_loss(self) -> float:
        """Eq. (2): L_max = b * ln(n)"""
        return self.b * math.log(self.n)

    def trade_cost(self, outcome_idx: int, delta: float) -> float:
        """
        Eq. (4): Cost = C(q_i + δ) - C(q)
        Positive delta = buying, negative = selling.
        Returns USDC cost (positive = pay, negative = receive).
        """
        q_new = self.q.copy()
        q_new[outcome_idx] += delta
        return self.cost(q_new) - self.cost(self.q)

    def execute_trade(self, outcome_idx: int, delta: float) -> Tuple[float, float]:
        """
        Execute a trade. Updates internal q vector.
        Returns (cost_usdc, new_price).
        """
        cost = self.trade_cost(outcome_idx, delta)
        self.q[outcome_idx] += delta
        new_price = self.price(outcome_idx)
        return cost, new_price

    # ──────────────────────────────────────────────
    # Inefficiency detection
    # ──────────────────────────────────────────────

    def inefficiency_signal(self, p_hat: float, outcome_idx: int = 0) -> dict:
        """
        Compare agent's estimated probability p̂ to LMSR market price.
        Entry condition from Doc 2 Section 4.
        
        Returns signal strength and direction.
        """
        p_market = self.price(outcome_idx)
        delta_p = p_hat - p_market  # Eq. (4) Doc 1: EV = p̂ - p

        return {
            "p_hat": round(p_hat, 4),
            "p_market": round(p_market, 4),
            "edge": round(delta_p, 4),
            "direction": "BUY" if delta_p > 0 else "SELL",
            "abs_edge": round(abs(delta_p), 4),
        }


@dataclass
class MarketState:
    """Snapshot of market state for a single Polymarket question"""
    market_id: str
    question: str
    outcome_yes_price: float   # Current YES price (0-1)
    outcome_no_price: float    # Current NO price (0-1)
    volume_usdc: float
    closes_at: str
    b_param: float = 100.0     # estimated liquidity param

    @property
    def implied_prob(self) -> float:
        """Market-implied probability of YES"""
        return self.outcome_yes_price

    def to_lmsr(self) -> LMSRMarket:
        """Initialize LMSR market from current prices"""
        market = LMSRMarket(n_outcomes=2, b=self.b_param)
        # Back-solve q from prices using softmax inverse
        # p_yes = e^(q_yes/b) / (e^(q_yes/b) + e^(q_no/b))
        # => q_yes - q_no = b * log(p_yes / p_no)
        eps = 1e-6
        p_yes = max(self.outcome_yes_price, eps)
        p_no = max(self.outcome_no_price, eps)
        log_ratio = math.log(p_yes / p_no)
        market.q[0] = self.b_param * log_ratio / 2
        market.q[1] = -self.b_param * log_ratio / 2
        return market
