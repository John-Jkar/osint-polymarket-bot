"""
Test suite — validates all equations from QR-PM-2026-0041
Run with: python -m pytest tests/ -v
"""

import math
import pytest
import numpy as np
import sys
sys.path.insert(0, ".")

from core.bayesian_engine import BayesianUpdater, Signal
from core.lmsr_pricing import LMSRMarket, MarketState
from core.position_sizing import PositionSizer


# ─────────────────────────────────────────────────────
# Doc 1 — Bayesian Engine Tests
# ─────────────────────────────────────────────────────

class TestBayesianUpdater:

    def test_prior_preserved(self):
        """Prior is correctly initialized"""
        u = BayesianUpdater(prior=0.3)
        assert abs(u.posterior - 0.3) < 1e-6

    def test_strong_positive_signal_increases_posterior(self):
        """Doc 1 Eq 1&2: positive signal should increase P(H=True)"""
        u = BayesianUpdater(prior=0.5)
        p_before = u.posterior
        signal = Signal(
            source="test",
            description="positive signal",
            likelihood_true=0.9,
            likelihood_false=0.1,
        )
        p_after = u.update(signal)
        assert p_after > p_before

    def test_strong_negative_signal_decreases_posterior(self):
        """Negative signal should decrease P(H=True)"""
        u = BayesianUpdater(prior=0.5)
        signal = Signal(
            source="test",
            description="negative signal",
            likelihood_true=0.1,
            likelihood_false=0.9,
        )
        p_after = u.update(signal)
        assert p_after < 0.5

    def test_log_space_eq3_numerical_stability(self):
        """Doc 1 Eq 3: Log-space prevents underflow with many signals"""
        u = BayesianUpdater(prior=0.5)
        # 100 weak signals — should stay numerically stable
        for _ in range(100):
            signal = Signal("test", "weak", 0.6, 0.4)
            u.update(signal)
        assert 0 < u.posterior < 1
        assert not math.isnan(u.posterior)
        assert not math.isinf(u.posterior)

    def test_bayes_update_manual_verification(self):
        """
        Manual verification of Eq 1: P(H|D) = P(D|H)·P(H) / P(D)
        """
        prior = 0.5
        p_d_given_h = 0.8
        p_d_given_not_h = 0.2

        # Manual Bayes calculation
        p_d = p_d_given_h * prior + p_d_given_not_h * (1 - prior)
        expected_posterior = (p_d_given_h * prior) / p_d

        u = BayesianUpdater(prior=prior)
        signal = Signal("test", "signal", p_d_given_h, p_d_given_not_h)
        computed = u.update(signal)

        assert abs(computed - expected_posterior) < 1e-6, \
            f"Expected {expected_posterior:.6f}, got {computed:.6f}"

    def test_sequential_update_order_independence(self):
        """Eq 2: Sequential updates give same result regardless of order"""
        s1 = Signal("test", "s1", 0.8, 0.2)
        s2 = Signal("test", "s2", 0.7, 0.4)

        u1 = BayesianUpdater(prior=0.5)
        u1.update(s1); u1.update(s2)

        u2 = BayesianUpdater(prior=0.5)
        u2.update(s2); u2.update(s1)

        assert abs(u1.posterior - u2.posterior) < 1e-9, \
            "Sequential Bayesian updates should be order-independent"

    def test_posterior_bounded(self):
        """Posterior must always be in (0, 1)"""
        u = BayesianUpdater(prior=0.5)
        for _ in range(50):
            u.update(Signal("t", "s", 0.99, 0.01))  # Very strong signals
        assert 0 < u.posterior < 1


# ─────────────────────────────────────────────────────
# Doc 2 — LMSR Tests
# ─────────────────────────────────────────────────────

class TestLMSR:

    def test_prices_sum_to_one(self):
        """Doc 2: Σ pᵢ = 1 (critical property)"""
        market = LMSRMarket(n_outcomes=2, b=100.0)
        prices = market.prices()
        assert abs(prices.sum() - 1.0) < 1e-10

    def test_initial_prices_equal_for_zero_q(self):
        """With q=0, prices should be uniform (0.5 for binary)"""
        market = LMSRMarket(n_outcomes=2, b=100.0)
        prices = market.prices()
        assert abs(prices[0] - 0.5) < 1e-10
        assert abs(prices[1] - 0.5) < 1e-10

    def test_prices_bounded_open_interval(self):
        """Doc 2 critical property: pᵢ ∈ (0, 1) ∀i"""
        market = LMSRMarket(n_outcomes=2, b=100.0)
        market.q = np.array([1000.0, -1000.0])  # Extreme quantities
        prices = market.prices()
        assert all(0 < p < 1 for p in prices)

    def test_cost_function_eq1(self):
        """Doc 2 Eq 1: C(q) = b·ln(Σ e^(qᵢ/b))"""
        b = 100.0
        q = np.array([50.0, 30.0])
        market = LMSRMarket(n_outcomes=2, b=b)
        market.q = q

        expected = b * math.log(math.exp(q[0]/b) + math.exp(q[1]/b))
        computed = market.cost()
        assert abs(computed - expected) < 1e-6

    def test_max_loss_eq2(self):
        """Doc 2 Eq 2: L_max = b·ln(n)"""
        market = LMSRMarket(n_outcomes=2, b=100_000.0)
        expected = 100_000.0 * math.log(2)  # ≈ $69,315
        assert abs(market.max_market_maker_loss() - expected) < 0.01

    def test_binary_market_max_loss_approx_69315(self):
        """Doc 2: For n=2, b=100,000: L_max ≈ $69,315"""
        market = LMSRMarket(n_outcomes=2, b=100_000.0)
        assert abs(market.max_market_maker_loss() - 69_315) < 1

    def test_price_function_is_softmax(self):
        """Doc 2 Eq 3: price function = softmax(q/b)"""
        b = 100.0
        q = np.array([150.0, 80.0])
        market = LMSRMarket(n_outcomes=2, b=b)
        market.q = q

        # Manual softmax
        x = q / b
        exp_x = np.exp(x - x.max())
        expected_softmax = exp_x / exp_x.sum()

        computed = market.prices()
        assert np.allclose(computed, expected_softmax, atol=1e-10)

    def test_trade_cost_eq4(self):
        """Doc 2 Eq 4: Trade cost = C(q + δ) - C(q)"""
        market = LMSRMarket(n_outcomes=2, b=100.0)
        delta = 10.0

        cost_before = market.cost()
        q_new = market.q.copy()
        q_new[0] += delta
        cost_after = market.cost(q_new)

        expected_cost = cost_after - cost_before
        computed_cost = market.trade_cost(outcome_idx=0, delta=delta)

        assert abs(computed_cost - expected_cost) < 1e-10

    def test_buying_increases_price(self):
        """Buying YES shares should increase YES price"""
        market = LMSRMarket(n_outcomes=2, b=100.0)
        p_before = market.price(0)
        market.execute_trade(outcome_idx=0, delta=50.0)
        p_after = market.price(0)
        assert p_after > p_before

    def test_inefficiency_signal_correct_direction(self):
        """Edge calculation: p̂ > p_market → BUY signal"""
        market = LMSRMarket(n_outcomes=2, b=100.0)
        # Market at 50%
        signal = market.inefficiency_signal(p_hat=0.70, outcome_idx=0)
        assert signal["direction"] == "BUY"
        assert abs(signal["edge"] - 0.20) < 1e-4

        signal2 = market.inefficiency_signal(p_hat=0.30, outcome_idx=0)
        assert signal2["direction"] == "SELL"


# ─────────────────────────────────────────────────────
# Doc 1 — EV & Position Sizing Tests
# ─────────────────────────────────────────────────────

class TestPositionSizing:

    def test_ev_equation_4(self):
        """Doc 1 Eq 4: EV = p̂ - p"""
        sizer = PositionSizer(bankroll=1000.0)
        p_hat = 0.70
        p_market = 0.50
        ev = sizer.expected_value(p_hat, p_market)
        expected = p_hat - p_market  # = 0.20
        assert abs(ev - expected) < 1e-10

    def test_zero_ev_at_fair_price(self):
        """If p̂ == p_market, EV = 0"""
        sizer = PositionSizer(bankroll=1000.0)
        assert abs(sizer.expected_value(0.6, 0.6)) < 1e-10

    def test_negative_ev_below_market(self):
        """If p̂ < p_market, EV < 0 (sell signal)"""
        sizer = PositionSizer(bankroll=1000.0)
        assert sizer.expected_value(0.3, 0.5) < 0

    def test_no_trade_below_min_edge(self):
        """Small edge should not generate a trade"""
        sizer = PositionSizer(bankroll=1000.0, min_edge=0.04)
        trade = sizer.size_trade("mkt1", "Question?", p_hat=0.51, p_market=0.50)
        assert trade is None

    def test_trade_above_min_edge(self):
        """Sufficient edge should generate a trade"""
        sizer = PositionSizer(bankroll=1000.0, min_edge=0.04)
        trade = sizer.size_trade("mkt1", "Question?", p_hat=0.65, p_market=0.50)
        assert trade is not None
        assert trade.direction == "YES"
        assert trade.recommended_size > 0

    def test_fractional_kelly_respects_multiplier(self):
        """Position size should use fractional Kelly, not full Kelly"""
        sizer_full = PositionSizer(bankroll=1000.0, kelly_multiplier=1.0, min_edge=0.0)
        sizer_frac = PositionSizer(bankroll=1000.0, kelly_multiplier=0.25, min_edge=0.0)

        t_full = sizer_full.size_trade("m", "Q", 0.70, 0.50)
        t_frac = sizer_frac.size_trade("m", "Q", 0.70, 0.50)

        assert t_frac.recommended_size <= t_full.recommended_size

    def test_max_position_cap(self):
        """Position should never exceed max_position_pct of bankroll"""
        sizer = PositionSizer(bankroll=1000.0, max_position_pct=0.20)
        trade = sizer.size_trade("m", "Q", 0.99, 0.01)  # Massive edge
        assert trade.recommended_size <= 1000.0 * 0.20 + 1e-6

    def test_confidence_levels(self):
        """HIGH confidence for large edge, LOW for small edge"""
        sizer = PositionSizer(bankroll=1000.0, min_edge=0.0)
        trade_high = sizer.size_trade("m", "Q", 0.70, 0.50)  # 20% edge
        trade_low = sizer.size_trade("m", "Q", 0.55, 0.50)   # 5% edge
        assert trade_high.confidence == "HIGH"
        assert trade_low.confidence == "LOW"


# ─────────────────────────────────────────────────────
# Integration Test
# ─────────────────────────────────────────────────────

class TestIntegration:

    def test_full_pipeline(self):
        """
        End-to-end: signals → Bayesian update → LMSR check → position size
        """
        # 1. Initialize components
        updater = BayesianUpdater(prior=0.5)
        sizer = PositionSizer(bankroll=1000.0, kelly_multiplier=0.25, min_edge=0.04)

        # 2. Process signals
        signals = [
            Signal("gov_api",   "Bill passed committee", 0.85, 0.30, weight=1.5),
            Signal("major_news","Positive news coverage", 0.75, 0.40, weight=1.2),
            Signal("twitter_kol","KOL endorsement",      0.70, 0.45, weight=0.8),
        ]
        p_hat = updater.update_batch(signals)
        assert 0.5 < p_hat < 1.0, f"Posterior {p_hat} should be above 50% after positive signals"

        # 3. LMSR check
        market = LMSRMarket(n_outcomes=2, b=100.0)  # Market at 50%
        ineff = market.inefficiency_signal(p_hat, outcome_idx=0)
        assert ineff["direction"] == "BUY"
        assert ineff["edge"] > 0

        # 4. Position sizing
        trade = sizer.size_trade("test_market", "Will bill pass?", p_hat, p_market=0.50)
        assert trade is not None
        assert trade.direction == "YES"
        assert trade.ev > 0
        assert trade.recommended_size > 0

        print(f"\n✅ Integration test passed:")
        print(f"   Posterior: {p_hat:.4f}")
        print(f"   Edge: {ineff['edge']:+.4f}")
        print(f"   Trade: {trade.direction} ${trade.recommended_size:.2f} USDC")
        print(f"   EV: {trade.ev:+.4f} ({trade.edge_bps}bps) [{trade.confidence}]")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
