"""
OSINT Prediction Market Bot — Main Orchestrator
QR-PM-2026-0041

Full pipeline:
  OSINT Signals → Bayesian Updater → LMSR Inefficiency Detection → Position Sizing → Trade
  
Latency budget (from Doc 1):
  Data ingestion:              120ms avg (340ms p99)
  Bayesian posterior compute:   15ms avg ( 28ms p99)
  LMSR price comparison:         3ms avg (  8ms p99)
  Order execution (CLOB):      690ms avg (1400ms p99)
  ─────────────────────────────────────────────────
  Total cycle:                 828ms avg (1776ms p99)
"""

import time
import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict

from core.bayesian_engine import BayesianUpdater, Signal
from core.lmsr_pricing import LMSRMarket, MarketState
from core.position_sizing import PositionSizer, TradeSignal
from core.signal_sources import (
    SignalSource, PolymarketAPISource, NewsAPISource,
    TwitterKOLSource, GovDataSource
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("PMBot")


@dataclass
class BotConfig:
    """Bot configuration"""
    bankroll_usdc: float = 1000.0
    kelly_multiplier: float = 0.25       # Fractional Kelly — never full Kelly on short markets
    min_edge: float = 0.04               # 4% minimum edge (covers ~2% fees + slippage)
    poll_interval_seconds: float = 30.0  # How often to scan markets
    max_open_positions: int = 5
    prior_probability: float = 0.5       # Default prior (adjust per market context)

    # Signal source API keys (set via env vars in production)
    news_api_key: str = ""
    twitter_bearer_token: str = ""
    gov_api_key: str = ""

    # Market filters
    min_volume_usdc: float = 5000.0      # Skip illiquid markets
    max_hours_to_close: float = 72.0     # Skip markets closing too soon (spreads too wide)
    min_hours_to_close: float = 1.0      # Skip markets closing too soon


@dataclass
class Position:
    """An open position"""
    trade_signal: TradeSignal
    opened_at: datetime
    status: str = "OPEN"  # OPEN / CLOSED / PENDING
    fill_price: Optional[float] = None
    pnl: Optional[float] = None


class PredictionMarketBot:
    """
    Main bot class. Orchestrates the full signal → trade pipeline.
    
    Architecture:
    ┌─────────────────────────────────────────────────────┐
    │  Signal Sources (OSINT)                             │
    │  NewsAPI / Twitter KOL / Gov APIs / Polymarket      │
    └──────────────────┬──────────────────────────────────┘
                       │ RawSignals
    ┌──────────────────▼──────────────────────────────────┐
    │  Bayesian Updater (log-space, Eq. 1-3)              │
    │  P(H|D₁,...,Dₜ) ∝ P(H) ∏ P(Dₖ|H)                 │
    └──────────────────┬──────────────────────────────────┘
                       │ p̂ (posterior probability)
    ┌──────────────────▼──────────────────────────────────┐
    │  LMSR Inefficiency Detection (Eq. 3 Doc 2)          │
    │  edge = p̂ - p_market                               │
    └──────────────────┬──────────────────────────────────┘
                       │ edge > min_edge?
    ┌──────────────────▼──────────────────────────────────┐
    │  Position Sizing (Eq. 4 + Fractional Kelly)         │
    │  EV = p̂ - p,  size = bankroll × f* × kelly_mult   │
    └──────────────────┬──────────────────────────────────┘
                       │ TradeSignal
    ┌──────────────────▼──────────────────────────────────┐
    │  Order Execution (Polymarket CLOB)                   │
    │  Limit order at p_hat ± slippage tolerance          │
    └─────────────────────────────────────────────────────┘
    """

    def __init__(self, config: BotConfig = None):
        self.config = config or BotConfig()
        self.sizer = PositionSizer(
            bankroll=self.config.bankroll_usdc,
            kelly_multiplier=self.config.kelly_multiplier,
            min_edge=self.config.min_edge,
        )

        # Initialize signal sources
        self.signal_sources: List[SignalSource] = [
            PolymarketAPISource(),
            NewsAPISource(api_key=self.config.news_api_key),
            TwitterKOLSource(bearer_token=self.config.twitter_bearer_token),
            GovDataSource(api_key=self.config.gov_api_key),
        ]

        # State
        self.open_positions: Dict[str, Position] = {}
        self.trade_history: List[Position] = []
        self.updaters: Dict[str, BayesianUpdater] = {}  # one per market
        self._running = False
        self._cycle_count = 0

    # ─────────────────────────────────────────────────────
    # Core pipeline
    # ─────────────────────────────────────────────────────

    def process_market(self, market: MarketState) -> Optional[TradeSignal]:
        """
        Full pipeline for a single market.
        Returns a TradeSignal if edge found, else None.
        
        Total cycle target: 828ms avg (Doc 1 latency table)
        """
        t_start = time.time()

        # Step 1: Get or create Bayesian updater for this market
        if market.market_id not in self.updaters:
            self.updaters[market.market_id] = BayesianUpdater(
                prior=self.config.prior_probability
            )
        updater = self.updaters[market.market_id]

        # Step 2: Fetch OSINT signals (target: 120ms)
        t_ingest = time.time()
        market_context = {
            "domain": self._infer_domain(market.question),
            "question": market.question,
            "closes_at": market.closes_at,
        }
        all_signals: List[Signal] = []
        for source in self.signal_sources:
            try:
                signals = source.fetch_signals(market.question, market_context)
                all_signals.extend(signals)
            except Exception as e:
                logger.warning(f"Signal source {source.name} failed: {e}")
        
        ingest_ms = (time.time() - t_ingest) * 1000
        logger.debug(f"[{market.market_id}] Ingested {len(all_signals)} signals in {ingest_ms:.0f}ms")

        # Step 3: Bayesian update (target: 15ms)
        t_bayes = time.time()
        if all_signals:
            p_hat = updater.update_batch(all_signals)
        else:
            p_hat = updater.posterior
        
        bayes_ms = (time.time() - t_bayes) * 1000
        logger.debug(f"[{market.market_id}] Posterior: {p_hat:.4f} in {bayes_ms:.0f}ms")

        # Step 4: LMSR inefficiency detection (target: 3ms)
        t_lmsr = time.time()
        lmsr = market.to_lmsr()
        signal = lmsr.inefficiency_signal(p_hat, outcome_idx=0)
        lmsr_ms = (time.time() - t_lmsr) * 1000

        logger.info(
            f"[{market.market_id}] p̂={p_hat:.3f} p_mkt={signal['p_market']:.3f} "
            f"edge={signal['edge']:+.3f} ({signal['direction']}) | "
            f"LMSR check: {lmsr_ms:.0f}ms"
        )

        # Step 5: Position sizing (gated on min edge)
        if abs(signal["edge"]) < self.config.min_edge:
            logger.debug(f"[{market.market_id}] Edge {signal['edge']:.3f} below threshold — no trade")
            total_ms = (time.time() - t_start) * 1000
            logger.debug(f"[{market.market_id}] Cycle complete in {total_ms:.0f}ms (no trade)")
            return None

        # Check position limits
        if len(self.open_positions) >= self.config.max_open_positions:
            logger.info(f"[{market.market_id}] Max positions ({self.config.max_open_positions}) reached — skipping")
            return None

        trade = self.sizer.size_trade(
            market_id=market.market_id,
            question=market.question,
            p_hat=p_hat,
            p_market=market.implied_prob,
        )

        total_ms = (time.time() - t_start) * 1000
        if trade:
            logger.info(
                f"✅ TRADE SIGNAL [{market.market_id}] | "
                f"{trade.direction} ${trade.recommended_size:.2f} USDC | "
                f"EV={trade.ev:+.3f} ({trade.edge_bps}bps) | "
                f"Confidence={trade.confidence} | "
                f"Cycle: {total_ms:.0f}ms"
            )

        return trade

    def execute_trade(self, trade: TradeSignal, market: MarketState) -> Optional[Position]:
        """
        Execute trade on Polymarket CLOB.
        Target latency: 690ms avg (Doc 1)
        
        In production: uses py-clob-client or direct API calls.
        """
        logger.info(f"[Execution] Sending order: {trade.direction} {trade.recommended_size} USDC on {trade.market_id}")
        
        # Prod implementation:
        # from py_clob_client.client import ClobClient
        # client = ClobClient(host=CLOB_URL, key=PRIVATE_KEY, chain_id=CHAIN_ID)
        # 
        # token_id = get_token_id(trade.market_id, trade.direction)
        # order = client.create_limit_order(
        #     token_id=token_id,
        #     price=trade.p_market,
        #     size=trade.recommended_size,
        #     side="BUY",
        # )
        # resp = client.post_order(order, OrderType.GTC)

        position = Position(
            trade_signal=trade,
            opened_at=datetime.utcnow(),
            fill_price=trade.p_market,  # Stub: assume fill at market
        )
        self.open_positions[trade.market_id] = position
        return position

    # ─────────────────────────────────────────────────────
    # Market scanning loop
    # ─────────────────────────────────────────────────────

    def scan_markets(self) -> List[MarketState]:
        """
        Fetch active markets from Polymarket API.
        Filters by volume, time-to-close, and existing positions.
        
        Prod: paginated calls to Gamma API /markets endpoint
        """
        # Stub — in prod this calls the real API
        logger.info("Scanning Polymarket for active markets...")
        return []

    def run_once(self) -> List[TradeSignal]:
        """Run one full scan-and-trade cycle"""
        self._cycle_count += 1
        logger.info(f"─── Cycle #{self._cycle_count} ───")

        markets = self.scan_markets()
        if not markets:
            logger.info("No markets returned from scanner")
            return []

        signals = []
        for market in markets:
            trade = self.process_market(market)
            if trade:
                position = self.execute_trade(trade, market)
                if position:
                    signals.append(trade)

        return signals

    async def run_loop(self):
        """Async main loop"""
        self._running = True
        logger.info(f"Bot started | Bankroll: ${self.config.bankroll_usdc:.2f} USDC | "
                    f"Kelly: {self.config.kelly_multiplier*100:.0f}% | "
                    f"Min edge: {self.config.min_edge*100:.1f}%")

        while self._running:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"Cycle error: {e}", exc_info=True)
            await asyncio.sleep(self.config.poll_interval_seconds)

    def stop(self):
        self._running = False
        logger.info("Bot stopped.")

    # ─────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────

    def _infer_domain(self, question: str) -> str:
        """Infer market domain from question text for signal routing"""
        q = question.lower()
        if any(w in q for w in ["president", "election", "senate", "congress", "vote", "poll"]):
            return "politics_us"
        if any(w in q for w in ["bitcoin", "eth", "crypto", "defi", "token"]):
            return "crypto"
        if any(w in q for w in ["war", "military", "nato", "ukraine", "russia", "china"]):
            return "geopolitics"
        if any(w in q for w in ["nba", "nfl", "soccer", "championship", "world cup"]):
            return "sports"
        return "general"

    def portfolio_summary(self) -> dict:
        """Current portfolio state"""
        return {
            "bankroll_usdc": round(self.sizer.bankroll, 2),
            "open_positions": len(self.open_positions),
            "total_trades": len(self.trade_history),
            "cycle_count": self._cycle_count,
            "positions": [
                {
                    "market_id": pid,
                    "direction": pos.trade_signal.direction,
                    "size": pos.trade_signal.recommended_size,
                    "ev": pos.trade_signal.ev,
                    "opened_at": pos.opened_at.isoformat(),
                }
                for pid, pos in self.open_positions.items()
            ]
        }
