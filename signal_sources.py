"""
OSINT Signal Ingestion Layer
Pluggable signal sources that feed into the Bayesian engine.

Each source returns Signal objects with calibrated likelihoods.
Signal velocity (how fast news spreads) is tracked for edge timing.
"""

import time
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import re

from core.bayesian_engine import Signal

logger = logging.getLogger(__name__)


@dataclass
class RawSignal:
    """Raw ingested data before Bayesian conversion"""
    source: str
    content: str
    url: Optional[str]
    published_at: datetime
    sentiment_score: float   # -1.0 (negative) to +1.0 (positive)
    relevance_score: float   # 0.0 to 1.0
    reach: int               # followers / views / authority score
    signal_id: str = field(default_factory=lambda: hashlib.md5(
        str(time.time()).encode()).hexdigest()[:8])


class SignalCalibrator:
    """
    Converts raw signals into calibrated likelihoods for the Bayesian engine.
    
    Key insight: sentiment + relevance → P(D|H=True) and P(D|H=False)
    """

    # Calibration tables — these should be tuned from historical data
    # Format: (sentiment_threshold, p_true, p_false)
    CALIBRATION = {
        "high_positive":  (0.7,  0.80, 0.25),  # Strong positive → likely true
        "moderate_pos":   (0.3,  0.65, 0.40),
        "neutral":        (0.0,  0.50, 0.50),  # No info
        "moderate_neg":   (-0.3, 0.35, 0.60),
        "high_negative":  (-0.7, 0.20, 0.75),  # Strong negative → likely false
    }

    # Source credibility weights
    SOURCE_WEIGHTS = {
        "gov_api":        1.5,   # Official government data — high weight
        "major_news":     1.2,   # Reuters, AP, Bloomberg
        "polymarket_api": 1.0,   # Market price itself (reference)
        "twitter_kol":    0.8,   # Key opinion leaders
        "twitter_general":0.5,
        "reddit":         0.4,
        "unknown":        0.3,
    }

    def calibrate(self, raw: RawSignal, market_domain: str = "general") -> Signal:
        """
        Convert RawSignal → Bayesian Signal with calibrated likelihoods.
        """
        # Map sentiment to likelihood table
        s = raw.sentiment_score
        if s >= 0.7:
            _, p_true, p_false = self.CALIBRATION["high_positive"]
        elif s >= 0.3:
            _, p_true, p_false = self.CALIBRATION["moderate_pos"]
        elif s >= -0.3:
            _, p_true, p_false = self.CALIBRATION["neutral"]
        elif s >= -0.7:
            _, p_true, p_false = self.CALIBRATION["moderate_neg"]
        else:
            _, p_true, p_false = self.CALIBRATION["high_negative"]

        # Adjust by relevance (irrelevant signals → push toward 0.5)
        r = raw.relevance_score
        p_true = 0.5 + (p_true - 0.5) * r
        p_false = 0.5 + (p_false - 0.5) * r

        # Source credibility weight
        weight = self.SOURCE_WEIGHTS.get(raw.source, self.SOURCE_WEIGHTS["unknown"])

        # Reach amplification (viral signals count more)
        reach_factor = min(1.0 + math.log10(max(raw.reach, 1)) / 10, 1.5)
        weight *= reach_factor

        return Signal(
            source=raw.source,
            description=raw.content[:200],
            likelihood_true=p_true,
            likelihood_false=p_false,
            weight=weight,
            timestamp=raw.published_at,
        )


import math


# ─────────────────────────────────────────────────────
# Abstract base class for all signal sources
# ─────────────────────────────────────────────────────

class SignalSource(ABC):
    """Base class for OSINT signal sources"""

    def __init__(self, name: str):
        self.name = name
        self.calibrator = SignalCalibrator()
        self._last_fetch: Optional[datetime] = None
        self._seen_ids: set = set()

    @abstractmethod
    def fetch_raw(self, query: str, market_context: dict) -> List[RawSignal]:
        """Fetch raw signals from source. Must be implemented."""
        pass

    def fetch_signals(self, query: str, market_context: dict) -> List[Signal]:
        """Fetch and calibrate signals"""
        raw = self.fetch_raw(query, market_context)
        signals = []
        for r in raw:
            if r.signal_id not in self._seen_ids:
                self._seen_ids.add(r.signal_id)
                sig = self.calibrator.calibrate(r, market_context.get("domain", "general"))
                signals.append(sig)
        self._last_fetch = datetime.utcnow()
        return signals

    def deduplicate(self, signals: List[RawSignal]) -> List[RawSignal]:
        return [s for s in signals if s.signal_id not in self._seen_ids]


# ─────────────────────────────────────────────────────
# Concrete signal sources
# ─────────────────────────────────────────────────────

class PolymarketAPISource(SignalSource):
    """
    Reads current market prices from Polymarket API.
    Cross-market signals: if related markets move, update priors.
    """

    BASE_URL = "https://clob.polymarket.com"
    GAMMA_URL = "https://gamma-api.polymarket.com"

    def __init__(self):
        super().__init__("polymarket_api")

    def fetch_raw(self, query: str, market_context: dict) -> List[RawSignal]:
        """
        In production: calls Polymarket CLOB API.
        Returns price signals from correlated markets.
        """
        # Stub — returns simulated signal
        # In prod: requests.get(f"{self.GAMMA_URL}/markets?q={query}")
        logger.info(f"[Polymarket] Fetching related markets for: {query}")
        return []  # Real implementation uses actual API

    def get_market_price(self, condition_id: str) -> Optional[float]:
        """Get current YES price for a market condition"""
        # Prod: requests.get(f"{self.BASE_URL}/book?token_id={condition_id}")
        return None

    def get_all_markets(self, active_only: bool = True) -> List[dict]:
        """Fetch all active markets"""
        # Prod: paginated requests to /markets endpoint
        return []


class NewsAPISource(SignalSource):
    """
    Real-time news from NewsAPI, GNews, or RSS feeds.
    Sentiment scored via keyword analysis.
    """

    # Positive/negative keyword sets for quick sentiment scoring
    POSITIVE_KEYWORDS = {
        "wins", "victory", "confirmed", "passes", "approved", "succeeds",
        "leads", "ahead", "gains", "surges", "breakthrough", "deal", "signed"
    }
    NEGATIVE_KEYWORDS = {
        "loses", "defeated", "fails", "rejected", "drops", "crisis",
        "scandal", "arrested", "collapses", "delays", "vetoed", "denied"
    }

    def __init__(self, api_key: str = ""):
        super().__init__("major_news")
        self.api_key = api_key

    def fetch_raw(self, query: str, market_context: dict) -> List[RawSignal]:
        """
        Prod: requests.get("https://newsapi.org/v2/everything", params={
            "q": query, "sortBy": "publishedAt", "apiKey": self.api_key
        })
        """
        logger.info(f"[NewsAPI] Fetching news for: {query}")
        return []

    def score_sentiment(self, text: str) -> float:
        """Quick keyword-based sentiment score"""
        text_lower = text.lower()
        pos = sum(1 for k in self.POSITIVE_KEYWORDS if k in text_lower)
        neg = sum(1 for k in self.NEGATIVE_KEYWORDS if k in text_lower)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    def score_relevance(self, text: str, query: str) -> float:
        """Simple relevance scoring via keyword overlap"""
        query_words = set(re.findall(r'\w+', query.lower()))
        text_words = set(re.findall(r'\w+', text.lower()))
        overlap = query_words & text_words
        return min(len(overlap) / max(len(query_words), 1), 1.0)


class TwitterKOLSource(SignalSource):
    """
    Twitter/X Key Opinion Leaders monitoring.
    Tracks specific high-signal accounts per market domain.
    """

    # Domain-specific KOL lists (expand per market type)
    KOL_LISTS: Dict[str, List[str]] = {
        "politics_us":  ["@PredictIt", "@ElectoralVote", "@FiveThirtyEight"],
        "crypto":       ["@WhalePanda", "@BitMEXResearch", "@coindesk"],
        "geopolitics":  ["@RALee85", "@michaelkofman", "@IrenaGallina"],
        "sports":       ["@ESPNStatsInfo", "@EliasSports"],
        "general":      [],
    }

    def __init__(self, bearer_token: str = ""):
        super().__init__("twitter_kol")
        self.bearer_token = bearer_token

    def fetch_raw(self, query: str, market_context: dict) -> List[RawSignal]:
        """
        Prod: Twitter v2 filtered stream or recent search API
        GET https://api.twitter.com/2/tweets/search/recent?query={query}
        """
        domain = market_context.get("domain", "general")
        kols = self.KOL_LISTS.get(domain, self.KOL_LISTS["general"])
        logger.info(f"[Twitter KOL] Monitoring {len(kols)} accounts for: {query}")
        return []


class GovDataSource(SignalSource):
    """
    Government and official data sources.
    Highest-weight signals — hardest to fake or front-run.
    
    Examples:
    - Federal Register (new regulations)
    - Congress.gov (bill status)
    - SEC EDGAR (corporate filings)
    - BLS.gov (economic data releases)
    - USASpending.gov (government contracts)
    """

    ENDPOINTS = {
        "congress": "https://api.congress.gov/v3",
        "federal_register": "https://www.federalregister.gov/api/v1",
        "sec_edgar": "https://efts.sec.gov/LATEST/search-index",
        "bls": "https://api.bls.gov/publicAPI/v2/timeseries/data",
    }

    def __init__(self, api_key: str = ""):
        super().__init__("gov_api")
        self.api_key = api_key

    def fetch_raw(self, query: str, market_context: dict) -> List[RawSignal]:
        """Prod: route to correct government API based on market domain"""
        domain = market_context.get("domain", "general")
        logger.info(f"[GovData] Fetching official data for: {query} (domain: {domain})")
        return []
