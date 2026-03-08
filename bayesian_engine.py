"""
Bayesian Signal Processing Engine
Based on QR-PM-2026-0041 equations (1), (2), (3), (4)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class Signal:
    """A single OSINT signal with likelihood ratio"""
    source: str           # e.g. "twitter", "news", "gov_api"
    description: str
    # P(D | H=True) — how likely is this signal if outcome is TRUE
    likelihood_true: float
    # P(D | H=False) — how likely is this signal if outcome is FALSE
    likelihood_false: float
    weight: float = 1.0   # signal credibility weight
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def log_likelihood_ratio(self) -> float:
        """
        log P(D|H=True) - log P(D|H=False)
        Positive = evidence FOR the outcome
        Negative = evidence AGAINST
        """
        eps = 1e-10
        l_true = max(self.likelihood_true, eps)
        l_false = max(self.likelihood_false, eps)
        return self.weight * (math.log(l_true) - math.log(l_false))


class BayesianUpdater:
    """
    Sequential Bayesian Updater — Equations (1), (2), (3) from Doc 1
    
    Runs in log-space for numerical stability:
    log P(H|D) = log P(H) + Σ log P(Dk|H) - log Z
    """

    def __init__(self, prior: float = 0.5):
        """
        Args:
            prior: Initial P(H=True) before any signals
        """
        assert 0 < prior < 1, "Prior must be in (0, 1)"
        self.prior = prior
        self._log_odds = math.log(prior / (1 - prior))  # log-space prior
        self.signals: List[Signal] = []
        self.log_posterior_history: List[float] = []
        self._log_posterior_true = math.log(prior)   # log P(H=True)
        self._log_posterior_false = math.log(1 - prior)  # log P(H=False)

    def update(self, signal: Signal) -> float:
        """
        Ingest one signal, update posterior (Eq. 2 & 3).
        Returns updated P(H=True).
        """
        self.signals.append(signal)

        # Eq. 3: log P(H|D) = log P(H) + Σ log P(Dk|H) - log Z
        eps = 1e-10
        self._log_posterior_true += math.log(max(signal.likelihood_true, eps)) * signal.weight
        self._log_posterior_false += math.log(max(signal.likelihood_false, eps)) * signal.weight

        p = self.posterior
        self.log_posterior_history.append(p)
        return p

    def update_batch(self, signals: List[Signal]) -> float:
        """Process multiple signals at once"""
        for s in signals:
            self.update(s)
        return self.posterior

    @property
    def posterior(self) -> float:
        """
        P(H=True | all signals so far)
        Normalized via log-sum-exp for numerical stability
        """
        log_Z = np.logaddexp(self._log_posterior_true, self._log_posterior_false)
        return math.exp(self._log_posterior_true - log_Z)

    @property
    def log_posterior(self) -> float:
        """log P(H=True | D)"""
        return math.log(max(self.posterior, 1e-10))

    def reset(self, prior: Optional[float] = None):
        """Reset to prior (new market)"""
        p = prior or self.prior
        self._log_posterior_true = math.log(p)
        self._log_posterior_false = math.log(1 - p)
        self.signals.clear()
        self.log_posterior_history.clear()

    def summary(self) -> dict:
        return {
            "prior": self.prior,
            "posterior": round(self.posterior, 4),
            "n_signals": len(self.signals),
            "signal_sources": [s.source for s in self.signals],
        }
