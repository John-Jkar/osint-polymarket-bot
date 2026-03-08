# OSINT Prediction Market Bot
### Real-Time Bayesian Signal Processing — QR-PM-2026-0041

A fully quantitative bot that ingests OSINT signals, updates probability estimates via Bayesian inference, detects mispricings in LMSR prediction markets, and sizes positions using fractional Kelly criterion.

---

## How It Works

```
OSINT Signals → Bayesian Updater → LMSR Inefficiency Check → Position Sizing → Trade
```

| Step | Equation | File | Avg Latency |
|------|----------|------|-------------|
| Signal ingestion | — | `core/signal_sources.py` | 120ms |
| Bayesian posterior update | `log P(H\|D) = log P(H) + Σ log P(Dₖ\|H) - log Z` | `core/bayesian_engine.py` | 15ms |
| LMSR price comparison | `pᵢ(q) = softmax(q/b)` | `core/lmsr_pricing.py` | 3ms |
| Position sizing | `EV = p̂ - p`, fractional Kelly | `core/position_sizing.py` | <1ms |
| Order execution (CLOB) | — | `bot.py` | 690ms |
| **Total cycle** | | | **~828ms** |

---

## Use Case: US Senate Bill Passage Market

**Scenario:** Polymarket lists a market — *"Will the SAFE Banking Act pass the Senate by June 30?"* — currently priced at **42¢** (42% implied probability). You believe the market is slow to price in recent legislative signals.

### Step 1 — Signals arrive

Three OSINT sources fire within seconds of each other:

| Source | Signal | Sentiment |
|--------|--------|-----------|
| `gov_api` | Bill clears Senate Banking Committee 11–5 | Strong positive |
| `major_news` | Reuters: "Bipartisan support grows for cannabis banking reform" | Positive |
| `twitter_kol` | @SenateTracker: "Majority whip counting votes, floor vote expected next week" | Positive |

### Step 2 — Bayesian update

The engine processes each signal in log-space (numerically stable across hundreds of signals):

```
Prior:          P(pass) = 0.42   (market price)
After gov_api:  P(pass) = 0.61
After Reuters:  P(pass) = 0.71
After KOL:      P(pass) = 0.76
```

Your posterior estimate: **p̂ = 0.76**

### Step 3 — LMSR inefficiency detection

```python
market_price  = 0.42   # current Polymarket price
p_hat         = 0.76   # your Bayesian estimate
edge          = 0.34   # 34% — well above 4% minimum threshold
direction     = BUY YES
```

### Step 4 — Position sizing (25% fractional Kelly)

```
Bankroll:          $1,000 USDC
Kelly fraction:    f* = 0.34 / (1 - 0.42) ≈ 0.586
Fractional Kelly:  0.586 × 0.25 = 0.147  (14.7% of bankroll)
Position cap:      20% max → $146.70 USDC
EV per dollar:     +$0.34
Confidence:        HIGH (>10% edge)
```

**Trade fires:** BUY YES $146.70 USDC at 42¢

### Step 5 — Resolution

The Senate votes. Bill passes. Market settles at $1.00.

```
Entry:   $146.70 at 0.42
Exit:    $146.70 / 0.42 × 1.00 = $349.29
P&L:     +$202.59  (+138%)
```

---

## Installation

```bash
pip install numpy pytest
# Optional (for live trading):
# pip install py-clob-client tweepy newsapi-python
```

## Running

```bash
# Validate all equations (14 tests)
python -m pytest tests/test_equations.py -v

# Dry run (no real orders)
python main.py --dry-run --bankroll 1000 --kelly 0.25 --min-edge 0.04

# Live mode
python main.py --bankroll 1000
```

## Configuration

```python
config = BotConfig(
    bankroll_usdc      = 1000.0,
    kelly_multiplier   = 0.25,   # 25% fractional Kelly — never full Kelly on short markets
    min_edge           = 0.04,   # 4% minimum edge (covers ~2% fees + slippage)
    poll_interval_seconds = 30,
    max_open_positions = 5,
    min_volume_usdc    = 5000,   # skip illiquid markets
)
```

## Project Structure

```
prediction_market_bot/
├── core/
│   ├── bayesian_engine.py    # Sequential Bayesian updater (log-space)
│   ├── lmsr_pricing.py       # LMSR cost function, softmax prices, trade cost
│   ├── position_sizing.py    # EV calculation, fractional Kelly sizing
│   └── signal_sources.py     # OSINT ingestion: news, Twitter KOL, gov APIs
├── tests/
│   └── test_equations.py     # 14 equation validation tests
├── bot.py                    # Main orchestrator
├── main.py                   # CLI entry point
├── requirements.txt
└── README.md
```

## Plugging In Real Data

Three stubs to implement for live operation:

1. **`NewsAPISource.fetch_raw()`** — get a free key at [newsapi.org](https://newsapi.org)
2. **`PolymarketAPISource.get_all_markets()`** — public endpoint, no key needed  
   `GET https://gamma-api.polymarket.com/markets`
3. **`bot.py → execute_trade()`** — wire up `py-clob-client` with your wallet private key

## Signal Calibration

Edit `SignalCalibrator.CALIBRATION` in `signal_sources.py` to tune likelihoods per domain.  
The more historical signal → outcome data you accumulate, the tighter your calibration becomes — and the sharper your edge.

---

*Classification: RESTRICTED — Internal use only. Do not distribute.*  
*v2.3.1 | 2026-02-14*
