"""
OSINT Prediction Market Bot — Entry Point
Usage: python main.py [--dry-run] [--bankroll 1000]
"""

import asyncio
import argparse
import logging
from bot import PredictionMarketBot, BotConfig

logger = logging.getLogger("main")


def parse_args():
    p = argparse.ArgumentParser(description="OSINT Prediction Market Bot")
    p.add_argument("--bankroll", type=float, default=1000.0, help="Starting bankroll in USDC")
    p.add_argument("--kelly", type=float, default=0.25, help="Kelly multiplier (0.25 = 25% fractional Kelly)")
    p.add_argument("--min-edge", type=float, default=0.04, help="Minimum edge to trade (default: 4%)")
    p.add_argument("--interval", type=float, default=30.0, help="Poll interval in seconds")
    p.add_argument("--dry-run", action="store_true", help="Simulate only, no real orders")
    p.add_argument("--test", action="store_true", help="Run equation tests and exit")
    return p.parse_args()


def run_tests():
    """Quick equation validation"""
    import pytest, sys
    sys.exit(pytest.main(["tests/", "-v", "--tb=short"]))


async def main():
    args = parse_args()

    if args.test:
        run_tests()
        return

    config = BotConfig(
        bankroll_usdc=args.bankroll,
        kelly_multiplier=args.kelly,
        min_edge=args.min_edge,
        poll_interval_seconds=args.interval,
    )

    bot = PredictionMarketBot(config=config)

    if args.dry_run:
        logger.info("🔵 DRY RUN MODE — No real orders will be placed")

    try:
        await bot.run_loop()
    except KeyboardInterrupt:
        bot.stop()
        summary = bot.portfolio_summary()
        logger.info(f"Final state: {summary}")


if __name__ == "__main__":
    asyncio.run(main())
