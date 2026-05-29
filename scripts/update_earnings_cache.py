"""
Fetch earnings announcement dates for all FTSE 350 stocks from Yahoo Finance
and save to backend/app/data/earnings_cache.json.

Run this once, then monthly:
    python3 scripts/update_earnings_cache.py

Takes ~15-20 minutes for 350 stocks (0.3s delay per request to avoid rate-limiting).
Existing cache entries are preserved; only missing/stale tickers are updated.
"""
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

from backend.app.services.data_fetcher import FTSE_STOCKS
from backend.app.services.earnings_fetcher import (
    fetch_earnings_for_ticker,
    load_earnings_cache,
    save_earnings_cache,
)


def main():
    logger.info("Loading existing earnings cache…")
    cache = load_earnings_cache()

    tickers  = list(FTSE_STOCKS.keys())
    total    = len(tickers)
    updated  = 0
    skipped  = 0

    logger.info("Fetching earnings dates for %d stocks…", total)

    for i, ticker in enumerate(tickers, 1):
        # Skip recently-fetched entries (less than 20 dates suggests a refresh is worthwhile)
        existing = cache.get(ticker, [])
        if len(existing) >= 4:
            # Already have data — update only if first item is more than 90 days old
            from datetime import date, datetime
            try:
                most_recent = datetime.strptime(existing[0][:10], "%Y-%m-%d").date()
                days_old = (date.today() - most_recent).days
                if days_old < 90:
                    skipped += 1
                    if i % 50 == 0:
                        logger.info("  [%d/%d] skipping %s (cache fresh)", i, total, ticker)
                    continue
            except Exception:
                pass

        dates = fetch_earnings_for_ticker(ticker)
        if dates:
            cache[ticker] = dates
            updated += 1
            if i % 10 == 0 or len(dates) > 0:
                logger.info("  [%d/%d] %s → %d dates", i, total, ticker, len(dates))
        else:
            logger.debug("  [%d/%d] %s → no earnings data", i, total, ticker)

        time.sleep(0.35)   # polite rate limiting

        # Save checkpoint every 50 tickers
        if i % 50 == 0:
            save_earnings_cache(cache)
            logger.info("  Checkpoint saved (%d/%d)", i, total)

    save_earnings_cache(cache)
    logger.info("Done. Updated %d, skipped %d (cache fresh), total in cache: %d",
                updated, skipped, len(cache))


if __name__ == "__main__":
    main()
