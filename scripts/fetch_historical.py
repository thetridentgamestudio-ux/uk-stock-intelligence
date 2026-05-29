"""Backfill 2 years of daily OHLCV data for all FTSE stocks."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import date, timedelta

from sqlalchemy.orm import sessionmaker

from backend.app.database import engine
from backend.app.services.data_fetcher import FTSE_STOCKS, fetch_prices, save_prices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

LOOKBACK_YEARS = 2


def main() -> None:
    Session = sessionmaker(bind=engine)
    db = Session()

    end_date = date.today()
    start_date = end_date - timedelta(days=365 * LOOKBACK_YEARS)

    total = len(FTSE_STOCKS)
    for i, (ticker, (name, _, _)) in enumerate(FTSE_STOCKS.items(), 1):
        logger.info("[%d/%d] %s (%s)", i, total, ticker, name)
        df = fetch_prices(ticker, start_date, end_date)
        saved = save_prices(db, ticker, df)
        if df.empty:
            logger.warning("  No data returned")
        else:
            logger.info("  %d rows fetched, %d new saved", len(df), saved)
        time.sleep(0.4)  # polite to Yahoo Finance

    db.close()
    logger.info("Historical fetch complete.")


if __name__ == "__main__":
    main()
