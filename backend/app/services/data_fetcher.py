import json
import logging
import os
from datetime import date
from functools import lru_cache

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

from ..models.db_models import StockPrice

logger = logging.getLogger(__name__)

_STOCKS_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "stocks.json",
)


@lru_cache(maxsize=1)
def _load_stocks() -> dict:
    """Load stock list from JSON. Cached after first read."""
    with open(_STOCKS_JSON) as f:
        return json.load(f)


@property
def FTSE_STOCKS() -> dict:
    return _load_stocks()


# Make FTSE_STOCKS importable as a plain dict reference
def get_stocks() -> dict:
    return _load_stocks()


# Keep FTSE_STOCKS as a module-level dict for backwards compatibility
import atexit as _atexit

try:
    FTSE_STOCKS = _load_stocks()
except FileNotFoundError:
    logger.warning("stocks.json not found — run scripts/update_stock_list.py first")
    FTSE_STOCKS = {}


def fetch_prices(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance. Returns empty DataFrame if unavailable."""
    try:
        df = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).date
        return df[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
    except Exception as exc:
        logger.error("Failed to fetch %s: %s", ticker, exc)
        return pd.DataFrame()


def save_prices(db: Session, ticker: str, df: pd.DataFrame) -> int:
    """Upsert price rows, skipping dates that already exist. Returns number saved."""
    if df.empty:
        return 0

    existing = {
        r.date
        for r in db.query(StockPrice.date).filter(StockPrice.ticker == ticker).all()
    }

    saved = 0
    for price_date, row in df.iterrows():
        if price_date in existing:
            continue
        db.add(StockPrice(
            ticker=ticker,
            date=price_date,
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
        ))
        saved += 1

    if saved:
        db.commit()
    return saved
