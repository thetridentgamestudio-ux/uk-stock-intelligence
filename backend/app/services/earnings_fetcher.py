"""
Earnings date cache — fetches and stores upcoming + historical earnings dates
for all FTSE 350 stocks using Yahoo Finance (free, no API key required).

Dates are cached in  backend/app/data/earnings_cache.json.
Refresh by running:  python3 scripts/update_earnings_cache.py

Data format:
  {
    "AZN.L": ["2026-07-27", "2026-02-06", "2025-10-31", ...],
    ...
  }
Dates stored as ISO strings, sorted descending (most recent first).

Sources used (in order of reliability):
  1. yf.Ticker(t).earnings_dates   — historical quarterly dates (requires lxml)
  2. yf.Ticker(t).calendar         — next upcoming date
  3. yf.Ticker(t).info['earningsTimestamp']  — next date as Unix timestamp
"""
import json
import logging
import os
from datetime import date, datetime

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
_CACHE_FILE = os.path.join(_DATA_DIR, "earnings_cache.json")

# Module-level cache — loaded once per process
_earnings_cache: dict | None = None


def load_earnings_cache() -> dict:
    """
    Returns {ticker: [date_str, ...]} with dates sorted descending.
    Loads from disk once; subsequent calls return the in-memory copy.
    """
    global _earnings_cache
    if _earnings_cache is not None:
        return _earnings_cache

    if not os.path.exists(_CACHE_FILE):
        logger.info("Earnings cache not found — earnings features disabled until "
                    "scripts/update_earnings_cache.py is run.")
        _earnings_cache = {}
        return _earnings_cache

    try:
        with open(_CACHE_FILE) as f:
            data = json.load(f)
        # Filter empty entries
        _earnings_cache = {k: v for k, v in data.items() if v}
        logger.info("Loaded earnings cache: %d stocks with dates", len(_earnings_cache))
    except Exception as exc:
        logger.error("Failed to load earnings cache: %s", exc)
        _earnings_cache = {}

    return _earnings_cache


def get_earnings_dates(ticker: str) -> list | None:
    """
    Returns sorted list of datetime.date objects for a ticker.
    Returns None if not in cache (caller should skip earnings features).
    """
    cache = load_earnings_cache()
    raw   = cache.get(ticker)
    if not raw:
        return None
    dates = []
    for s in raw:
        try:
            dates.append(datetime.strptime(s[:10], "%Y-%m-%d").date())
        except ValueError:
            pass
    return sorted(dates) if dates else None


def fetch_earnings_for_ticker(ticker: str) -> list[str]:
    """
    Fetch earnings dates from Yahoo Finance for a single ticker.

    Strategy:
      1. Try earnings_dates (needs lxml — gives 8 quarters of history)
      2. Try calendar (next upcoming date)
      3. Try info['earningsTimestamp'] (next upcoming date as Unix ts)

    Returns list of unique ISO date strings, most recent first.
    Returns [] if no data found.
    """
    dates: set[str] = set()

    try:
        t = yf.Ticker(ticker)

        # ── Source 1: earnings_dates (historical, most valuable) ─────────────
        try:
            ed = t.earnings_dates
            if ed is not None and not ed.empty:
                for ts in ed.index:
                    try:
                        d = pd.Timestamp(ts).date()
                        dates.add(str(d))
                    except Exception:
                        pass
        except Exception:
            pass   # lxml may not be installed; fall through to source 2

        # ── Source 2: calendar (next upcoming date) ───────────────────────────
        try:
            cal = t.calendar
            if cal and "Earnings Date" in cal:
                for ts in cal["Earnings Date"]:
                    try:
                        d = str(ts)[:10]
                        if len(d) == 10:
                            dates.add(d)
                    except Exception:
                        pass
        except Exception:
            pass

        # ── Source 3: earningsTimestamp from info ─────────────────────────────
        if not dates:
            try:
                ts = t.fast_info.get("earningsTimestamp") or t.info.get("earningsTimestamp")
                if ts:
                    d = str(datetime.utcfromtimestamp(int(ts)).date())
                    dates.add(d)
            except Exception:
                pass

    except Exception as exc:
        logger.debug("Earnings fetch failed for %s: %s", ticker, exc)

    return sorted(dates, reverse=True) if dates else []


def save_earnings_cache(cache: dict) -> None:
    """Persist the cache dict to disk."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    non_empty = {k: v for k, v in cache.items() if v}
    with open(_CACHE_FILE, "w") as f:
        json.dump(non_empty, f, indent=2)
    logger.info("Saved earnings cache: %d stocks with dates → %s",
                len(non_empty), _CACHE_FILE)
