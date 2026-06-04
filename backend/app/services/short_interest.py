"""
UK short interest data from FCA mandatory disclosures.

Free sources:
  - shortdata.co.uk (aggregates FCA short position announcements)
  - Short Tracker UK (shorttracker.co.uk)

Under UK Short Selling Regulation, any short position >0.5% of issued
share capital must be disclosed publicly with 1-day lag.

Features added per stock:
  short_interest_pct  — current short % of float (0-100)
  short_change_5d     — % change in short interest over 5 days (-100 to +100)
  short_recent_buy    — binary: short covering (buyback) signal in last 5 days

Academic backing: Asquith et al. (2005) + Bender et al. (2023) — high
short interest predicts mean reversion; short covering predicts rallies.
"""
import logging
from datetime import date, timedelta
import json
import os

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
_CACHE_FILE = os.path.join(_DATA_DIR, "short_interest_cache.json")

_cache: dict = {}
_cache_date: str = ""


def _load_cache() -> dict:
    """Load cached short interest data from disk."""
    global _cache, _cache_date
    today = str(date.today())
    if _cache and _cache_date == today:
        return _cache
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE) as f:
                _cache = json.load(f)
            _cache_date = today
        except Exception as exc:
            logger.warning("Short interest cache load failed: %s", exc)
            _cache = {}
    return _cache


def _save_cache(cache: dict) -> None:
    """Save short interest data to disk."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    try:
        with open(_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as exc:
        logger.warning("Short interest cache save failed: %s", exc)


def get_short_interest(ticker: str) -> dict | None:
    """
    Return cached short interest for a ticker.
    Format: {short_pct: float, count: int, change_5d: float, updated: str}
    """
    cache = _load_cache()
    entry = cache.get(ticker)
    if not entry:
        return None
    # Only return if fetched today
    if entry.get("updated") != str(date.today()):
        return None
    return entry


def short_interest_features(ticker: str) -> tuple[float | None, float | None]:
    """
    Return (short_interest_pct, short_change_5d) for a stock.
    Used as features in the model.
    """
    entry = get_short_interest(ticker)
    if not entry:
        return None, None
    return entry.get("short_pct"), entry.get("change_5d")


def short_interest_confidence_nudge(ticker: str, direction: str) -> tuple[float, str | None]:
    """
    Return (nudge_pp, flag) based on short interest signals.

    High short interest = bearish (mean reversion expected)
    Short covering (recent drop in shorts) = bullish (shorts covering)
    """
    entry = get_short_interest(ticker)
    if not entry:
        return 0.0, None

    short_pct = entry.get("short_pct", 0)
    change_5d = entry.get("change_5d", 0)
    count = entry.get("count", 0)

    if count < 2:  # Need multiple data points
        return 0.0, None

    nudge = 0.0
    flag = None

    # High short interest = bearish signal for BEARISH, headwind for BULLISH
    if short_pct > 10:  # >10% is elevated
        magnitude = min(1.5, (short_pct - 10) / 10)  # scale by how high
        if direction == "BEARISH":
            nudge += magnitude
            flag = f"📊 {short_pct:.1f}% short (elevated)"
        else:
            nudge -= magnitude
            flag = f"📊 {short_pct:.1f}% short (headwind)"

    # Short covering = bullish (shorts buying back)
    if change_5d < -15:  # >15% drop in shorts = covering
        nudge += min(1.0, abs(change_5d) / 30)
        if direction == "BULLISH":
            flag = "📈 Short covering detected"

    # New shorts building = bearish
    if change_5d > 20:  # >20% increase = new shorts
        nudge -= min(0.8, change_5d / 50)
        if direction == "BEARISH":
            flag = "📈 New short positions"

    return round(nudge, 2), flag


def fetch_and_cache_short_interest(ticker_list: list[str]) -> dict:
    """
    Fetch short interest data for a list of tickers.

    In production, this would call shortdata.co.uk API or web scrape.
    For now, returns empty cache — to be populated by scheduled fetcher.

    Returns updated cache dict.
    """
    cache = _load_cache()
    today = str(date.today())

    # TODO: Implement actual data fetch from shortdata.co.uk or shorttracker.co.uk
    # For now, placeholder — would use requests + BeautifulSoup to scrape:
    #   https://shortdata.co.uk/ (requires parsing HTML tables)
    #   https://shorttracker.co.uk/ (similar)
    #
    # Format per ticker:
    # cache[ticker] = {
    #     "short_pct": float (0-100),
    #     "change_5d": float (-100 to +100),
    #     "count": int (number of disclosed shorts),
    #     "updated": date_str
    # }

    _save_cache(cache)
    return cache


# ── Minimal stub for now (would expand with actual web scraper) ───────────────
def _stub_update_short_interest():
    """
    Placeholder. Real implementation would:
    1. GET shortdata.co.uk dashboard or API
    2. Parse HTML/JSON for ticker → short % mapping
    3. Track 5-day changes in short positions
    4. Cache results
    """
    logger.info("Short interest fetcher: stub (implement web scraper)")
    pass
