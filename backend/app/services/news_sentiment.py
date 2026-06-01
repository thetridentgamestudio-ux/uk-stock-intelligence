"""
FinBERT-powered news sentiment for UK stocks.

Sources scraped (free, no API key):
  - LSE RNS (Regulatory News Service) via Investegate RSS
  - Reuters UK Business RSS
  - Motley Fool UK RSS

FinBERT (ProsusAI/finbert) runs locally on CPU — ~100 headlines/second.
First run downloads ~450MB model to ~/.cache/huggingface/hub/.

Sentiment cache: backend/app/data/news_sentiment_cache.json
  { "AZN.L": {"score": 0.72, "label": "positive", "count": 3,
               "headlines": ["..."], "updated": "2026-06-01"} }

Confidence modifier applied in predictor.py:
  positive sentiment + BULLISH  prediction → +2 pp boost
  negative sentiment + BEARISH  prediction → +2 pp boost
  negative sentiment + BULLISH  prediction → -2 pp penalty
  positive sentiment + BEARISH  prediction → -2 pp penalty
"""
import json
import logging
import os
import re
import time
from datetime import date, datetime
from functools import lru_cache

logger = logging.getLogger(__name__)

_DATA_DIR   = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
_CACHE_FILE = os.path.join(_DATA_DIR, "news_sentiment_cache.json")

# Free RSS feeds (require browser User-Agent header)
RSS_FEEDS = [
    # BBC Business — reliable, broad UK market coverage
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    # The Guardian UK Business
    "https://www.theguardian.com/uk/business/rss",
    # City A.M. — City of London focused
    "https://www.cityam.com/feed/",
    # This is Money — UK personal finance & markets
    "https://www.thisismoney.co.uk/money/news/index.rss",
]

_RSS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

_sentiment_pipeline = None
_cache: dict = {}
_cache_date: str = ""


def _load_finbert():
    """Load FinBERT pipeline once; cached for the process lifetime."""
    global _sentiment_pipeline
    if _sentiment_pipeline is not None:
        return _sentiment_pipeline
    try:
        from transformers import pipeline
        logger.info("Loading FinBERT model (first run downloads ~450MB)…")
        _sentiment_pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            device=-1,          # CPU; set to 0 for GPU
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT loaded ✓")
    except Exception as exc:
        logger.error("Failed to load FinBERT: %s", exc)
        _sentiment_pipeline = None
    return _sentiment_pipeline


def _load_cache() -> dict:
    """Load persisted sentiment cache from disk."""
    global _cache, _cache_date
    today = str(date.today())
    if _cache and _cache_date == today:
        return _cache
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE) as f:
                _cache = json.load(f)
            _cache_date = today
        except Exception:
            _cache = {}
    return _cache


def _save_cache(cache: dict) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def _ticker_mentions(text: str, ticker_names: dict[str, str]) -> list[str]:
    """
    Return list of tickers mentioned in text.
    Matches both the ticker stub (e.g. 'AZN') and the company name.
    """
    found = []
    text_lower = text.lower()
    for ticker, name in ticker_names.items():
        stub = ticker.replace(".L", "")
        if re.search(r"\b" + re.escape(stub) + r"\b", text, re.IGNORECASE):
            found.append(ticker)
        elif name and len(name) > 4 and name.lower() in text_lower:
            found.append(ticker)
    return list(set(found))


def fetch_and_score_news(ticker_names: dict[str, str]) -> dict:
    """
    Fetch RSS headlines, run FinBERT, return sentiment dict per ticker.

    Parameters
    ----------
    ticker_names : {ticker: company_name} — used to match headlines.

    Returns
    -------
    { ticker: {"score": float, "label": str, "count": int,
               "headlines": [str], "updated": str} }
    """
    try:
        import feedparser
    except ImportError:
        logger.warning("feedparser not installed — pip3 install feedparser")
        return {}

    pipe = _load_finbert()
    if pipe is None:
        return {}

    cache = _load_cache()
    today = str(date.today())

    # Collect all headlines from RSS feeds
    all_items: list[dict] = []
    import ssl, urllib.request
    _ssl_ctx = ssl.create_default_context()
    _ssl_ctx.check_hostname = False
    _ssl_ctx.verify_mode    = ssl.CERT_NONE

    for feed_url in RSS_FEEDS:
        try:
            # Fetch raw XML with browser headers (feedparser's built-in fetcher
            # doesn't send User-Agent, causing most UK news sites to return empty)
            req  = urllib.request.Request(feed_url, headers=_RSS_HEADERS)
            resp = urllib.request.urlopen(req, context=_ssl_ctx, timeout=10)
            xml  = resp.read()
            feed = feedparser.parse(xml)

            for entry in feed.entries[:60]:
                title   = getattr(entry, "title",   "")
                summary = getattr(entry, "summary", "")
                text    = f"{title}. {summary}"[:400]
                all_items.append({"text": text, "title": title})
            if feed.entries:
                logger.info("  %s: %d items", feed_url.split("/")[2], len(feed.entries))
        except Exception as exc:
            logger.debug("Feed %s failed: %s", feed_url, exc)

    if not all_items:
        logger.warning("No RSS items fetched — check internet connection")
        return cache

    logger.info("Fetched %d headlines from %d feeds", len(all_items), len(RSS_FEEDS))

    # Match headlines to tickers
    ticker_headlines: dict[str, list[str]] = {t: [] for t in ticker_names}
    for item in all_items:
        for ticker in _ticker_mentions(item["text"], ticker_names):
            ticker_headlines[ticker].append(item["title"])

    # Score with FinBERT
    updated_tickers = 0
    for ticker, headlines in ticker_headlines.items():
        if not headlines:
            continue
        try:
            results = pipe(headlines[:10])   # max 10 headlines per stock
            scores = []
            for r in results:
                label = r["label"].lower()
                if label == "positive":
                    scores.append(r["score"])
                elif label == "negative":
                    scores.append(-r["score"])
                # neutral → 0

            avg_score = float(sum(scores) / len(scores)) if scores else 0.0
            label     = "positive" if avg_score > 0.1 else ("negative" if avg_score < -0.1 else "neutral")

            cache[ticker] = {
                "score":     round(avg_score, 4),
                "label":     label,
                "count":     len(headlines),
                "headlines": headlines[:5],
                "updated":   today,
            }
            updated_tickers += 1
        except Exception as exc:
            logger.debug("FinBERT scoring failed for %s: %s", ticker, exc)

    _save_cache(cache)
    logger.info("News sentiment updated for %d stocks", updated_tickers)
    return cache


def get_sentiment(ticker: str) -> dict | None:
    """
    Return today's cached sentiment for a ticker, or None if unavailable.
    Stale entries (older than today) are treated as None.
    """
    cache = _load_cache()
    entry = cache.get(ticker)
    if not entry:
        return None
    # Only return if fetched today
    if entry.get("updated") != str(date.today()):
        return None
    return entry


def sentiment_confidence_nudge(ticker: str, direction: str) -> tuple[float, str | None]:
    """
    Return (nudge_pp, flag_string) for a prediction.

    nudge_pp : confidence adjustment in percentage points
    flag_string : human-readable label for frontend (or None)
    """
    entry = get_sentiment(ticker)
    if not entry:
        return 0.0, None

    score = entry["score"]
    label = entry["label"]
    count = entry["count"]

    # Only apply nudge when we have meaningful signal (≥2 headlines)
    if count < 2 or label == "neutral":
        return 0.0, None

    # Scale nudge by score strength (max ±3 pp)
    nudge_magnitude = min(3.0, abs(score) * 4.0)

    if label == "positive" and direction == "BULLISH":
        return +nudge_magnitude, f"📰 +ve news ({count} articles)"
    elif label == "negative" and direction == "BEARISH":
        return +nudge_magnitude, f"📰 -ve news ({count} articles)"
    elif label == "positive" and direction == "BEARISH":
        return -nudge_magnitude, f"📰 +ve news (counter-signal)"
    elif label == "negative" and direction == "BULLISH":
        return -nudge_magnitude, f"📰 -ve news (counter-signal)"

    return 0.0, None
