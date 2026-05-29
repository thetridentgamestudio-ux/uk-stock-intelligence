import logging
from datetime import datetime, timezone

import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_news_headlines(ticker: str, max_items: int = 5) -> list[dict]:
    """
    Fetch recent news headlines for a ticker via Yahoo Finance.

    Returns list of dicts: {title, source, url, published_at}

    Note: For production, replace with direct LSE RNS feed scraping at
    https://www.londonstockexchange.com/exchange/news/market-news/market-news-home.html
    RNS announcements are the primary price movers for LSE stocks and
    are more reliable than general news aggregators.
    """
    try:
        news = yf.Ticker(ticker).news or []
        return [
            {
                "title": item.get("title", ""),
                "source": item.get("publisher", ""),
                "url": item.get("link", ""),
                "published_at": datetime.fromtimestamp(
                    item.get("providerPublishTime", 0), tz=timezone.utc
                ).isoformat(),
            }
            for item in news[:max_items]
            if item.get("title")
        ]
    except Exception as exc:
        logger.warning("Could not fetch news for %s: %s", ticker, exc)
        return []
