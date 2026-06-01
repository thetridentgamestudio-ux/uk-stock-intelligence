"""
Fetch today's news headlines from free RSS feeds, score with FinBERT,
and cache results for use by the prediction pipeline.

Run once per day, ideally at 8am after RNS announcements are published:
    python3 scripts/fetch_news_sentiment.py

Takes ~2-5 minutes depending on internet speed and CPU.
The FinBERT model (~450MB) is downloaded on first run only.
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

from backend.app.services.data_fetcher import FTSE_STOCKS
from backend.app.services.news_sentiment import fetch_and_score_news


def main():
    ticker_names = {
        ticker: info["name"] if isinstance(info, dict) else info[0]
        for ticker, info in FTSE_STOCKS.items()
    }

    logger.info("Fetching news sentiment for %d stocks…", len(ticker_names))
    results = fetch_and_score_news(ticker_names)

    with_news = {k: v for k, v in results.items() if v.get("count", 0) > 0}
    logger.info("Done — %d stocks with news today", len(with_news))

    if with_news:
        logger.info("Top stories:")
        for ticker, entry in sorted(with_news.items(),
                                    key=lambda x: abs(x[1]["score"]), reverse=True)[:10]:
            name = ticker_names.get(ticker, ticker)
            logger.info("  %-10s %-35s %+.2f (%s) — %s",
                        ticker, name[:35], entry["score"], entry["label"],
                        entry["headlines"][0][:60] if entry["headlines"] else "")


if __name__ == "__main__":
    main()
