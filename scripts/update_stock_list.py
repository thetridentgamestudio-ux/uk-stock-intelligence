"""
Scrape FTSE 100 + FTSE 250 constituents from Wikipedia and save to
backend/app/data/stocks.json.

Run this whenever the index composition changes (quarterly rebalances).
"""
import json
import logging
import os
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "backend", "app", "data", "stocks.json",
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    )
}

SOURCES = {
    "FTSE100": "https://en.wikipedia.org/wiki/FTSE_100_Index",
    "FTSE250": "https://en.wikipedia.org/wiki/FTSE_250_Index",
}


def scrape_index(url: str, index_name: str) -> dict:
    logger.info("Fetching %s from Wikipedia...", index_name)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        logger.error("Request failed: %s", exc)
        return {}

    soup = BeautifulSoup(resp.content, "html.parser")
    stocks = {}

    for table in soup.find_all("table", class_="wikitable"):
        rows = table.find_all("tr")
        if not rows:
            continue

        hdrs = [
            c.get_text(strip=True).lower()
            for c in rows[0].find_all(["th", "td"])
        ]

        ticker_col  = next((i for i, h in enumerate(hdrs) if "ticker" in h or "epic" in h), None)
        company_col = next((i for i, h in enumerate(hdrs) if "company" in h), 0)
        sector_col  = next((i for i, h in enumerate(hdrs) if "sector" in h), None)

        if ticker_col is None:
            continue

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) <= ticker_col:
                continue

            ticker = cells[ticker_col].get_text(strip=True)
            # Skip blanks, headers repeated inside table, and suspiciously long strings
            if not ticker or len(ticker) > 6 or ticker.lower() in ("ticker", "epic"):
                continue

            company = (
                re.sub(r"\[.*?\]", "", cells[company_col].get_text(strip=True)).strip()
                if company_col < len(cells)
                else "Unknown"
            )
            sector = (
                re.sub(r"\[.*?\]", "", cells[sector_col].get_text(strip=True)).strip()
                if sector_col and sector_col < len(cells)
                else "Unknown"
            )

            stocks[ticker + ".L"] = {
                "name": company,
                "sector": sector,
                "index": index_name,
            }

        if stocks:
            break  # found the constituents table — stop searching

    logger.info("  → %d stocks found", len(stocks))
    return stocks


def main() -> None:
    all_stocks: dict = {}

    for index_name, url in SOURCES.items():
        batch = scrape_index(url, index_name)
        all_stocks.update(batch)
        time.sleep(1)  # polite pause between Wikipedia requests

    if not all_stocks:
        logger.error("No stocks scraped — aborting.")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_stocks, f, indent=2, sort_keys=True)

    logger.info("Saved %d stocks to %s", len(all_stocks), OUTPUT_PATH)

    # Also seed the database
    from sqlalchemy.orm import sessionmaker
    from backend.app.database import engine
    from backend.app.models.db_models import Stock

    Session = sessionmaker(bind=engine)
    db = Session()
    added = 0
    for ticker, info in all_stocks.items():
        if not db.query(Stock).filter(Stock.ticker == ticker).first():
            db.add(Stock(
                ticker=ticker,
                name=info["name"],
                sector=info["sector"],
                index_name=info["index"],
            ))
            added += 1
    db.commit()
    db.close()
    logger.info("Added %d new stocks to database.", added)


if __name__ == "__main__":
    main()
