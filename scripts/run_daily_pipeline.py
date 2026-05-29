"""
Run after LSE closes (~16:35 UK time).

What it does:
  1. Fetches today's prices
  2. Evaluates yesterday's predictions against actual prices
  3. Generates new predictions for tomorrow
  4. Saves those predictions to DB
  5. Prints a summary report
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import date, timedelta

from sqlalchemy.orm import sessionmaker

from backend.app.database import engine
from backend.app.services.accuracy_checker import evaluate_pending_predictions
from backend.app.services.data_fetcher import FTSE_STOCKS, fetch_prices, save_prices
from backend.app.services.predictor import run_predictions, save_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    Session = sessionmaker(bind=engine)
    db = Session()
    today = date.today()

    # ── 1. Fetch today's prices ──────────────────────────────────────────────
    logger.info("Step 1/4 — Fetching today's prices (%s)...", today)
    for ticker in FTSE_STOCKS:
        # end=today+1 because yfinance's end date is exclusive — without +1, today is never fetched
        df = fetch_prices(ticker, today - timedelta(days=1), today + timedelta(days=1))
        saved = save_prices(db, ticker, df)
        if saved:
            logger.info("  %s: %d new row(s)", ticker, saved)
        time.sleep(0.3)

    # ── 2. Evaluate yesterday's predictions ──────────────────────────────────
    logger.info("Step 2/4 — Checking yesterday's predictions against actual prices...")
    result = evaluate_pending_predictions(db)
    if result["evaluated"] > 0:
        acc = result["correct"] / result["evaluated"] * 100
        logger.info(
            "  %d predictions evaluated — %d correct (%.0f%%)",
            result["evaluated"], result["correct"], acc
        )
    else:
        logger.info("  No predictions to evaluate yet (first run?)")

    # ── 3. Generate new predictions ──────────────────────────────────────────
    logger.info("Step 3/4 — Generating predictions for tomorrow...")
    predictions = run_predictions(db)

    if not predictions:
        logger.error("  No predictions generated — is the model trained?")
        db.close()
        return

    # ── 4. Save predictions ──────────────────────────────────────────────────
    logger.info("Step 4/4 — Saving predictions to database...")
    saved_count = save_predictions(db, predictions, today)
    logger.info("  Saved %d new predictions.", saved_count)

    # ── Summary report ───────────────────────────────────────────────────────
    gainers = [p for p in predictions if p["direction"] == "BULLISH"][:5]
    losers  = [p for p in predictions if p["direction"] == "BEARISH"][:5]

    print("\n" + "═" * 55)
    print(f"  UK STOCK INTELLIGENCE — {today}  (tomorrow's picks)")
    print("═" * 55)

    print("\n  TOP 5 PREDICTED GAINERS")
    print("  " + "─" * 50)
    for p in gainers:
        bar = "█" * int(p["confidence"] / 10)
        print(f"  {p['ticker']:<10} {p['name']:<28} {p['confidence']:>4.0f}%  {bar}")

    print("\n  TOP 5 PREDICTED LOSERS")
    print("  " + "─" * 50)
    for p in losers:
        bar = "█" * int(p["confidence"] / 10)
        print(f"  {p['ticker']:<10} {p['name']:<28} {p['confidence']:>4.0f}%  {bar}")

    # Show accuracy if we have history
    if result["evaluated"] > 0:
        acc = result["correct"] / result["evaluated"] * 100
        print(f"\n  Yesterday's accuracy: {result['correct']}/{result['evaluated']} correct ({acc:.0f}%)")

    print(f"\n  {len(predictions)} stocks analysed.")
    print("═" * 55 + "\n")

    db.close()


if __name__ == "__main__":
    main()
