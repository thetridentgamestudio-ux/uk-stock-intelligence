"""
APScheduler jobs — run inside the FastAPI process (Europe/London timezone).

Schedule:
  08:00 Mon–Fri  → fetch_news_sentiment  (FinBERT scores RSS headlines before open)
  17:15 Mon–Fri  → full_daily_pipeline   (prices + accuracy + predictions after close)

Both jobs also run on-demand from the CLI scripts:
  python3 scripts/fetch_news_sentiment.py
  python3 scripts/run_daily_pipeline.py
"""
import logging
from contextlib import asynccontextmanager
from datetime import date, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from ..database import SessionLocal
from ..services.data_fetcher import FTSE_STOCKS, fetch_prices, save_prices

logger = logging.getLogger(__name__)

_scheduler = AsyncIOScheduler(timezone="Europe/London")


# ── 08:00 — Morning news sentiment ────────────────────────────────────────────

async def _morning_news() -> None:
    """
    Fetch RSS headlines and score with FinBERT.
    Runs at 08:00 after RNS filings are published (RNS deadline is 07:00).
    """
    logger.info("Scheduler: starting morning news sentiment fetch…")
    try:
        from ..services.data_fetcher import FTSE_STOCKS
        from ..services.news_sentiment import fetch_and_score_news

        ticker_names = {
            ticker: info["name"] if isinstance(info, dict) else info[0]
            for ticker, info in FTSE_STOCKS.items()
        }
        results = fetch_and_score_news(ticker_names)
        with_news = sum(1 for v in results.values() if v.get("count", 0) > 0)
        logger.info("Scheduler: news sentiment done — %d stocks with coverage", with_news)
    except Exception as exc:
        logger.error("Scheduler: morning news failed: %s", exc)


# ── 17:15 — Full daily pipeline ───────────────────────────────────────────────

async def _full_daily_pipeline() -> None:
    """
    Full pipeline after LSE close:
      1. Fetch today's prices (end=tomorrow so today is included)
      2. Evaluate yesterday's predictions → accuracy tracking
      3. Generate new predictions for tomorrow
      4. Save predictions to DB
    """
    logger.info("Scheduler: starting daily pipeline…")
    today = date.today()
    db    = SessionLocal()

    try:
        # ── 1. Fetch prices ──────────────────────────────────────────────────
        fetched = 0
        for ticker in FTSE_STOCKS:
            df    = fetch_prices(ticker, today - timedelta(days=1), today + timedelta(days=1))
            saved = save_prices(db, ticker, df)
            if saved:
                fetched += saved
        logger.info("Scheduler: fetched %d new price rows", fetched)

        # ── 2. Evaluate predictions ──────────────────────────────────────────
        from ..services.accuracy_checker import evaluate_pending_predictions
        result = evaluate_pending_predictions(db)
        if result["evaluated"]:
            acc = result["correct"] / result["evaluated"] * 100
            logger.info("Scheduler: evaluated %d predictions — %.0f%% correct",
                        result["evaluated"], acc)

        # ── 3. Generate + save predictions ──────────────────────────────────
        from ..services.predictor import run_predictions, save_predictions
        predictions = run_predictions(db)
        if predictions:
            saved = save_predictions(db, predictions, today)
            logger.info("Scheduler: saved %d new predictions for %s",
                        saved, today + timedelta(days=1))
            # Log top 3 picks
            gainers = [p for p in predictions if p["direction"] == "BULLISH"][:3]
            losers  = [p for p in predictions if p["direction"] == "BEARISH"][:3]
            for p in gainers:
                logger.info("  ▲ %s  %s  %.0f%%", p["ticker"], p["name"], p["confidence"])
            for p in losers:
                logger.info("  ▼ %s  %s  %.0f%%", p["ticker"], p["name"], p["confidence"])
        else:
            logger.warning("Scheduler: no predictions generated — model trained?")

    except Exception as exc:
        logger.error("Scheduler: daily pipeline error: %s", exc)
    finally:
        db.close()


# ── Lifespan hook — registers jobs when FastAPI starts ────────────────────────

@asynccontextmanager
async def lifespan(app):
    _scheduler.add_job(
        _morning_news,
        CronTrigger(day_of_week="mon-fri", hour=8, minute=0, timezone="Europe/London"),
        id="morning_news",
        replace_existing=True,
    )
    _scheduler.add_job(
        _full_daily_pipeline,
        CronTrigger(day_of_week="mon-fri", hour=17, minute=15, timezone="Europe/London"),
        id="daily_pipeline",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info(
        "Scheduler started — "
        "news @ 08:00 Mon–Fri | pipeline @ 17:15 Mon–Fri (London time)"
    )
    yield
    _scheduler.shutdown()
    logger.info("Scheduler stopped")
