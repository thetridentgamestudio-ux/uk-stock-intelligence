import logging
from contextlib import asynccontextmanager
from datetime import date, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from ..database import SessionLocal
from ..services.data_fetcher import FTSE_STOCKS, fetch_prices, save_prices

logger = logging.getLogger(__name__)

_scheduler = AsyncIOScheduler(timezone="Europe/London")


async def _daily_pipeline() -> None:
    """Fetch latest prices after LSE close (runs at 17:15 Mon–Fri)."""
    today = date.today()
    yesterday = today - timedelta(days=1)
    db = SessionLocal()
    try:
        for ticker in FTSE_STOCKS:
            df = fetch_prices(ticker, yesterday, today)
            saved = save_prices(db, ticker, df)
            if saved:
                logger.info("Daily pipeline: saved %d rows for %s", saved, ticker)
    except Exception as exc:
        logger.error("Daily pipeline error: %s", exc)
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app):
    _scheduler.add_job(
        _daily_pipeline,
        CronTrigger(day_of_week="mon-fri", hour=17, minute=15),
        id="daily_pipeline",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info("Scheduler started — daily pipeline runs at 17:15 Mon–Fri (London time)")
    yield
    _scheduler.shutdown()
    logger.info("Scheduler stopped")
