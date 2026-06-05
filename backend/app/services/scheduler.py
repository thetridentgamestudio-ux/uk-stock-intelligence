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

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

_scheduler = AsyncIOScheduler(timezone="Europe/London")


# ── 1st of month, 06:00 — Monthly model retrain ──────────────────────────────

async def _monthly_retrain() -> None:
    """
    Retrain XGBoost + LightGBM ensemble with recency weighting.
    Runs at 06:00 on the 1st of each month — before the market opens.

    Spawned as subprocess to avoid joblib multiprocessing conflicting
    with the asyncio event loop (same issue as FinBERT/PyTorch segfault).
    After subprocess completes, clears the cached model bundle so the
    server picks up the new model on next prediction request.
    """
    import subprocess
    import os

    script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        ))),
        "scripts", "monthly_retrain.py"
    )
    python = "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
    cwd = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )))

    logger.info("Scheduler: spawning monthly retrain as subprocess…")
    try:
        result = subprocess.run(
            [python, script],
            capture_output=True,
            text=True,
            timeout=600,   # 10 minute timeout
            cwd=cwd,
        )
        if result.returncode == 0:
            logger.info("Scheduler: monthly retrain subprocess completed")
            # Clear cached bundle — server picks up new model on next request
            import backend.app.services.predictor as _pred
            _pred._bundle = None
            logger.info("Scheduler: model bundle cleared — new model loaded on next prediction")
        else:
            logger.error("Scheduler: retrain subprocess failed (rc=%d): %s",
                         result.returncode, result.stderr[-500:] if result.stderr else "")
    except subprocess.TimeoutExpired:
        logger.error("Scheduler: monthly retrain timed out after 10 minutes")
    except Exception as exc:
        logger.error("Scheduler: monthly retrain error: %s", exc)


# ── 08:00 — Morning news sentiment ────────────────────────────────────────────

async def _morning_news() -> None:
    """
    Fetch RSS headlines and score with FinBERT.
    Runs at 08:00 after RNS filings are published (RNS deadline is 07:00).

    IMPORTANT: FinBERT (PyTorch) MUST run in a separate process.
    Running PyTorch inside an async uvicorn process causes Segmentation fault: 11
    due to OpenMP/MKL workers conflicting with asyncio's event loop.
    We spawn the standalone script as a subprocess instead.
    """
    import subprocess
    import os

    script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        ))),
        "scripts", "fetch_news_sentiment.py"
    )
    python = "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"

    logger.info("Scheduler: spawning news sentiment script as subprocess…")
    try:
        result = subprocess.run(
            [python, script],
            capture_output=True,
            text=True,
            timeout=300,   # 5 minute timeout
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            ))),
        )
        if result.returncode == 0:
            logger.info("Scheduler: news sentiment subprocess completed successfully")
        else:
            logger.error("Scheduler: news subprocess failed (rc=%d): %s",
                         result.returncode, result.stderr[-500:] if result.stderr else "")
    except subprocess.TimeoutExpired:
        logger.error("Scheduler: news sentiment subprocess timed out after 5 minutes")
    except Exception as exc:
        logger.error("Scheduler: news sentiment subprocess error: %s", exc)


# ── 17:15 — Full daily pipeline ───────────────────────────────────────────────

async def _full_daily_pipeline() -> None:
    """
    Full pipeline after LSE close — spawned as subprocess.

    Reason: run_predictions() loads joblib/XGBoost models which use
    multiprocessing internally. Running these inside the asyncio event
    loop risks the same segfault seen with FinBERT. Subprocess isolation
    is the safest pattern for any heavy ML work.
    """
    import subprocess
    import os

    script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        ))),
        "scripts", "run_daily_pipeline.py"
    )
    python = "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
    cwd = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )))

    logger.info("Scheduler: spawning daily pipeline as subprocess…")
    try:
        result = subprocess.run(
            [python, script],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=cwd,
        )
        if result.returncode == 0:
            logger.info("Scheduler: daily pipeline subprocess completed")
            # Log last few lines of output (top picks)
            output_lines = result.stdout.strip().split("\n")
            for line in output_lines[-12:]:
                if line.strip():
                    logger.info("  %s", line)
        else:
            logger.error("Scheduler: pipeline subprocess failed (rc=%d): %s",
                         result.returncode, result.stderr[-500:] if result.stderr else "")
    except subprocess.TimeoutExpired:
        logger.error("Scheduler: daily pipeline timed out after 10 minutes")
    except Exception as exc:
        logger.error("Scheduler: daily pipeline error: %s", exc)


# ── Lifespan hook — registers jobs when FastAPI starts ────────────────────────

@asynccontextmanager
async def lifespan(app):
    _scheduler.add_job(
        _monthly_retrain,
        CronTrigger(day=1, hour=6, minute=0, timezone="Europe/London"),
        id="monthly_retrain",
        replace_existing=True,
    )
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
        "retrain @ 06:00 1st of month | "
        "news @ 08:00 Mon–Fri | "
        "pipeline @ 17:15 Mon–Fri (London time)"
    )
    yield
    _scheduler.shutdown()
    logger.info("Scheduler stopped")
