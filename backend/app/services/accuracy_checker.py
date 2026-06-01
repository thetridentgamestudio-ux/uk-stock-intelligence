import logging
from datetime import date

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def evaluate_pending_predictions(db: Session) -> dict:
    """
    For any prediction whose target_date has passed but was_correct is NULL,
    look up the actual price movement and mark it correct or incorrect.
    """
    pending = db.execute(
        text(
            "SELECT id, ticker, prediction_date, target_date, direction "
            "FROM predictions "
            "WHERE was_correct IS NULL AND target_date <= :today"
        ),
        {"today": date.today()},
    ).fetchall()

    if not pending:
        logger.info("No pending predictions to evaluate.")
        return {"evaluated": 0, "correct": 0}

    evaluated = correct = 0

    for pred_id, ticker, prediction_date, target_date, direction in pending:
        # Price the prediction was based on (close on prediction day)
        price_before = db.execute(
            text(
                "SELECT close FROM stock_prices "
                "WHERE ticker = :t AND date = :d"
            ),
            {"t": ticker, "d": prediction_date},
        ).scalar()

        # Price on target_date; fall back to first available price after that date
        # (handles rare cases where market was closed on the exact target_date)
        price_after = db.execute(
            text(
                "SELECT close FROM stock_prices "
                "WHERE ticker = :t AND date = :d"
            ),
            {"t": ticker, "d": target_date},
        ).scalar()

        if price_after is None:
            price_after = db.execute(
                text(
                    "SELECT close FROM stock_prices "
                    "WHERE ticker = :t AND date > :d "
                    "ORDER BY date ASC LIMIT 1"
                ),
                {"t": ticker, "d": target_date},
            ).scalar()

        if price_before is None or price_after is None:
            logger.debug("Missing prices for %s on %s — skipping", ticker, prediction_date)
            continue

        actual_change = (price_after - price_before) / price_before * 100
        was_correct = (direction == "BULLISH" and actual_change > 0) or (
            direction == "BEARISH" and actual_change < 0
        )

        db.execute(
            text(
                "UPDATE predictions "
                "SET actual_change_pct = :chg, was_correct = :ok "
                "WHERE id = :id"
            ),
            {"chg": round(actual_change, 2), "ok": int(was_correct), "id": pred_id},
        )

        evaluated += 1
        if was_correct:
            correct += 1

    db.commit()
    logger.info(
        "Evaluated %d predictions — %d correct (%.0f%%)",
        evaluated,
        correct,
        (correct / evaluated * 100) if evaluated else 0,
    )
    return {"evaluated": evaluated, "correct": correct}
