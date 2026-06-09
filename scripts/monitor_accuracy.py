"""
Daily accuracy monitor — run this to check yesterday's prediction performance.

Shows:
  - Overall accuracy percentage
  - BULLISH vs BEARISH accuracy breakdown
  - Market conditions (% up, average return)
  - Worst predictions (lowest accuracy)
  - Comparison to baseline

Run manually:
  python3 scripts/monitor_accuracy.py [--days=1]

Or integrate into cron for automatic daily checks:
  0 9 * * * cd /path/to/app && python3 scripts/monitor_accuracy.py >> /tmp/daily_accuracy.log
"""

import logging
import sys
import os
from datetime import date, timedelta
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


def monitor_accuracy(target_date: date = None, days_back: int = 1) -> dict:
    """
    Check prediction accuracy for a given date.

    Parameters:
    -----------
    target_date : date, optional
        Date to evaluate (default: today)
    days_back : int
        How many days back to look (default 1 = yesterday's predictions)

    Returns:
    --------
    dict with accuracy metrics
    """
    from backend.app.database import SessionLocal
    from backend.app.models.db_models import Prediction
    import pandas as pd

    if target_date is None:
        target_date = date.today() - timedelta(days=days_back)

    db = SessionLocal()
    try:
        preds = db.query(Prediction).filter(
            Prediction.target_date == target_date
        ).all()

        if not preds:
            logger.warning(f"No predictions found for {target_date}")
            return {"date": target_date, "total": 0, "accuracy": None}

        # Separate by direction
        correct = 0
        total = 0
        bullish_preds = []
        bearish_preds = []

        for p in preds:
            if p.actual_change_pct is not None:
                total += 1
                is_correct = (
                    (p.direction == "BULLISH" and p.actual_change_pct > 0) or
                    (p.direction == "BEARISH" and p.actual_change_pct < 0)
                )
                if is_correct:
                    correct += 1

                if p.direction == "BULLISH":
                    bullish_preds.append({
                        'ticker': p.ticker,
                        'conf': p.confidence,
                        'actual': p.actual_change_pct,
                        'correct': is_correct
                    })
                else:
                    bearish_preds.append({
                        'ticker': p.ticker,
                        'conf': p.confidence,
                        'actual': p.actual_change_pct,
                        'correct': is_correct
                    })

        # Calculate accuracy
        accuracy = correct / max(total, 1) * 100

        bullish_correct = sum(1 for p in bullish_preds if p['correct'])
        bullish_acc = bullish_correct / max(len(bullish_preds), 1) * 100 if bullish_preds else 0

        bearish_correct = sum(1 for p in bearish_preds if p['correct'])
        bearish_acc = bearish_correct / max(len(bearish_preds), 1) * 100 if bearish_preds else 0

        # Market conditions
        all_returns = [p['actual'] for p in bullish_preds + bearish_preds]
        up_count = sum(1 for r in all_returns if r > 0)
        down_count = sum(1 for r in all_returns if r < 0)
        avg_return = pd.Series(all_returns).mean() * 100

        # Log results
        logger.info("=" * 70)
        logger.info(f"ACCURACY REPORT: {target_date}")
        logger.info("=" * 70)
        logger.info(f"Overall:        {correct:3d}/{total:3d} = {accuracy:5.1f}%")
        logger.info(f"BULLISH calls:  {bullish_correct:3d}/{len(bullish_preds):3d} = {bullish_acc:5.1f}%")
        logger.info(f"BEARISH calls:  {bearish_correct:3d}/{len(bearish_preds):3d} = {bearish_acc:5.1f}%")
        logger.info("")
        logger.info(f"Market conditions:")
        logger.info(f"  Up:   {up_count:3d} ({up_count/len(all_returns)*100:.0f}%)")
        logger.info(f"  Down: {down_count:3d} ({down_count/len(all_returns)*100:.0f}%)")
        logger.info(f"  Avg return: {avg_return:+.2f}%")

        # Alert on poor performance
        if accuracy < 45 and len(all_returns) > 100:
            if avg_return < -5:
                logger.warning("⚠️  CRASH DAY DETECTED (avg return %.1f%%) — accuracy naturally lower", avg_return)
            else:
                logger.warning("⚠️  POOR PERFORMANCE — below 45% on normal market day")

        # Baseline comparison
        logger.info(f"\n  Baseline (random 50/50): 50.0%")
        logger.info(f"  Current vs baseline:     {accuracy - 50:+.1f}pp")

        return {
            "date": target_date,
            "total": total,
            "accuracy": round(accuracy, 1),
            "bullish_acc": round(bullish_acc, 1),
            "bearish_acc": round(bearish_acc, 1),
            "avg_return": round(avg_return, 2),
            "market_condition": "CRASH" if avg_return < -5 else ("DOWN" if avg_return < -1 else "UP"),
        }

    finally:
        db.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--days", type=int, default=1, help="Days back to check (default 1 = yesterday)")
    parser.add_argument("--date", type=str, help="Specific date (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.date:
        target = date.fromisoformat(args.date)
        monitor_accuracy(target_date=target)
    else:
        monitor_accuracy(days_back=args.days)
