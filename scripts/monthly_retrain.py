"""
Monthly model retrain — run on the 1st of each month.

Why monthly retraining matters:
  Markets shift regime roughly every 3-6 months. A model trained once
  on historical data will gradually drift as the current regime diverges
  from the training distribution. Monthly retraining keeps the model
  fresh and captures the most recent market behaviour.

  Research (arXiv:2601.08896): Monthly retraining with expanding window +
  recency weighting gives +1 to +3pp accuracy improvement vs. static model.

Schedule (automated via crontab):
  0 6 1 * *  /path/to/python3 scripts/monthly_retrain.py

Or run manually:
  python3 scripts/monthly_retrain.py
"""
import logging
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("MONTHLY RETRAIN — %s", date.today())
    logger.info("=" * 60)

    from backend.app.database import SessionLocal
    from backend.app.ml.train import train_model
    from backend.app.config import settings

    db = SessionLocal()
    try:
        logger.info("Training XGB + LGB ensemble with recency weighting...")
        result = train_model(db, model_path=settings.model_path)

        logger.info("")
        logger.info("Training complete:")
        logger.info("  Accuracy      : %.1f%%", result["accuracy"] * 100)
        logger.info("  Features      : %d", result["feature_count"])
        logger.info("  Train samples : %d", result["train_samples"])
        logger.info("  Test samples  : %d", result["test_samples"])
        logger.info("  Split date    : %s", result["split_date"])
        logger.info("")
        logger.info("Model saved — predictions will use new model from next pipeline run.")

        # Log to a permanent retrain history file
        history_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "retrain_history.log"
        )
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, "a") as f:
            f.write(
                f"{date.today()},{result['accuracy']*100:.2f}%,"
                f"{result['feature_count']} features,"
                f"{result['train_samples']} train rows\n"
            )
        logger.info("Retrain history logged to %s", history_path)

    except Exception as exc:
        logger.error("Monthly retrain FAILED: %s", exc)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
