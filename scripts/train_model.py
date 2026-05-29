"""
Train XGBoost prediction model on stored historical prices.

Usage:
    python3 scripts/train_model.py            # standard train
    python3 scripts/train_model.py --evaluate # also run walk-forward CV
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from sqlalchemy.orm import sessionmaker

from backend.app.database import engine
from backend.app.ml.train import train_model, walk_forward_evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    run_eval = "--evaluate" in sys.argv

    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        # ── Train production model ───────────────────────────────────────────
        logger.info("Loading training data and fitting model…")
        results = train_model(db)

        logger.info("Training complete.")
        logger.info("  Accuracy      : %.1f%%", results["accuracy"] * 100)
        logger.info("  Features used : %d",     results["feature_count"])
        logger.info("  Train rows    : %d",      results["train_samples"])
        logger.info("  Test rows     : %d",      results["test_samples"])
        logger.info("  Split date    : %s",      results["split_date"])

        if results["accuracy"] < 0.52:
            logger.warning(
                "Backtest accuracy is below 52%% — expected for technical features alone. "
                "Real-world accuracy (tracked daily) is the meaningful measure."
            )

        # ── Walk-forward evaluation (optional) ──────────────────────────────
        if run_eval:
            logger.info("")
            logger.info("Running walk-forward cross-validation (5 folds)…")
            wf = walk_forward_evaluate(db, n_folds=5)

            if "error" in wf:
                logger.warning("Walk-forward failed: %s", wf["error"])
            else:
                logger.info("Walk-forward results:")
                for i, fold in enumerate(wf["folds"], 1):
                    logger.info(
                        "  Fold %d: train %s→%s  test %s→%s  acc=%.1f%%",
                        i,
                        fold["train_start"][:7], fold["train_end"][:7],
                        fold["test_start"][:7],  fold["test_end"][:7],
                        fold["accuracy"],
                    )
                logger.info(
                    "  Mean accuracy: %.1f%% ± %.1f%%",
                    wf["mean_accuracy"], wf["std_accuracy"]
                )
                logger.info(
                    "  (This is a more reliable estimate than the single backtest above)"
                )

    except ValueError as exc:
        logger.error("Cannot train: %s", exc)
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
