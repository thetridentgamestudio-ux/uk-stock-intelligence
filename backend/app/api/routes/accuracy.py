from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from ...database import get_db
from ...ml.train import walk_forward_evaluate

router = APIRouter(prefix="/accuracy", tags=["accuracy"])


@router.get("/summary")
def accuracy_summary(db: Session = Depends(get_db)):
    """Overall accuracy across all evaluated predictions."""
    row = db.execute(
        text(
            "SELECT COUNT(*), "
            "SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END), "
            "COUNT(DISTINCT prediction_date) "
            "FROM predictions WHERE was_correct IS NOT NULL"
        )
    ).fetchone()

    total, correct, days = row or (0, 0, 0)
    total = total or 0
    correct = correct or 0

    return {
        "total_predictions": total,
        "correct": correct,
        "accuracy_pct": round(correct / total * 100, 1) if total else None,
        "days_tracked": days or 0,
    }


@router.get("/history")
def accuracy_history(
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Recent predictions with actual outcomes."""
    rows = db.execute(
        text(
            "SELECT ticker, prediction_date, target_date, direction, "
            "confidence, actual_change_pct, was_correct "
            "FROM predictions WHERE was_correct IS NOT NULL "
            "ORDER BY prediction_date DESC, confidence DESC "
            "LIMIT :lim"
        ),
        {"lim": limit},
    ).fetchall()

    return [
        {
            "ticker": r[0],
            "prediction_date": str(r[1]),
            "target_date": str(r[2]),
            "direction": r[3],
            "confidence": r[4],
            "actual_change_pct": r[5],
            "was_correct": bool(r[6]),
        }
        for r in rows
    ]


@router.get("/walk-forward")
def walk_forward_accuracy(n_folds: int = 5, db: Session = Depends(get_db)):
    """
    Run rolling walk-forward cross-validation and return per-fold accuracy.
    Gives a statistically reliable accuracy estimate (more honest than single backtest).
    Warning: takes ~2-3 minutes.
    """
    return walk_forward_evaluate(db, n_folds=n_folds)


@router.get("/by-stock")
def accuracy_by_stock(db: Session = Depends(get_db)):
    """Accuracy broken down per stock, best performers first."""
    rows = db.execute(
        text(
            "SELECT ticker, COUNT(*) as total, "
            "SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct "
            "FROM predictions WHERE was_correct IS NOT NULL "
            "GROUP BY ticker "
            "ORDER BY SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) DESC"
        )
    ).fetchall()

    return [
        {
            "ticker": r[0],
            "total": r[1],
            "correct": r[2],
            "accuracy_pct": round(r[2] / r[1] * 100, 1),
        }
        for r in rows
    ]
