from datetime import date

import logging

from fastapi import APIRouter, Depends, HTTPException

logger = logging.getLogger(__name__)
from sqlalchemy.orm import Session

from ...database import get_db
from ...models.schemas import DailyPredictionsOut, PredictionOut
from ...services.explainer import generate_explanation
from ...services.predictor import run_predictions
from ...services.rns_scraper import fetch_news_headlines
from ...services.market_sentiment import get_market_sentiment

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/daily", response_model=DailyPredictionsOut)
def get_daily_predictions(db: Session = Depends(get_db)):
    predictions = run_predictions(db)
    if not predictions:
        raise HTTPException(
            status_code=503,
            detail="No predictions available. Ensure the model is trained (run train_model.py).",
        )

    gainers = [p for p in predictions if p["direction"] == "BULLISH"][:10]
    losers = [p for p in predictions if p["direction"] == "BEARISH"][:10]

    return {
        "date": str(date.today()),
        "top_gainers": gainers,
        "top_losers": losers,
        "total_analysed": len(predictions),
    }


@router.get("/market-sentiment")
def market_sentiment_endpoint():
    """
    Returns current market regime based on VIX + FTSE momentum.
    Used by the frontend Fear/Greed meter.
    """
    try:
        return get_market_sentiment()
    except Exception as exc:
        logger.warning("Market sentiment fetch failed: %s", exc)
        raise HTTPException(status_code=503, detail="Market data temporarily unavailable")


@router.get("/{ticker}/explain", response_model=PredictionOut)
def explain_prediction(ticker: str, db: Session = Depends(get_db)):
    """Return prediction + Claude-generated explanation for a single stock."""
    normalised = ticker.upper()
    if not normalised.endswith(".L"):
        normalised += ".L"

    predictions = run_predictions(db)
    pred = next((p for p in predictions if p["ticker"] == normalised), None)

    if pred is None:
        raise HTTPException(status_code=404, detail=f"No prediction found for {ticker}")

    headlines = [h["title"] for h in fetch_news_headlines(normalised)]
    explanation = generate_explanation(
        ticker=pred["ticker"],
        name=pred["name"],
        direction=pred["direction"],
        confidence=pred["confidence"],
        rsi=pred["technical_score"],
        volume_ratio=pred["volume_ratio"],
        return_5d=pred.get("return_5d", 0.0),
        news_headlines=headlines,
    )

    return {**pred, "explanation": explanation}
