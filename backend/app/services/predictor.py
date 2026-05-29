import logging
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..models.db_models import Prediction
from ..config import settings
from ..ml.features import (
    CS_RANK_FEATURES,
    FEATURE_COLS,
    add_cross_sectional_ranks,
    compute_features,
)
from ..services.data_fetcher import FTSE_STOCKS
from ..services.market_sentiment import get_market_sentiment

logger = logging.getLogger(__name__)

_bundle = None          # {"model": XGBClassifier, "feature_cols": [...]}
_earnings_cache = None  # loaded once from disk


def _load_bundle():
    """Load model + feature list. Supports both legacy (raw model) and new (dict) formats."""
    global _bundle
    if _bundle is not None:
        return _bundle
    try:
        raw = joblib.load(settings.model_path)
        if isinstance(raw, dict):
            _bundle = raw
        else:
            # Legacy: plain XGBClassifier — derive feature cols from model
            _bundle = {"model": raw, "feature_cols": FEATURE_COLS}
        logger.info("Model loaded — %d features", len(_bundle["feature_cols"]))
    except FileNotFoundError:
        logger.warning("No model at %s — run train_model.py first.", settings.model_path)
    return _bundle


def _load_earnings_cache() -> dict:
    global _earnings_cache
    if _earnings_cache is None:
        try:
            from ..services.earnings_fetcher import load_earnings_cache
            _earnings_cache = load_earnings_cache()
        except Exception:
            _earnings_cache = {}
    return _earnings_cache


def _get_earnings_dates(ticker: str) -> list | None:
    """Return list of date objects for a ticker, or None if not in cache."""
    from datetime import datetime
    cache = _load_earnings_cache()
    raw   = cache.get(ticker)
    if not raw:
        return None
    return [datetime.strptime(s[:10], "%Y-%m-%d").date() for s in raw]


def run_predictions(db: Session) -> list[dict]:
    """
    Score every stock in FTSE_STOCKS.

    Steps:
      1. Fetch last 270 price rows per stock from DB.
      2. Compute per-stock technical features (+ earnings if cache loaded).
      3. Build cross-sectional rank features across all stocks.
      4. Run model predictions.
      5. Apply market-regime confidence nudge.

    Returns list sorted by confidence (desc).
    """
    bundle = _load_bundle()
    if bundle is None:
        return []

    model        = bundle["model"]
    feature_cols = bundle["feature_cols"]

    # Fetch once for all stocks
    sentiment    = get_market_sentiment()
    regime_score = sentiment.get("regime_score", 0)   # -3 to +3
    regime       = sentiment.get("regime", "NEUTRAL")
    NUDGE_PER_PT = 0.5   # pp per regime_score unit

    use_cs_ranks  = any(col in feature_cols for col in CS_RANK_FEATURES)

    # ── Step 1-2: per-stock feature computation ───────────────────────────────
    per_stock: list[dict] = []   # {"ticker", "name", "latest_row", "features_df"}

    for ticker, info in FTSE_STOCKS.items():
        name = info["name"] if isinstance(info, dict) else info[0]

        rows = db.execute(
            text(
                "SELECT date, open, high, low, close, volume "
                "FROM stock_prices WHERE ticker = :t "
                "ORDER BY date DESC LIMIT 270"
            ),
            {"t": ticker},
        ).fetchall()

        if len(rows) < 220:
            continue

        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        features = compute_features(df)

        if features.empty:
            continue

        per_stock.append({
            "ticker":      ticker,
            "name":        name,
            "features_df": features,
        })

    if not per_stock:
        return []

    # ── Step 3: cross-sectional rank features ─────────────────────────────────
    if use_cs_ranks:
        # Concatenate all latest-row snapshots, compute ranks, then re-extract
        latest_rows = pd.concat(
            [s["features_df"].iloc[[-1]].assign(ticker=s["ticker"]) for s in per_stock]
        )
        latest_ranked = add_cross_sectional_ranks(latest_rows)

        # Map rank values back to per-stock records
        rank_lookup: dict[str, dict] = {}
        for _, row in latest_ranked.iterrows():
            tkr = row.get("ticker")
            if tkr:
                rank_lookup[tkr] = {col: row.get(col, np.nan) for col in CS_RANK_FEATURES}
    else:
        rank_lookup = {}

    # ── Step 4-5: predict + regime nudge ──────────────────────────────────────
    results = []

    for s in per_stock:
        ticker = s["ticker"]
        name   = s["name"]
        latest = s["features_df"].iloc[[-1]].copy()

        # Inject cross-sectional ranks if available
        if ticker in rank_lookup:
            for col, val in rank_lookup[ticker].items():
                latest[col] = val

        # Build feature vector — only columns the model was trained on
        available = [c for c in feature_cols if c in latest.columns]
        missing   = [c for c in feature_cols if c not in latest.columns]
        if missing:
            for col in missing:
                latest[col] = np.nan

        X = latest[feature_cols]

        try:
            prob_up = float(model.predict_proba(X)[0][1])
        except Exception as exc:
            logger.debug("Prediction failed for %s: %s", ticker, exc)
            continue

        direction = "BULLISH" if prob_up >= 0.5 else "BEARISH"
        raw_conf  = (prob_up if prob_up >= 0.5 else 1 - prob_up) * 100

        # ── Regime nudge: aligned picks boosted, counter-trend penalised ────
        nudge = regime_score * NUDGE_PER_PT
        if direction == "BEARISH":
            nudge = -nudge

        # ── Earnings proximity adjustment ─────────────────────────────────────
        earnings_flag = None   # surfaced to frontend
        earnings_nudge = 0.0
        ed_dates = _get_earnings_dates(ticker)
        if ed_dates:
            from datetime import date as _date
            today = _date.today()
            future = [d for d in ed_dates if d >= today]
            past   = [d for d in ed_dates if d < today]
            days_to   = (future[0] - today).days  if future else None
            days_since = (today - past[-1]).days   if past   else None

            if days_to is not None and days_to <= 2:
                earnings_flag  = "⚡ Earnings in ≤2 days"
                earnings_nudge = -3.0   # high uncertainty — reduce confidence
            elif days_to is not None and days_to <= 7:
                earnings_flag  = f"📅 Earnings in {days_to}d"
                earnings_nudge = -1.5
            elif days_since is not None and days_since <= 5:
                # Fresh earnings — PEAD boost if we know the post-earnings direction
                earnings_flag  = f"📊 Earnings {days_since}d ago"
                earnings_nudge = +1.5   # mild PEAD tailwind
            elif days_since is not None and days_since <= 20:
                earnings_flag  = f"📊 Post-earnings ({days_since}d)"
                earnings_nudge = +0.5

        confidence = round(min(99.0, max(50.0, raw_conf + nudge + earnings_nudge)), 1)

        # ── Pull extra info for frontend ──────────────────────────────────────
        lbu     = int(latest.get("lbu_score",  [0]).values[0])
        cs_rank = float(latest.get("cs_rank_1m", [np.nan]).values[0]) if "cs_rank_1m" in latest.columns else None

        results.append({
            "ticker":          ticker,
            "name":            name,
            "direction":       direction,
            "confidence":      confidence,
            "prob_up":         round(prob_up * 100, 1),
            "technical_score": round(float(latest["rsi_14"].values[0]), 1),
            "volume_ratio":    round(float(latest["volume_ratio"].values[0]), 2),
            "return_5d":       round(float(latest["return_5d"].values[0]) * 100, 2),
            "lbu_score":       lbu,
            "cs_rank_1m":      round(cs_rank * 100, 1) if cs_rank is not None and not np.isnan(cs_rank) else None,
            "market_regime":   regime,
            "earnings_flag":   earnings_flag,
        })

    return sorted(results, key=lambda x: x["confidence"], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def _next_trading_day(from_date: date) -> date:
    d = from_date + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def save_predictions(db: Session, predictions: list[dict], prediction_date: date) -> int:
    """
    Persist predictions to DB. Skips any ticker already saved for this date.
    Returns number of rows inserted.
    """
    target = _next_trading_day(prediction_date)
    saved  = 0

    for pred in predictions:
        exists = db.query(Prediction).filter(
            Prediction.ticker == pred["ticker"],
            Prediction.prediction_date == prediction_date,
        ).first()
        if exists:
            continue

        db.add(Prediction(
            ticker=pred["ticker"],
            prediction_date=prediction_date,
            target_date=target,
            direction=pred["direction"],
            confidence=pred["confidence"],
            technical_score=pred["technical_score"],
            volume_score=pred["volume_ratio"],
        ))
        saved += 1

    if saved:
        db.commit()
    return saved
