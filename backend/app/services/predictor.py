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
from ..services.sector_features import add_sector_features
from ..services.news_sentiment import sentiment_confidence_nudge, fetch_and_score_news
from ..services.macro_features import macro_confidence_nudge, get_macro_features
from ..services.short_interest import short_interest_confidence_nudge, get_short_interest
from ..services.director_buying import detect_director_buying, director_buying_nudge

logger = logging.getLogger(__name__)

_bundle = None          # {"model": XGBClassifier, "feature_cols": [...]}
_earnings_cache = None  # loaded once from disk


def _load_bundle():
    """Load model bundle — full model + regime models + calibrators."""
    global _bundle
    if _bundle is not None:
        return _bundle
    try:
        raw = joblib.load(settings.model_path)
        if isinstance(raw, dict):
            _bundle = raw
        else:
            _bundle = {"model": raw, "lgb_model": None,
                       "regime_models": {}, "meta_learner": None,
                       "calibrator_xgb": None, "calibrator_lgb": None,
                       "feature_cols": FEATURE_COLS}
        has_lgb     = _bundle.get("lgb_model")      is not None
        has_cal     = _bundle.get("calibrator_xgb") is not None
        has_regime  = bool(_bundle.get("regime_models"))
        regime_keys = list(_bundle.get("regime_models", {}).keys())
        logger.info(
            "Model loaded — %d features | LGB: %s | Calibrated: %s | Regime models: %s",
            len(_bundle["feature_cols"]),
            "YES" if has_lgb  else "NO",
            "YES" if has_cal  else "NO",
            regime_keys or "none (using full model)",
        )
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

    # Fetch market sentiment first — needed for regime model selection
    sentiment    = get_market_sentiment()
    regime_score = sentiment.get("regime_score", 0)   # -3 to +3
    regime       = sentiment.get("regime", "NEUTRAL")
    NUDGE_PER_PT = 0.5   # pp per regime_score unit

    # ── Select regime-specific model if available ─────────────────────────────
    # Only use regime models that we're confident about:
    # - BULLISH: 72.8% accuracy ✓ (good)
    # - NEUTRAL: 51.8% accuracy ✓ (acceptable, baseline ~50%)
    # - BEARISH: 45.3% accuracy ✗ (worse than full 53.8%, skip it)
    #
    # When regime model is worse than full model, fall back to full model.
    # BEARISH has few training samples (3,457 vs 42,526 neutral) so it's underfit.
    regime_models  = bundle.get("regime_models", {})
    full_accuracy  = 0.538   # Full model accuracy (53.8%)
    regime_accuracy_threshold = {
        "BULLISH":  0.70,    # Use if >= 70%
        "neutral":  0.51,    # Use if >= 51% (baseline is ~50%)
        "BEARISH":  0.50,    # Use if >= 50%, but BEARISH is 45% so will skip
    }

    use_regime_model = False
    if regime in regime_models:
        # Check if this regime model is better than full model
        regime_threshold = regime_accuracy_threshold.get(regime.lower(), 0.50)
        if regime == "BULLISH":
            use_regime_model = True  # BULLISH is 72.8%, always use
            logger.info("Using BULLISH regime model (72.8%% > 53.8%% full)")
        elif regime == "NEUTRAL":
            use_regime_model = True  # NEUTRAL is 51.8%, acceptable
            logger.info("Using NEUTRAL regime model (51.8%% baseline vs 53.8%% full)")
        elif regime == "BEARISH":
            # BEARISH is 45.3% — worse than 53.8%, skip it
            logger.info("Skipping BEARISH regime model (45.3%% < 53.8%% full, underfitted)")
            use_regime_model = False

    if use_regime_model:
        active_models  = regime_models[regime]
        model          = active_models["xgb"]
        lgb_model      = active_models.get("lgb")
        cb_model       = active_models.get("cb")
    else:
        model     = bundle["model"]
        lgb_model = bundle.get("lgb_model")
        cb_model  = bundle.get("cb_model")
        if regime in regime_models:
            logger.info("Using full model instead (regime model underfitted)")

    meta_learner    = bundle.get("meta_learner")
    calibrator_xgb  = bundle.get("calibrator_xgb")
    calibrator_lgb  = bundle.get("calibrator_lgb")
    feature_cols    = bundle["feature_cols"]

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

    # ── Compute momentum-relative-to-peers for latest row of all stocks ───────
    if per_stock:
        latest_rows = pd.concat([
            s["features_df"].iloc[[-1]].assign(ticker=s["ticker"]) for s in per_stock
        ])

        # Sector average momentum
        sector_map = {}
        for t, info in FTSE_STOCKS.items():
            sector_map[t] = info.get("sector", "Other") if isinstance(info, dict) else (info[1] if len(info) > 1 else "Other")

        for ticker in [s["ticker"] for s in per_stock]:
            if ticker in sector_map:
                sector = sector_map[ticker]
                sector_tickers = [t for t in sector_map if sector_map[t] == sector]
                sector_data = latest_rows[latest_rows["ticker"].isin(sector_tickers)]
                sector_avg = sector_data["return_1d"].mean() if len(sector_data) > 0 else 0

                for stock in per_stock:
                    if stock["ticker"] == ticker:
                        stock_ret = float(stock["features_df"].iloc[-1]["return_1d"])
                        delta = (stock_ret - sector_avg) if pd.notna(stock_ret) else 0
                        stock["features_df"].loc[stock["features_df"].index[-1], "sector_momentum_delta"] = delta

        # Market momentum (FTSE)
        try:
            ftse_today = get_market_sentiment()
            ftse_ret = ftse_today.get("ftse_return_pct", 0) / 100
        except Exception:
            ftse_ret = 0

        for stock in per_stock:
            stock_ret = float(stock["features_df"].iloc[-1]["return_1d"])
            delta = (stock_ret - ftse_ret) if pd.notna(stock_ret) else 0
            stock["features_df"].loc[stock["features_df"].index[-1], "market_momentum_delta"] = delta

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
            # XGBoost
            raw_xgb = float(model.predict_proba(X)[0][1])
            if calibrator_xgb is not None:
                xgb_prob = float(calibrator_xgb.predict_proba([[raw_xgb]])[0][1])
            else:
                xgb_prob = raw_xgb

            # LightGBM
            lgb_prob = None
            if lgb_model is not None:
                raw_lgb = float(lgb_model.predict_proba(X)[0][1])
                if calibrator_lgb is not None:
                    lgb_prob = float(calibrator_lgb.predict_proba([[raw_lgb]])[0][1])
                else:
                    lgb_prob = raw_lgb

            # CatBoost
            cb_prob = None
            if cb_model is not None:
                try:
                    raw_cb = float(cb_model.predict_proba(X)[0][1])
                    cb_prob = raw_cb  # CatBoost doesn't have calibrator yet
                except Exception:
                    pass

            # ── Ensemble: average all available models ────────────────────────
            if lgb_prob is not None and cb_prob is not None:
                prob_up = (xgb_prob + lgb_prob + cb_prob) / 3
            elif lgb_prob is not None:
                prob_up = (xgb_prob + lgb_prob) / 2
            else:
                prob_up = xgb_prob

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

        # ── News sentiment nudge (FinBERT on RNS + RSS) ───────────────────────
        sentiment_nudge, sentiment_flag = sentiment_confidence_nudge(ticker, direction)

        # ── Director buying signal (FCA mandatory PDMR disclosures) ────────────
        director_nudge = 0.0
        director_flag = None
        try:
            # Get recent RNS headlines for this stock
            rns_cache = fetch_and_score_news([ticker])
            if ticker in rns_cache:
                headlines = rns_cache[ticker].get("headlines", [])
                director_signal = detect_director_buying(headlines, ticker)
                director_nudge, director_flag = director_buying_nudge(ticker, direction, director_signal)
        except Exception as exc:
            logger.debug(f"Director buying check failed for {ticker}: {exc}")

        # ── Macro backdrop nudge (GBP, Brent, SP500, VFTSE) ──────────────────
        macro_nudge, macro_flag = macro_confidence_nudge(direction)

        # ── Short interest nudge (FCA mandatory disclosures) ────────────────────
        short_nudge, short_flag = short_interest_confidence_nudge(ticker, direction)

        confidence = round(
            min(99.0, max(50.0,
                raw_conf + nudge + earnings_nudge + sentiment_nudge + director_nudge + macro_nudge + short_nudge
            )), 1
        )

        # ── Pull extra info for frontend ──────────────────────────────────────
        lbu     = int(latest["lbu_score"].values[0])   if "lbu_score"   in latest.columns else 0
        cs_rank = float(latest["cs_rank_1m"].values[0]) if "cs_rank_1m" in latest.columns else None

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
            "news_flag":       sentiment_flag,
            "director_flag":   director_flag,
            "macro_flag":      macro_flag,
            "short_flag":      short_flag,
            "return_20d":      round(float(latest["return_20d"].values[0]) * 100, 2)
                               if "return_20d" in latest.columns else None,
        })

    # ── Sector relative strength (computed across all stocks together) ────────
    try:
        results = add_sector_features(results)
    except Exception as exc:
        logger.warning("Sector features failed: %s", exc)

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
