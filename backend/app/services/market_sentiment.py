"""
Market-level sentiment using free Yahoo Finance data.

Since per-stock UK options PCR is unavailable for free, we use:
  - ^VIX  : US volatility index (global fear gauge, correlates with UK markets)
  - ^FTSE : FTSE 100 index price (momentum + local volatility)

Combined into a Market Regime: BULLISH / NEUTRAL / BEARISH
Used as a confidence modifier in predictions.
"""
import logging
from datetime import datetime, timezone
from functools import lru_cache

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Cache refreshes once per process — enough for daily use
_cache: dict = {}
_cache_date: str = ""


def get_market_sentiment() -> dict:
    """
    Returns market sentiment dict:
      vix          : latest VIX close
      vix_level    : 'LOW' | 'ELEVATED' | 'HIGH'
      ftse_1d      : FTSE 1-day return %
      ftse_5d      : FTSE 5-day return %
      ftse_20d     : FTSE 20-day return %
      regime       : 'BULLISH' | 'NEUTRAL' | 'BEARISH'
      regime_score : -2 to +2 (used as feature)
      summary      : plain-English sentence
    """
    global _cache, _cache_date
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if _cache and _cache_date == today:
        return _cache

    try:
        vix_data   = yf.Ticker("^VIX").history(period="10d")   # 5 trading days minimum
        ftse_data  = yf.Ticker("^FTSE").history(period="60d")  # ~42 trading days; need >=20

        if vix_data.empty or ftse_data.empty:
            return _neutral_sentiment("Market data unavailable")

        vix   = float(vix_data["Close"].iloc[-1])
        ftse  = ftse_data["Close"]
        f1d   = round((ftse.iloc[-1] / ftse.iloc[-2] - 1) * 100, 2) if len(ftse) >= 2  else 0.0
        f5d   = round((ftse.iloc[-1] / ftse.iloc[-5] - 1) * 100, 2) if len(ftse) >= 5  else 0.0
        f20d  = round((ftse.iloc[-1] / ftse.iloc[-20] - 1) * 100, 2) if len(ftse) >= 20 else 0.0

        # VIX level
        if vix < 15:
            vix_level = "LOW"
            vix_score = +1
        elif vix < 20:
            vix_level = "MODERATE"
            vix_score = 0
        elif vix < 25:
            vix_level = "ELEVATED"
            vix_score = -0.5
        elif vix < 30:
            vix_level = "HIGH"
            vix_score = -1
        else:
            vix_level = "EXTREME"
            vix_score = -2

        # FTSE momentum score (detect crash days more aggressively)
        ftse_score = 0

        # 1-day move — crash days show large 1d drops
        if f1d < -3.0:  # Crash detected (-3% in 1 day is severe)
            ftse_score -= 2
        elif f1d < -1.5:
            ftse_score -= 1.5
        elif f1d > 2.0:
            ftse_score += 1

        # 5-day move
        if f5d > 1.0:
            ftse_score += 0.5
        elif f5d < -1.0:
            ftse_score -= 0.5

        # 20-day move (trend)
        if f20d > 3.0:
            ftse_score += 1
        elif f20d < -3.0:
            ftse_score -= 1

        regime_score = vix_score + ftse_score  # -4 to +3

        # More aggressive BEARISH detection (regime_score <= -1.5)
        if regime_score >= 1.5:
            regime = "BULLISH"
        elif regime_score <= -1.5:
            regime = "BEARISH"
        else:
            regime = "NEUTRAL"

        vix_str = f"VIX at {vix:.1f} ({vix_level.lower()} fear)"
        ftse_str = f"FTSE {'+' if f5d >= 0 else ''}{f5d:.1f}% over 5 days"
        summary = f"{vix_str}, {ftse_str} → {regime} market regime"

        _cache = {
            "vix": round(vix, 1),
            "vix_level": vix_level,
            "ftse_1d": f1d,
            "ftse_5d": f5d,
            "ftse_20d": f20d,
            "regime": regime,
            "regime_score": regime_score,
            "summary": summary,
        }
        _cache_date = today
        return _cache

    except Exception as exc:
        logger.error("Market sentiment fetch failed: %s", exc)
        return _neutral_sentiment(str(exc))


def _neutral_sentiment(reason: str) -> dict:
    return {
        "vix": None,
        "vix_level": "UNKNOWN",
        "ftse_1d": 0.0,
        "ftse_5d": 0.0,
        "ftse_20d": 0.0,
        "regime": "NEUTRAL",
        "regime_score": 0,
        "summary": f"Market data unavailable ({reason})",
    }
