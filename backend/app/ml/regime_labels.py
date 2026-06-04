"""
Historical market regime labelling — no lookahead bias.

For each date in training history, computes the market regime
using ONLY data available on or before that date.

Regime definitions (consistent with market_sentiment.py live detection):
  BULLISH : FTSE 20d return > +2%  AND  VIX < 18
  BEARISH : FTSE 20d return < -2%  OR   VIX > 25
  NEUTRAL : everything else

Data sources (free, via yfinance):
  ^FTSE — FTSE 100 index (UK market proxy)
  ^VIX  — CBOE Volatility Index (global fear gauge)

Cache: computed once per session, stored in memory.

Academic backing: RegimeFolio (arXiv:2510.14986) — regime-aware ML gave
15–20% improvement in forecast accuracy vs regime-agnostic baseline.
"""
import logging
from functools import lru_cache

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Thresholds (aligned with market_sentiment.py live logic)
BULL_FTSE_20D  =  0.02   # +2% 20-day return = bullish momentum
BEAR_FTSE_20D  = -0.02   # -2% 20-day return = bearish momentum
BULL_VIX_MAX   = 18.0    # VIX below 18 = calm (BULL regime)
BEAR_VIX_MIN   = 25.0    # VIX above 25 = fear  (BEAR regime)

REGIME_LABELS  = ("BULLISH", "NEUTRAL", "BEARISH")
MIN_REGIME_ROWS = 2000   # Minimum rows to train a regime model


@lru_cache(maxsize=1)
def _fetch_regime_history(period: str = "10y") -> pd.DataFrame:
    """
    Fetch FTSE + VIX history and compute daily regime labels.
    Cached for the process lifetime (refreshed on restart).

    Returns DataFrame indexed by date with columns:
      ftse_20d_ret  — FTSE 100 20-day rolling return
      vix           — VIX closing level
      regime        — "BULLISH" / "NEUTRAL" / "BEARISH"
    """
    logger.info("Fetching FTSE + VIX history for regime labelling (%s)...", period)

    try:
        ftse_raw = yf.Ticker("^FTSE").history(period=period)["Close"]
        vix_raw  = yf.Ticker("^VIX").history(period=period)["Close"]
    except Exception as exc:
        logger.error("Failed to fetch regime history: %s", exc)
        return pd.DataFrame()

    # Normalise both indexes to plain date (strip different timezones)
    ftse = pd.Series(
        ftse_raw.values,
        index=pd.to_datetime([d.date() for d in ftse_raw.index]),
        name="ftse",
    )
    vix = pd.Series(
        vix_raw.values,
        index=pd.to_datetime([d.date() for d in vix_raw.index]),
        name="vix",
    )

    # Align on common dates
    df = pd.concat([ftse, vix], axis=1).dropna()

    # 20-day FTSE return (no lookahead — uses only past 20 days)
    df["ftse_20d_ret"] = df["ftse"].pct_change(20)

    # Regime label per date
    df["regime"] = "NEUTRAL"
    bull_mask = (df["ftse_20d_ret"] > BULL_FTSE_20D) & (df["vix"] < BULL_VIX_MAX)
    bear_mask = (df["ftse_20d_ret"] < BEAR_FTSE_20D) | (df["vix"] > BEAR_VIX_MIN)

    df.loc[bull_mask, "regime"] = "BULLISH"
    df.loc[bear_mask, "regime"] = "BEARISH"

    counts = df["regime"].value_counts()
    total  = max(len(df), 1)
    logger.info(
        "Regime distribution: BULLISH=%d (%.0f%%) | NEUTRAL=%d (%.0f%%) | BEARISH=%d (%.0f%%)",
        counts.get("BULLISH", 0), counts.get("BULLISH", 0) / total * 100,
        counts.get("NEUTRAL",  0), counts.get("NEUTRAL",  0) / total * 100,
        counts.get("BEARISH",  0), counts.get("BEARISH",  0) / total * 100,
    )

    return df[["ftse_20d_ret", "vix", "regime"]]


def add_regime_labels(data: pd.DataFrame) -> pd.DataFrame:
    """
    Join historical regime labels onto a training DataFrame.
    The DataFrame index must be a DatetimeIndex of trading dates.

    Adds column: 'regime' — "BULLISH" / "NEUTRAL" / "BEARISH"
    Rows where regime is unknown (before VIX history starts) are labelled NEUTRAL.
    """
    regime_hist = _fetch_regime_history()

    if regime_hist.empty:
        data = data.copy()
        data["regime"] = "NEUTRAL"
        return data

    # Normalise both indexes to date-only strings to avoid dtype/tz mismatches
    # (regime_hist uses datetime64[s], training data uses datetime64[us])
    regime_str = regime_hist["regime"].copy()
    regime_str.index = regime_hist.index.strftime("%Y-%m-%d")

    data_str = pd.Index(
        pd.to_datetime(data.index).strftime("%Y-%m-%d")
    )

    regime_values = data_str.map(regime_str).fillna("NEUTRAL")

    data = data.copy()
    data["regime"] = regime_values.values
    return data


def get_current_regime() -> str:
    """
    Return today's regime from the most recent historical label.
    Used as a fallback if market_sentiment.py is unavailable.
    """
    hist = _fetch_regime_history()
    if hist.empty:
        return "NEUTRAL"
    return str(hist["regime"].iloc[-1])
