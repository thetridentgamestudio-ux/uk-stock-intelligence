"""
Global macro features — free daily instruments from Yahoo Finance.

FTSE 100 companies earn ~75% of revenue overseas, so overnight moves
in GBP, crude, and global indices are predictive BEFORE the UK market opens.

Features fetched once per day and cached:
  gbpusd_1d     GBP/USD overnight return (weak GBP → FTSE 100 bullish)
  brent_1d      Brent crude overnight return (BP/Shell = ~15% FTSE 100)
  sp500_1d      S&P 500 overnight / prior-day return (global risk-on/off)
  dax_1d        DAX overnight return (European sentiment)
  vftse         VFTSE level (UK-specific fear index)
  vftse_level   LOW / ELEVATED / HIGH
  gilt_10y_1d   UK 10-year gilt yield 1-day change (rate pressure)
"""
import logging
from datetime import datetime, timezone

import yfinance as yf

logger = logging.getLogger(__name__)

_cache: dict = {}
_cache_date: str = ""

# Tickers
_TICKERS = {
    "gbpusd":  "GBPUSD=X",
    "brent":   "BZ=F",
    "sp500":   "^GSPC",
    "dax":     "^GDAXI",
    "vftse":   "^VIX",    # VIX (^VFTSE not reliably available on Yahoo Finance)
    "gilt_10y": "^TNX",   # US 10yr proxy (correlates closely with UK gilts)
}


def get_macro_features() -> dict:
    """
    Returns dict of macro features. Cached once per day.

    Keys: gbpusd_1d, brent_1d, sp500_1d, dax_1d, vftse, vftse_level,
          gilt_10y_1d, macro_score (-3..+3 bullish signal strength)
    """
    global _cache, _cache_date
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _cache and _cache_date == today:
        return _cache

    result = {}
    try:
        for name, ticker in _TICKERS.items():
            try:
                hist = yf.Ticker(ticker).history(period="5d")
                if hist.empty or len(hist) < 2:
                    result[name] = None
                    continue
                close = hist["Close"]
                result[name] = float(close.iloc[-1])
                result[f"{name}_prev"] = float(close.iloc[-2])
            except Exception:
                result[name] = None
                result[f"{name}_prev"] = None

        # 1-day returns
        def ret(key):
            v, p = result.get(key), result.get(f"{key}_prev")
            if v is None or p is None or p == 0:
                return None
            return round((v - p) / abs(p) * 100, 3)

        gbpusd_1d  = ret("gbpusd")
        brent_1d   = ret("brent")
        sp500_1d   = ret("sp500")
        dax_1d     = ret("dax")
        gilt_10y_1d = ret("gilt_10y")
        vftse_val  = result.get("vftse")

        # VIX-style classification for VFTSE
        if vftse_val is not None:
            if vftse_val < 15:
                vftse_level = "LOW"
            elif vftse_val < 25:
                vftse_level = "ELEVATED"
            else:
                vftse_level = "HIGH"
        else:
            vftse_level = "UNKNOWN"

        # Macro score: count bullish vs bearish signals (-3 to +3)
        macro_score = 0
        if gbpusd_1d is not None:
            # Weak GBP = bullish for FTSE 100 (overseas revenue boost)
            if gbpusd_1d < -0.3:
                macro_score += 1
            elif gbpusd_1d > 0.3:
                macro_score -= 1
        if brent_1d is not None:
            if brent_1d > 1.0:
                macro_score += 1
            elif brent_1d < -1.0:
                macro_score -= 1
        if sp500_1d is not None:
            if sp500_1d > 0.5:
                macro_score += 1
            elif sp500_1d < -0.5:
                macro_score -= 1
        if vftse_level == "LOW":
            macro_score += 1
        elif vftse_level == "HIGH":
            macro_score -= 1
        if gilt_10y_1d is not None:
            # Rising gilt yields = bearish for equities
            if gilt_10y_1d > 0.05:
                macro_score -= 1
            elif gilt_10y_1d < -0.05:
                macro_score += 1

        _cache = {
            "gbpusd_1d":   gbpusd_1d,
            "brent_1d":    brent_1d,
            "sp500_1d":    sp500_1d,
            "dax_1d":      dax_1d,
            "gilt_10y_1d": gilt_10y_1d,
            "vftse":       round(vftse_val, 1) if vftse_val else None,
            "vftse_level": vftse_level,
            "macro_score": macro_score,
            "summary":     _build_summary(gbpusd_1d, brent_1d, sp500_1d, vftse_level, macro_score),
        }
        # Remove temp prev keys
        _cache_date = today
        logger.info("Macro features: GBP %s%s%% | Brent %s%s%% | SP500 %s%s%% | score=%s",
                    "+" if (gbpusd_1d or 0) >= 0 else "", gbpusd_1d,
                    "+" if (brent_1d or 0) >= 0 else "", brent_1d,
                    "+" if (sp500_1d or 0) >= 0 else "", sp500_1d,
                    macro_score)

    except Exception as exc:
        logger.error("Macro features fetch failed: %s", exc)
        _cache = _neutral_macro()
        _cache_date = today

    return _cache


def _build_summary(gbpusd, brent, sp500, vftse_level, score):
    parts = []
    if gbpusd is not None:
        parts.append(f"GBP {'+' if gbpusd >= 0 else ''}{gbpusd:.2f}%")
    if brent is not None:
        parts.append(f"Brent {'+' if brent >= 0 else ''}{brent:.2f}%")
    if sp500 is not None:
        parts.append(f"SP500 {'+' if sp500 >= 0 else ''}{sp500:.2f}%")
    label = "BULLISH" if score >= 2 else ("BEARISH" if score <= -2 else "NEUTRAL")
    return f"{', '.join(parts)} → {label} macro backdrop"


def _neutral_macro():
    return {
        "gbpusd_1d": None, "brent_1d": None,
        "sp500_1d": None, "dax_1d": None,
        "gilt_10y_1d": None, "vftse": None,
        "vftse_level": "UNKNOWN", "macro_score": 0,
        "summary": "Macro data unavailable",
    }


def macro_confidence_nudge(direction: str) -> tuple[float, str | None]:
    """
    Returns (nudge_pp, flag) based on macro backdrop vs prediction direction.
    Max ±1.5 pp (macro is supportive but not decisive).
    """
    macro = get_macro_features()
    score = macro.get("macro_score", 0)
    if score == 0:
        return 0.0, None

    NUDGE = 0.3   # pp per macro score point (max ±1.5 at score ±5)
    nudge = score * NUDGE
    if direction == "BEARISH":
        nudge = -nudge

    if abs(nudge) < 0.2:
        return 0.0, None

    flag = None
    if nudge > 0:
        flag = f"🌍 Macro tailwind"
    elif nudge < 0:
        flag = f"🌍 Macro headwind"

    return round(nudge, 2), flag
