"""
Sector Relative Strength features.

For each stock, computes how it performs RELATIVE to the average of all
stocks in its sector. A stock up +2% when its sector is flat is a very
different signal from one up +2% when its whole sector rose +3%.

Features added per stock:
  sector_rel_1d    — return_1d  minus sector average return_1d
  sector_rel_5d    — return_5d  minus sector average return_5d
  sector_rel_20d   — return_20d minus sector average return_20d
  sector_rel_rsi   — rsi_14     minus sector average rsi_14
  sector_rank_1m   — percentile rank of return_20d within sector (0-1)

Academic backing: Springer OR Spectrum (2022) — cross-sectional momentum
is stronger when measured within-sector than in absolute terms.
"""
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from ..services.data_fetcher import FTSE_STOCKS

logger = logging.getLogger(__name__)

# Normalise sector names to handle Wikipedia capitalisation inconsistencies
_SECTOR_ALIASES = {
    "financial services": "Financial Services",
    "support services":   "Support Services",
    "household goods & home construction": "Household Goods",
    "multiline utilities": "Utilities",
    "nonlife insurance": "Insurance",
    "life insurance":    "Insurance",
}


def _normalise_sector(raw: str) -> str:
    if not raw:
        return "Unknown"
    lower = raw.strip().lower()
    return _SECTOR_ALIASES.get(lower, raw.strip().title())


def _build_sector_map() -> dict[str, str]:
    """Returns {ticker: normalised_sector}."""
    return {
        ticker: _normalise_sector(info.get("sector", "Unknown"))
        for ticker, info in FTSE_STOCKS.items()
        if isinstance(info, dict)
    }


def add_sector_features(
    records: list[dict],
    feature_keys: tuple[str, ...] = ("return_1d", "return_5d", "return_20d", "rsi_14"),
) -> list[dict]:
    """
    Given a list of prediction record dicts (each with ticker + numeric features),
    compute sector-relative features and add them in-place.

    Parameters
    ----------
    records : list of dicts — each must have 'ticker' and the keys in feature_keys.
    feature_keys : which features to compute sector averages for.

    Returns the same list with sector_rel_* and sector_rank_1m keys added.
    """
    sector_map = _build_sector_map()

    # Group records by sector
    sector_buckets: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        sector = sector_map.get(rec["ticker"], "Unknown")
        rec["_sector"] = sector
        sector_buckets[sector].append(rec)

    # Compute per-sector averages
    sector_avgs: dict[str, dict] = {}
    for sector, recs in sector_buckets.items():
        avgs = {}
        for key in feature_keys:
            vals = [r[key] for r in recs if r.get(key) is not None and not _isnan(r[key])]
            avgs[key] = float(np.mean(vals)) if vals else np.nan
        sector_avgs[sector] = avgs

    # Add relative features to each record
    for rec in records:
        sector  = rec["_sector"]
        avgs    = sector_avgs.get(sector, {})

        rec["sector_rel_1d"]  = _safe_diff(rec.get("return_1d"),  avgs.get("return_1d"))
        rec["sector_rel_5d"]  = _safe_diff(rec.get("return_5d"),  avgs.get("return_5d"))
        rec["sector_rel_20d"] = _safe_diff(rec.get("return_20d"), avgs.get("return_20d"))
        rec["sector_rel_rsi"] = _safe_diff(rec.get("technical_score"), avgs.get("rsi_14"))

        # Percentile rank of 1-month return within sector
        sector_ret20 = [
            r.get("return_5d", np.nan)
            for r in sector_buckets[sector]
            if not _isnan(r.get("return_5d", np.nan))
        ]
        if len(sector_ret20) >= 2:
            this_val = rec.get("return_5d", np.nan)
            if not _isnan(this_val):
                rec["sector_rank_1m"] = float(
                    sum(v <= this_val for v in sector_ret20) / len(sector_ret20)
                )
            else:
                rec["sector_rank_1m"] = None
        else:
            rec["sector_rank_1m"] = None

        del rec["_sector"]   # clean up temp key

    return records


def _safe_diff(a, b):
    if a is None or b is None:
        return None
    if _isnan(a) or _isnan(b):
        return None
    return round(float(a) - float(b), 5)


def _isnan(v):
    try:
        return np.isnan(float(v))
    except (TypeError, ValueError):
        return True


# ── Training-time sector features ─────────────────────────────────────────────

def add_sector_features_to_dataframe(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Add sector-relative features to a multi-stock training DataFrame.

    combined : DataFrame indexed by date, with 'ticker' column and feature columns.
    Sector averages are computed PER DATE so there is no look-ahead bias.
    """
    sector_map = _build_sector_map()
    combined   = combined.copy()
    combined["_sector"] = combined["ticker"].map(sector_map).fillna("Unknown")

    feature_keys = ["return_1d", "return_5d", "return_20d", "rsi_14"]

    # Per-date, per-sector averages
    for key in feature_keys:
        if key not in combined.columns:
            continue
        sector_avg = (
            combined.groupby([combined.index, "_sector"])[key]
            .transform("mean")
        )
        out_col = key.replace("return_", "sector_rel_").replace("rsi_14", "sector_rel_rsi")
        if "return" in key:
            out_col = f"sector_rel_{key.split('_')[1]}"   # sector_rel_1d etc.
        combined[out_col] = combined[key] - sector_avg

    # Sector percentile rank of 5d return per date
    combined["sector_rank_1m"] = (
        combined.groupby([combined.index, "_sector"])["return_5d"]
        .rank(pct=True)
        if "return_5d" in combined.columns
        else np.nan
    )

    combined.drop(columns=["_sector"], inplace=True, errors="ignore")
    return combined
