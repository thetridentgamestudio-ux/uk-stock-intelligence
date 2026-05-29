import numpy as np
import pandas as pd

# ── Original 14 features ─────────────────────────────────────────────────────
_BASE_FEATURES = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_position",
    "price_vs_ma5", "price_vs_ma10", "price_vs_ma20",
    "volume_ratio",
    "daily_range",
]

# ── Phase-1 signals ───────────────────────────────────────────────────────────
_NEW_FEATURES = [
    # Signal 1 — Long Build-Up (enhanced)
    "lbu_score",            # 0-4: price up 2 days + volume up + momentum up
    "consecutive_up_closes",# number of consecutive up closes (capped at 5)
    "hhhl_pattern",         # 1 if today: high > prev high AND low > prev low
    "volume_trend_3d",      # avg volume last 3 days vs avg volume days 4-6

    # Signal 3 — Breakout
    "dist_from_20d_high",   # (close - 20d rolling max) / 20d rolling max
    "dist_from_52w_high",   # (close - 252d rolling max) / 252d rolling max

    # Signal 4 — Close above prev day/week high
    "above_prev_day_high",  # binary: close > yesterday's high
    "above_prev_week_high", # binary: close > highest high of prev 5 days

    # Signal 5 — Golden crossover + volume spike
    "ma20_vs_ma50",         # ma20/ma50 - 1
    "ma50_vs_ma200",        # ma50/ma200 - 1
    "volume_spike",         # binary: volume > 2x 20-day average
]

# ── Phase-2 signals (research-backed additions) ───────────────────────────────
_PHASE2_FEATURES = [
    # Extended momentum returns
    "return_60d",           # ~3-month momentum
    "return_120d",          # ~6-month momentum
    "return_252d",          # ~12-month momentum

    # Candle structure — gap and intraday conviction
    "gap_up_pct",           # (open - prev_close) / prev_close — overnight gap
    "candle_body_ratio",    # abs(close - open) / range — conviction of move
    "upper_wick_ratio",     # selling rejection at top of range
    "lower_wick_ratio",     # buying support at bottom of range

    # Indicator velocity — are signals accelerating or fading?
    "rsi_delta_5d",         # RSI today minus RSI 5 days ago
    "macd_hist_delta",      # MACD histogram change (1-day)
    "bb_position_delta",    # Bollinger Band position change (5-day)
    "obv_slope",            # OBV momentum — volume-weighted flow direction

    # New technical indicators (research-selected)
    "atr_ratio",            # ATR / 20d avg ATR — volatility regime
    "adx",                  # Average Directional Index — trend strength 0-100
    "di_diff",              # +DI - -DI — directional bias
    "stoch_k",              # Stochastic %K(14) — close vs recent range
    "stoch_d",              # Stochastic %D(3) — smoothed %K signal line
    "williams_r",           # Williams %R(14) — overbought/oversold
    "cmf",                  # Chaikin Money Flow(20) — buying/selling pressure
    "cci",                  # Commodity Channel Index(20) — mean deviation
    "price_accel",          # return_1d minus avg_daily_return_5d — acceleration
]

FEATURE_COLS = _BASE_FEATURES + _NEW_FEATURES + _PHASE2_FEATURES

# Cross-sectional rank features — appended AFTER per-stock feature computation
# (computed in train.py / predictor.py across the full universe per date)
CS_RANK_FEATURES = [
    "cs_rank_1m",           # percentile rank of return_20d across all stocks
    "cs_rank_3m",           # percentile rank of return_60d
    "cs_rank_6m",           # percentile rank of return_120d
    "cs_rank_12_1m",        # percentile rank of (return_252d - return_20d)
]

# Earnings features — NOT used in model training (insufficient historical data).
# Applied as post-processing confidence modifiers in predictor.py instead.
# Kept here for reference and future use when historical data is available.
EARNINGS_FEATURES = [
    "days_to_earnings",     # calendar days until next earnings
    "days_since_earnings",  # calendar days since last earnings
    "pre_earnings_flag",    # binary: ≤5 days before earnings
    "post_earnings_flag",   # binary: ≤20 days after earnings (PEAD window)
]


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    hl  = high - low
    hpc = (high - close.shift(1)).abs()
    lpc = (low  - close.shift(1)).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """
    Returns (adx, di_diff) where di_diff = +DI - -DI.
    ADX is smoothed with Wilder's EMA.
    """
    atr = _atr(high, low, close, period)

    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    sm_plus_dm  = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
    sm_minus_dm = minus_dm.ewm(alpha=1 / period, adjust=False).mean()
    sm_atr      = atr

    plus_di  = 100 * sm_plus_dm  / sm_atr.replace(0, np.nan)
    minus_di = 100 * sm_minus_dm / sm_atr.replace(0, np.nan)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx, (plus_di - minus_di)


def compute_features(
    df: pd.DataFrame,
    earnings_dates: list | None = None,
) -> pd.DataFrame:
    """
    Compute all technical features from OHLCV data.

    Parameters
    ----------
    df : DataFrame with columns open, high, low, close, volume (lowercase).
    earnings_dates : optional list of datetime-like objects representing
                     historical + upcoming earnings announcement dates
                     for this stock (used to compute PEAD features).

    Returns DataFrame with FEATURE_COLS (+ earnings features if provided).
    Rows with NaN in required features are dropped.
    """
    df   = df.copy().sort_index()
    open_ = df["open"]
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    volume = df["volume"]

    # ── Original 14 features ─────────────────────────────────────────────────

    df["return_1d"]  = close.pct_change(1)
    df["return_5d"]  = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)
    df["return_20d"] = close.pct_change(20)

    df["rsi_14"] = _rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    ma20  = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_position"] = (close - ma20) / (2 * std20.replace(0, np.nan))

    df["price_vs_ma5"]  = close / close.rolling(5).mean() - 1
    df["price_vs_ma10"] = close / close.rolling(10).mean() - 1
    df["price_vs_ma20"] = close / ma20 - 1

    vol_ma20 = volume.rolling(20).mean()
    df["volume_ratio"] = volume / vol_ma20.replace(0, np.nan)
    df["daily_range"]  = (high - low) / close

    # ── Signal 1: Long Build-Up (enhanced) ───────────────────────────────────

    price_up_today     = (close > close.shift(1)).astype(int)
    price_up_yesterday = (close.shift(1) > close.shift(2)).astype(int)
    volume_above_avg   = (volume > vol_ma20).astype(int)
    momentum_positive  = (df["return_5d"] > 0).astype(int)
    df["lbu_score"] = price_up_today + price_up_yesterday + volume_above_avg + momentum_positive

    daily_up     = (close > close.shift(1)).astype(int)
    streak_group = (daily_up != daily_up.shift()).cumsum()
    raw_streak   = daily_up.groupby(streak_group).cumcount() + 1
    df["consecutive_up_closes"] = raw_streak.where(daily_up == 1, 0).clip(upper=5).astype(float)

    df["hhhl_pattern"] = (
        (high > high.shift(1)) & (low > low.shift(1))
    ).astype(int)

    vol_recent = volume.rolling(3).mean()
    vol_prior  = volume.shift(3).rolling(3).mean()
    df["volume_trend_3d"] = vol_recent / vol_prior.replace(0, np.nan)

    # ── Signal 3: Breakout ────────────────────────────────────────────────────

    high_20d  = close.rolling(20).max().shift(1)
    high_252d = close.rolling(252).max().shift(1)
    df["dist_from_20d_high"] = (close - high_20d)  / high_20d.replace(0, np.nan)
    df["dist_from_52w_high"] = (close - high_252d) / high_252d.replace(0, np.nan)

    # ── Signal 4: Close above prev day / week high ────────────────────────────

    prev_day_high  = high.shift(1)
    prev_week_high = high.rolling(5).max().shift(1)
    df["above_prev_day_high"]  = (close > prev_day_high).astype(int)
    df["above_prev_week_high"] = (close > prev_week_high).astype(int)

    # ── Signal 5: Golden crossover + volume spike ─────────────────────────────

    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    df["ma20_vs_ma50"]  = close.rolling(20).mean() / ma50.replace(0, np.nan) - 1
    df["ma50_vs_ma200"] = ma50 / ma200.replace(0, np.nan) - 1
    df["volume_spike"]  = (volume > 2 * vol_ma20).astype(int)

    # ── Phase-2: Extended momentum returns ───────────────────────────────────

    df["return_60d"]  = close.pct_change(60)
    df["return_120d"] = close.pct_change(120)
    df["return_252d"] = close.pct_change(252)

    # ── Phase-2: Candle structure / Gap features ──────────────────────────────

    prev_close   = close.shift(1)
    candle_range = (high - low).replace(0, np.nan)

    df["gap_up_pct"]       = (open_ - prev_close) / prev_close.replace(0, np.nan)
    df["candle_body_ratio"]  = (close - open_).abs() / candle_range
    df["upper_wick_ratio"]   = (high - pd.concat([open_, close], axis=1).max(axis=1)) / candle_range
    df["lower_wick_ratio"]   = (pd.concat([open_, close], axis=1).min(axis=1) - low) / candle_range

    # ── Phase-2: Indicator velocity ──────────────────────────────────────────

    df["rsi_delta_5d"]      = df["rsi_14"] - df["rsi_14"].shift(5)
    df["macd_hist_delta"]   = df["macd_hist"] - df["macd_hist"].shift(1)
    df["bb_position_delta"] = df["bb_position"] - df["bb_position"].shift(5)

    # OBV slope: cumulative volume-weighted direction, normalised by avg volume
    obv_direction = np.sign(close.diff()).fillna(0)
    obv           = (obv_direction * volume).cumsum()
    vol_avg20     = vol_ma20.replace(0, np.nan)
    df["obv_slope"] = obv.diff(5) / (vol_avg20 * 5)   # change per day, volume-normalised

    # ── Phase-2: ATR / ADX ───────────────────────────────────────────────────

    atr          = _atr(high, low, close, period=14)
    atr_20d_avg  = atr.rolling(20).mean()
    df["atr_ratio"] = atr / atr_20d_avg.replace(0, np.nan)

    adx_vals, di_diff = _adx(high, low, close, period=14)
    df["adx"]    = adx_vals
    df["di_diff"] = di_diff

    # ── Phase-2: Stochastic %K/%D + Williams %R ──────────────────────────────

    low_14  = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    range_14 = (high_14 - low_14).replace(0, np.nan)

    df["stoch_k"]   = 100 * (close - low_14)   / range_14
    df["stoch_d"]   = df["stoch_k"].rolling(3).mean()
    df["williams_r"] = -100 * (high_14 - close) / range_14

    # ── Phase-2: Chaikin Money Flow (20) ─────────────────────────────────────

    mfm = ((close - low) - (high - close)) / candle_range   # money flow multiplier
    mfv = mfm * volume                                        # money flow volume
    df["cmf"] = mfv.rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)

    # ── Phase-2: Commodity Channel Index (20) ────────────────────────────────

    typical_price = (high + low + close) / 3
    tp_ma         = typical_price.rolling(20).mean()
    tp_mad        = typical_price.rolling(20).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    df["cci"] = (typical_price - tp_ma) / (0.015 * tp_mad.replace(0, np.nan))

    # ── Phase-2: Price acceleration ───────────────────────────────────────────

    avg_daily_ret_5d = df["return_5d"] / 5   # avg daily return over last 5 days
    df["price_accel"] = df["return_1d"] - avg_daily_ret_5d

    # ── Target ────────────────────────────────────────────────────────────────

    df["target"] = (close.shift(-1) > close).astype(int)

    # Note: earnings features are NOT included in model training — insufficient
    # historical data for UK stocks. Applied as post-processing in predictor.py.
    return df.dropna(subset=FEATURE_COLS + ["target"])


def _add_earnings_features(df: pd.DataFrame, earnings_dates: list) -> None:
    """
    Mutates df in-place to add earnings proximity features.

    earnings_dates : list of date-like objects (datetime.date or pd.Timestamp).
    All features are capped at 60 to keep scale bounded.
    """
    if not earnings_dates:
        df["days_to_earnings"]   = np.nan
        df["days_since_earnings"] = np.nan
        df["pre_earnings_flag"]  = 0
        df["post_earnings_flag"] = 0
        return

    import bisect
    eds = sorted(pd.Timestamp(e) for e in earnings_dates)

    days_to   = []
    days_since = []

    for ts in df.index:
        ts = pd.Timestamp(ts)
        idx = bisect.bisect_right(eds, ts)

        d_since = (ts - eds[idx - 1]).days if idx > 0 else np.nan
        d_to    = (eds[idx] - ts).days     if idx < len(eds) else np.nan

        days_since.append(min(d_since, 60) if not np.isnan(d_since) else np.nan)
        days_to.append(min(d_to, 60)       if not np.isnan(d_to)    else np.nan)

    df["days_to_earnings"]    = days_to
    df["days_since_earnings"] = days_since
    df["pre_earnings_flag"]   = (pd.Series(days_to, index=df.index) <= 5).astype(int)
    df["post_earnings_flag"]  = (pd.Series(days_since, index=df.index) <= 20).astype(int)


def add_cross_sectional_ranks(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame indexed by date with columns return_20d, return_60d,
    return_120d, return_252d (one row per stock per date), add cross-sectional
    percentile rank features.

    Call this AFTER concatenating all per-stock feature DataFrames.
    """
    combined = combined.copy()

    # 12-1 month momentum: 252d return minus most recent 21d return
    r252 = combined.get("return_252d", pd.Series(dtype=float))
    r21  = combined.get("return_20d",  pd.Series(dtype=float))
    combined["_return_12_1m"] = r252 - r21

    rank_map = {
        "cs_rank_1m":     "return_20d",
        "cs_rank_3m":     "return_60d",
        "cs_rank_6m":     "return_120d",
        "cs_rank_12_1m":  "_return_12_1m",
    }

    for rank_col, src_col in rank_map.items():
        if src_col in combined.columns:
            combined[rank_col] = combined.groupby(level=0)[src_col].rank(pct=True)
        else:
            combined[rank_col] = np.nan

    combined.drop(columns=["_return_12_1m"], errors="ignore", inplace=True)
    return combined
