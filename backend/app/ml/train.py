import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sqlalchemy import text
from sqlalchemy.orm import Session
from xgboost import XGBClassifier

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

try:
    import catboost as cb
    _CB_AVAILABLE = True
except ImportError:
    _CB_AVAILABLE = False

from .features import (
    CS_RANK_FEATURES,
    FEATURE_COLS,
    _MOMENTUM_FEATURES,
    add_cross_sectional_ranks,
    compute_features,
)
from .regime_labels import (
    add_regime_labels,
    MIN_REGIME_ROWS,
    REGIME_LABELS,
)

logger = logging.getLogger(__name__)


def _train_single_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    weights: np.ndarray,
    scale_pos_weight: float,
    label: str = "",
    hyperparams: dict | None = None,
) -> tuple:
    """
    Train XGBoost + LightGBM + CatBoost ensemble with given data and weights.
    Supports custom hyperparameters for tuning.
    Returns (xgb_model, lgb_model, cb_model).
    """
    if hyperparams is None:
        hyperparams = {}

    # Build params with defaults
    lr = hyperparams.get("learning_rate", 0.04)
    depth = hyperparams.get("max_depth", 4)
    subsample = hyperparams.get("subsample", 0.8)
    colsample = hyperparams.get("colsample_bytree", 0.8)

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=depth,
        learning_rate=lr,
        subsample=subsample,
        colsample_bytree=colsample,
        min_child_weight=3,
        gamma=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_train, y_train, sample_weight=weights, verbose=False)

    # LightGBM
    lgb_m = None
    if _LGB_AVAILABLE:
        lgb_m = lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=depth,
            learning_rate=lr,
            subsample=subsample,
            colsample_bytree=colsample,
            min_child_samples=20,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1,
        )
        lgb_m.fit(X_train, y_train, sample_weight=weights)

    # CatBoost — better at non-linear relationships + categorical features
    cb_m = None
    if _CB_AVAILABLE:
        cb_m = cb.CatBoostClassifier(
            iterations=400,
            depth=depth,
            learning_rate=lr,
            subsample=subsample,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=0,
            thread_count=-1,  # Use all cores
        )
        cb_m.fit(X_train, y_train, sample_weight=weights)

    logger.info(
        "Trained %sensemble (XGB + LGB + %s) — %d rows | scale_pos_weight=%.3f | lr=%.3f depth=%d",
        f"{label} " if label else "",
        "CB" if cb_m is not None else "no CB",
        len(X_train), scale_pos_weight, lr, depth,
    )
    return xgb, lgb_m, cb_m


def _compute_recency_weights(dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Assign sample weights based on how recent each date is.
    Recent observations are more representative of current market regime.

    Weight schedule (expanding window with soft decay):
      Last 6 months    → 2.0×
      6–12 months ago  → 1.5×
      12–24 months ago → 1.2×
      Older            → 1.0×
    """
    today  = pd.Timestamp.now()
    months = ((today - dates).days / 30.44).round(1)

    weights = np.where(months <= 6,  2.0,
              np.where(months <= 12, 1.5,
              np.where(months <= 24, 1.2,
                                     1.0)))
    return weights.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_training_data(db: Session) -> pd.DataFrame:
    """
    Load all OHLCV rows from the DB, compute per-stock features, then add
    cross-sectional rank features (computed per date across all stocks).
    """
    rows = db.execute(
        text(
            "SELECT ticker, date, open, high, low, close, volume "
            "FROM stock_prices ORDER BY ticker, date"
        )
    ).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["ticker", "date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])

    stock_frames = []
    for ticker, group in df.groupby("ticker"):
        group = group.set_index("date").drop(columns=["ticker"])
        feats = compute_features(group)
        if len(feats) >= 30:
            feats["ticker"] = ticker
            stock_frames.append(feats)

    if not stock_frames:
        return pd.DataFrame()

    combined = pd.concat(stock_frames)

    # Add cross-sectional rank features (per-date percentile rank across universe)
    combined = add_cross_sectional_ranks(combined)

    # Add momentum-relative-to-peers features
    combined = _add_momentum_features(combined)

    return combined


def _add_momentum_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sector_momentum_delta and market_momentum_delta for each row.
    These are relative momentum signals: stock return minus peer/market return.
    """
    from ..services.data_fetcher import FTSE_STOCKS
    import yfinance as yf

    data = data.copy()

    # Fetch FTSE market return for each date
    try:
        ftse_hist = yf.Ticker("^FTSE").history(start="2020-01-01")["Close"]
        ftse_ret = ftse_hist.pct_change(1)
        ftse_ret.index = pd.to_datetime(ftse_ret.index.date)
    except Exception as exc:
        logger.warning("Could not fetch FTSE returns for market momentum: %s", exc)
        ftse_ret = pd.Series(dtype=float)

    # Build sector mapping
    sector_map = {}
    for ticker, info in FTSE_STOCKS.items():
        if isinstance(info, dict):
            sector = info.get("sector", "Other")
        else:
            sector = info[1] if len(info) > 1 else "Other"
        sector_map[ticker] = sector

    # Compute sector averages per date and fill momentum features
    for date in data.index.unique():
        # Get all rows for this date
        date_mask = (data.index == date)
        date_data = data[date_mask].copy()

        if len(date_data) == 0:
            continue

        # Compute sector averages for this date
        sector_avgs = {}
        for sector in set(sector_map.values()):
            sector_tickers = [t for t in sector_map if sector_map[t] == sector]
            sector_ret_vals = date_data[date_data["ticker"].isin(sector_tickers)]["return_1d"]
            if len(sector_ret_vals) > 0:
                sector_avgs[sector] = sector_ret_vals.mean()
            else:
                sector_avgs[sector] = 0.0

        # Fill momentum features for all stocks on this date
        for idx, row in date_data.iterrows():
            ticker = row.get("ticker")
            try:
                stock_ret = float(row["return_1d"])
            except (ValueError, TypeError, KeyError):
                continue

            # Sector momentum
            if ticker in sector_map and not np.isnan(stock_ret):
                sector = sector_map[ticker]
                sector_avg = sector_avgs.get(sector, 0.0)
                data.loc[idx, "sector_momentum_delta"] = (stock_ret - sector_avg)

            # Market momentum
            if not np.isnan(stock_ret) and date in ftse_ret.index:
                market_ret = float(ftse_ret.loc[date])
                data.loc[idx, "market_momentum_delta"] = (stock_ret - market_ret)

    logger.info("Computed momentum-relative-to-peers features")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Feature column resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_feature_cols(data: pd.DataFrame) -> list[str]:
    """
    Return the list of feature columns actually present and non-empty in `data`.
    CS-rank features and momentum features are included if they were computed.
    """
    base = FEATURE_COLS.copy()

    # Cross-sectional ranks — include if computed
    for col in CS_RANK_FEATURES:
        if col in data.columns:
            base.append(col)

    # Momentum features — include if computed
    for col in _MOMENTUM_FEATURES:
        if col in data.columns:
            base.append(col)

    # Remove any columns that somehow ended up all-NaN
    available = [c for c in base if c in data.columns and data[c].notna().any()]
    return available


# ─────────────────────────────────────────────────────────────────────────────
# Production model training (single walk-forward split)
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    db: Session,
    model_path: str = "models/xgboost_model.pkl",
) -> dict:
    """
    Train XGBoost using a walk-forward split (first 80% of dates = train).
    Returns evaluation metrics dict.
    """
    data = load_training_data(db)

    if data.empty or len(data) < 200:
        raise ValueError(
            f"Only {len(data)} training rows — need at least 200. "
            "Run fetch_historical.py first."
        )

    feature_cols = _resolve_feature_cols(data)

    # Drop rows where any feature is NaN (cross-sectional ranks can be NaN for early dates)
    data_clean = data.dropna(subset=feature_cols + ["target"])

    all_dates  = sorted(data_clean.index.unique())
    split_date = all_dates[int(len(all_dates) * 0.8)]

    train = data_clean[data_clean.index < split_date]
    test  = data_clean[data_clean.index >= split_date]

    X_train, y_train = train[feature_cols], train["target"]
    X_test,  y_test  = test[feature_cols],  test["target"]

    # ── Recency weighting — recent data matters more than 3-year-old data ──────
    # Markets shift regime; a model trained equally on 2021 and 2026 data
    # is partially fitted on a dead regime. Upweight recent observations.
    #
    # Weight schedule (research-backed: arXiv:2601.08896):
    #   Last 6 months:    2.0×  (current regime — highest priority)
    #   6–12 months ago:  1.5×  (recent history — still very relevant)
    #   12–24 months ago: 1.2×  (medium-term — some regime drift)
    #   Older than 2 yrs: 1.0×  (baseline — long-term structural patterns)
    train_weights = _compute_recency_weights(train.index)
    logger.info("Recency weights: min=%.2f max=%.2f mean=%.2f — recent data upweighted",
                train_weights.min(), train_weights.max(), train_weights.mean())

    # Fix class imbalance (markets have slight upward bias)
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    logger.info("Class balance — up: %d  down: %d  scale_pos_weight=%.3f",
                pos, neg, scale_pos_weight)

    # ── Full-universe model (baseline + fallback) ─────────────────────────────
    model, lgb_model, cb_model = _train_single_ensemble(
        X_train, y_train, train_weights, scale_pos_weight, label="full"
    )
    xgb_accuracy = accuracy_score(y_test, model.predict(X_test))

    xgb_prob = model.predict_proba(X_test)[:, 1]
    lgb_prob = None
    cb_prob = None

    if lgb_model is not None:
        lgb_accuracy = accuracy_score(y_test, lgb_model.predict(X_test))
        lgb_prob = lgb_model.predict_proba(X_test)[:, 1]
    else:
        lgb_accuracy = 0.0

    if cb_model is not None:
        cb_accuracy = accuracy_score(y_test, cb_model.predict(X_test))
        cb_prob = cb_model.predict_proba(X_test)[:, 1]
    else:
        cb_accuracy = 0.0

    # Ensemble: average all available models
    if lgb_prob is not None and cb_prob is not None:
        ensemble_prob = (xgb_prob + lgb_prob + cb_prob) / 3
    elif lgb_prob is not None:
        ensemble_prob = (xgb_prob + lgb_prob) / 2
    else:
        ensemble_prob = xgb_prob

    ensemble_pred = (ensemble_prob >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, ensemble_pred)

    if lgb_prob is not None and cb_prob is not None:
        logger.info("Full model — XGB=%.1f%%  LGB=%.1f%%  CB=%.1f%%  Ensemble=%.1f%%",
                    xgb_accuracy * 100, lgb_accuracy * 100, cb_accuracy * 100, accuracy * 100)
    elif lgb_prob is not None:
        logger.info("Full model — XGB=%.1f%%  LGB=%.1f%%  Ensemble=%.1f%%",
                    xgb_accuracy * 100, lgb_accuracy * 100, accuracy * 100)
    else:
        logger.info("Full model — XGB=%.1f%% (LGB/CB unavailable)", accuracy * 100)

    # ── Regime-conditional models ─────────────────────────────────────────────
    # Label each training row with the market regime on that date.
    # Train a separate XGB+LGB ensemble for each regime.
    # At prediction time, select the model matching today's regime.
    logger.info("Building regime-conditional models (BULLISH / NEUTRAL / BEARISH)...")
    train_labelled = add_regime_labels(train)
    regime_models: dict[str, dict] = {}

    for regime in REGIME_LABELS:
        regime_train = train_labelled[train_labelled["regime"] == regime]
        n = len(regime_train)

        if n < MIN_REGIME_ROWS:
            logger.warning(
                "Regime %s: only %d rows (need %d) — skipping, will use full model",
                regime, n, MIN_REGIME_ROWS,
            )
            continue

        rX = regime_train[feature_cols]
        ry = regime_train["target"]
        rw = _compute_recency_weights(regime_train.index)

        neg_r = int((ry == 0).sum())
        pos_r = int((ry == 1).sum())
        spw_r = neg_r / pos_r if pos_r > 0 else 1.0

        rxgb, rlgb, rcb = _train_single_ensemble(rX, ry, rw, spw_r, label=regime)

        # Evaluate on test set rows matching this regime
        test_labelled = add_regime_labels(test)
        regime_test = test_labelled[test_labelled["regime"] == regime]
        if len(regime_test) >= 50:
            rX_test = regime_test[feature_cols]
            ry_test = regime_test["target"]
            r_xgb_prob = rxgb.predict_proba(rX_test)[:, 1]

            r_lgb_prob = None
            r_cb_prob = None

            if rlgb:
                r_lgb_prob = rlgb.predict_proba(rX_test)[:, 1]
            if rcb:
                r_cb_prob = rcb.predict_proba(rX_test)[:, 1]

            # Ensemble
            if r_lgb_prob is not None and r_cb_prob is not None:
                r_pred_prob = (r_xgb_prob + r_lgb_prob + r_cb_prob) / 3
            elif r_lgb_prob is not None:
                r_pred_prob = (r_xgb_prob + r_lgb_prob) / 2
            else:
                r_pred_prob = r_xgb_prob

            r_pred = (r_pred_prob >= 0.5).astype(int)
            r_acc = accuracy_score(ry_test, r_pred)
            logger.info(
                "Regime %s model: %d train / %d test → %.1f%% accuracy",
                regime, n, len(regime_test), r_acc * 100,
            )
        else:
            logger.info("Regime %s model: %d train rows", regime, n)

        regime_models[regime] = {"xgb": rxgb, "lgb": rlgb, "cb": rcb}

    logger.info(
        "Regime models trained: %s",
        ", ".join(regime_models.keys()) or "NONE (fell back to full model)",
    )

    # ── Platt calibration (sigmoid) — fits A,B params on held-out test set ──────
    # Maps raw model probabilities to calibrated probabilities so that
    # a 65% prediction truly means ~65% historical accuracy (enables Kelly sizing)
    from sklearn.linear_model import LogisticRegression

    def _fit_platt(mdl, X_cal, y_cal):
        """Fit a Platt scaler: logistic regression on raw model probabilities."""
        try:
            raw_probs = mdl.predict_proba(X_cal)[:, 1].reshape(-1, 1)
            platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
            platt.fit(raw_probs, y_cal)
            return platt
        except Exception as exc:
            logger.warning("Platt calibration failed: %s", exc)
            return None

    calibrator_xgb = _fit_platt(model, X_test, y_test)
    calibrator_lgb  = _fit_platt(lgb_model, X_test, y_test) if lgb_model else None
    if calibrator_xgb:
        logger.info("Platt calibration fitted on %d samples", len(X_test))

    # ── Stacking meta-learner (trained on OOF predictions) ─────────────────
    meta_learner = None
    if _LGB_AVAILABLE and lgb_model is not None:
        # Generate out-of-fold predictions on test set for meta-learner training
        from sklearn.linear_model import LogisticRegression

        xgb_oof = model.predict_proba(X_test)[:, 1]
        lgb_oof = lgb_model.predict_proba(X_test)[:, 1]
        meta_features = np.column_stack([xgb_oof, lgb_oof])

        meta_learner = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
        meta_learner.fit(meta_features, y_test)

        # Evaluate meta-learner
        meta_pred = meta_learner.predict(meta_features)
        meta_accuracy = accuracy_score(y_test, meta_pred)
        logger.info("Stacking meta-learner: %d OOF samples, accuracy=%.1f%%",
                    len(X_test), meta_accuracy * 100)

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump({
        "model":           model,
        "lgb_model":       lgb_model,
        "regime_models":   regime_models,   # dict[regime → {xgb, lgb}]
        "meta_learner":    meta_learner,
        "calibrator_xgb":  calibrator_xgb,
        "calibrator_lgb":  calibrator_lgb,
        "feature_cols":    feature_cols,
    }, model_path)
    logger.info(
        "Model saved to %s (accuracy=%.1f%%, features=%d, "
        "regime models=%s, meta-learner=%s)",
        model_path, accuracy * 100, len(feature_cols),
        list(regime_models.keys()) or "none",
        "YES" if meta_learner else "NO",
    )

    return {
        "accuracy":      accuracy,
        "train_samples": len(X_train),
        "test_samples":  len(X_test),
        "split_date":    str(split_date.date()),
        "feature_count": len(feature_cols),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward cross-validation (for reliable accuracy measurement)
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_evaluate(
    db: Session,
    n_folds: int = 5,
    train_years: int = 2,
    test_months: int = 3,
) -> dict:
    """
    Rolling walk-forward evaluation — gives a reliable accuracy estimate.

    Each fold:
      - Train on `train_years` years of data
      - Test on the next `test_months` months
      - Slide the window forward by `test_months`

    Returns dict with per-fold accuracy and overall mean/std.
    """
    data = load_training_data(db)

    if data.empty:
        raise ValueError("No training data available.")

    feature_cols = _resolve_feature_cols(data)
    data_clean   = data.dropna(subset=feature_cols + ["target"])
    all_dates    = sorted(data_clean.index.unique())

    # Build fold boundaries
    fold_results = []
    test_size_td = pd.DateOffset(months=test_months)
    train_size_td = pd.DateOffset(years=train_years)

    # Start after we have enough training data
    first_train_end = all_dates[0] + train_size_td
    current_test_start = first_train_end

    while True:
        test_end = current_test_start + test_size_td
        if test_end > all_dates[-1]:
            break

        train_start = current_test_start - train_size_td
        train_mask = (data_clean.index >= train_start) & (data_clean.index < current_test_start)
        test_mask  = (data_clean.index >= current_test_start) & (data_clean.index < test_end)

        train_fold = data_clean[train_mask]
        test_fold  = data_clean[test_mask]

        if len(train_fold) < 500 or len(test_fold) < 50:
            current_test_start = test_end
            continue

        X_tr, y_tr = train_fold[feature_cols], train_fold["target"]
        X_te, y_te = test_fold[feature_cols],  test_fold["target"]

        neg = int((y_tr == 0).sum())
        pos = int((y_tr == 1).sum())
        spw = neg / pos if pos > 0 else 1.0

        m = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        m.fit(X_tr, y_tr, verbose=False)
        acc = accuracy_score(y_te, m.predict(X_te))

        fold_results.append({
            "train_start":  str(train_start.date()),
            "train_end":    str(current_test_start.date()),
            "test_start":   str(current_test_start.date()),
            "test_end":     str(test_end.date()),
            "train_n":      len(X_tr),
            "test_n":       len(X_te),
            "accuracy":     round(acc * 100, 2),
        })

        logger.info(
            "Fold %d: train [%s → %s]  test [%s → %s]  acc=%.1f%%",
            len(fold_results),
            fold_results[-1]["train_start"], fold_results[-1]["train_end"],
            fold_results[-1]["test_start"],  fold_results[-1]["test_end"],
            acc * 100,
        )

        current_test_start = test_end
        if len(fold_results) >= n_folds:
            break

    if not fold_results:
        return {"error": "Not enough data to form folds."}

    accs = [f["accuracy"] for f in fold_results]
    return {
        "folds":        fold_results,
        "mean_accuracy": round(float(np.mean(accs)), 2),
        "std_accuracy":  round(float(np.std(accs)), 2),
        "n_folds":       len(fold_results),
        "feature_count": len(feature_cols),
    }
