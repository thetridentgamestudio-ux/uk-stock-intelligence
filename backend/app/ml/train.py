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

from .features import (
    CS_RANK_FEATURES,
    FEATURE_COLS,
    add_cross_sectional_ranks,
    compute_features,
)

logger = logging.getLogger(__name__)


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

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Feature column resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_feature_cols(data: pd.DataFrame) -> list[str]:
    """
    Return the list of feature columns actually present and non-empty in `data`.
    CS-rank features are included if they were computed.
    """
    base = FEATURE_COLS.copy()

    # Cross-sectional ranks — include if computed
    for col in CS_RANK_FEATURES:
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

    # Fix class imbalance (markets have slight upward bias)
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    logger.info("Class balance — up: %d  down: %d  scale_pos_weight=%.3f",
                pos, neg, scale_pos_weight)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    xgb_accuracy = accuracy_score(y_test, model.predict(X_test))

    # ── LightGBM (trains alongside XGBoost; predictions are averaged) ─────────
    lgb_model = None
    if _LGB_AVAILABLE:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1,
        )
        lgb_model.fit(X_train, y_train)
        lgb_accuracy = accuracy_score(y_test, lgb_model.predict(X_test))

        # Ensemble accuracy (average probabilities)
        xgb_prob = model.predict_proba(X_test)[:, 1]
        lgb_prob  = lgb_model.predict_proba(X_test)[:, 1]
        ensemble_pred = ((xgb_prob + lgb_prob) / 2 >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, ensemble_pred)
        logger.info("XGB=%.1f%%  LGB=%.1f%%  Ensemble=%.1f%%",
                    xgb_accuracy * 100, lgb_accuracy * 100, accuracy * 100)
    else:
        accuracy = xgb_accuracy
        logger.info("LightGBM not available — using XGBoost only (%.1f%%)", accuracy * 100)

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

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump({
        "model":           model,
        "lgb_model":       lgb_model,
        "calibrator_xgb":  calibrator_xgb,
        "calibrator_lgb":  calibrator_lgb,
        "feature_cols":    feature_cols,
    }, model_path)
    logger.info("Model saved to %s (accuracy=%.1f%%, features=%d)",
                model_path, accuracy * 100, len(feature_cols))

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
