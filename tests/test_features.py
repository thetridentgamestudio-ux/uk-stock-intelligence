from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from backend.app.ml.features import FEATURE_COLS, compute_features


@pytest.fixture
def price_df():
    """60 days of synthetic OHLCV data with realistic price movement."""
    n = 60
    np.random.seed(42)
    close = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n)))
    dates = pd.to_datetime([date.today() - timedelta(days=n - i) for i in range(n)])
    return pd.DataFrame(
        {
            "open": close * np.random.uniform(0.995, 1.005, n),
            "high": close * np.random.uniform(1.001, 1.020, n),
            "low": close * np.random.uniform(0.980, 0.999, n),
            "close": close,
            "volume": np.random.randint(500_000, 5_000_000, n).astype(float),
        },
        index=dates,
    )


def test_all_feature_columns_present(price_df):
    features = compute_features(price_df)
    for col in FEATURE_COLS:
        assert col in features.columns, f"Missing feature column: {col}"


def test_no_inf_or_nan_in_features(price_df):
    features = compute_features(price_df)
    assert not features[FEATURE_COLS].isin([np.inf, -np.inf]).any().any()
    assert not features[FEATURE_COLS].isna().any().any()


def test_rsi_bounded_0_100(price_df):
    features = compute_features(price_df)
    assert features["rsi_14"].between(0, 100).all(), "RSI must be in [0, 100]"


def test_target_is_binary(price_df):
    features = compute_features(price_df)
    assert set(features["target"].unique()).issubset({0, 1})


def test_returns_non_empty(price_df):
    features = compute_features(price_df)
    assert len(features) > 0


def test_fewer_rows_than_input(price_df):
    """dropna should trim some leading rows."""
    features = compute_features(price_df)
    assert len(features) < len(price_df)


def test_volume_ratio_positive(price_df):
    features = compute_features(price_df)
    assert (features["volume_ratio"] > 0).all()


def test_too_little_data_returns_empty():
    """With only 5 rows, there should not be enough for any features."""
    tiny = pd.DataFrame(
        {
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.5] * 5,
            "volume": [1_000_000.0] * 5,
        },
        index=pd.date_range("2024-01-01", periods=5),
    )
    assert compute_features(tiny).empty
