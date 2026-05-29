from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backend.app.services.data_fetcher import FTSE_STOCKS, fetch_prices, save_prices


# ── Stock list sanity checks ──────────────────────────────────────────────────

def test_ftse_stocks_not_empty():
    assert len(FTSE_STOCKS) > 0


def test_all_tickers_end_with_dot_l():
    bad = [t for t in FTSE_STOCKS if not t.endswith(".L")]
    assert bad == [], f"Tickers missing .L suffix: {bad}"


def test_all_stocks_have_three_fields():
    for ticker, info in FTSE_STOCKS.items():
        assert len(info) == 3, f"{ticker} should have (name, sector, index)"


# ── fetch_prices ──────────────────────────────────────────────────────────────

def _mock_history_df():
    return pd.DataFrame(
        {
            "Open": [500.0],
            "High": [510.0],
            "Low": [495.0],
            "Close": [505.0],
            "Volume": [2_000_000.0],
        },
        index=pd.to_datetime([date.today()]),
    )


@patch("yfinance.Ticker")
def test_fetch_prices_returns_lowercase_columns(mock_ticker_cls):
    mock_ticker_cls.return_value.history.return_value = _mock_history_df()
    df = fetch_prices("HSBA.L", date.today() - timedelta(days=1), date.today())
    assert not df.empty
    assert set(df.columns) == {"open", "high", "low", "close", "volume"}


@patch("yfinance.Ticker")
def test_fetch_prices_empty_on_no_data(mock_ticker_cls):
    mock_ticker_cls.return_value.history.return_value = pd.DataFrame()
    df = fetch_prices("FAKE.L", date.today() - timedelta(days=1), date.today())
    assert df.empty


@patch("yfinance.Ticker")
def test_fetch_prices_returns_empty_on_exception(mock_ticker_cls):
    mock_ticker_cls.return_value.history.side_effect = Exception("network error")
    df = fetch_prices("HSBA.L", date.today() - timedelta(days=1), date.today())
    assert df.empty


# ── save_prices ───────────────────────────────────────────────────────────────

def _single_day_df(price_date: date = None) -> pd.DataFrame:
    price_date = price_date or date.today()
    return pd.DataFrame(
        {"open": [500.0], "high": [510.0], "low": [495.0], "close": [505.0], "volume": [2e6]},
        index=[price_date],
    )


def test_save_prices_returns_count(db):
    df = _single_day_df(date(2024, 1, 2))
    saved = save_prices(db, "HSBA.L", df)
    assert saved == 1


def test_save_prices_skips_duplicate(db):
    df = _single_day_df(date(2024, 1, 3))
    save_prices(db, "HSBA.L", df)
    saved_again = save_prices(db, "HSBA.L", df)
    assert saved_again == 0


def test_save_prices_empty_df_returns_zero(db):
    saved = save_prices(db, "HSBA.L", pd.DataFrame())
    assert saved == 0
