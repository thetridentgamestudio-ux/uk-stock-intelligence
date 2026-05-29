from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ── /api/stocks ───────────────────────────────────────────────────────────────

def test_stocks_endpoint_returns_list():
    response = client.get("/api/stocks/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


# ── /api/predictions/daily ────────────────────────────────────────────────────

MOCK_PREDICTIONS = [
    {
        "ticker": "HSBA.L",
        "name": "HSBC Holdings",
        "direction": "BULLISH",
        "confidence": 65.0,
        "prob_up": 65.0,
        "technical_score": 55.0,
        "volume_ratio": 1.2,
        "return_5d": 1.5,
    },
    {
        "ticker": "BP.L",
        "name": "BP",
        "direction": "BEARISH",
        "confidence": 70.0,
        "prob_up": 30.0,
        "technical_score": 42.0,
        "volume_ratio": 0.9,
        "return_5d": -2.1,
    },
]


@patch("backend.app.api.routes.predictions.run_predictions")
def test_daily_predictions_structure(mock_run):
    mock_run.return_value = MOCK_PREDICTIONS
    response = client.get("/api/predictions/daily")
    assert response.status_code == 200
    data = response.json()
    assert "top_gainers" in data
    assert "top_losers" in data
    assert "date" in data
    assert "total_analysed" in data
    assert data["total_analysed"] == 2


@patch("backend.app.api.routes.predictions.run_predictions")
def test_daily_predictions_separates_directions(mock_run):
    mock_run.return_value = MOCK_PREDICTIONS
    data = client.get("/api/predictions/daily").json()
    assert all(p["direction"] == "BULLISH" for p in data["top_gainers"])
    assert all(p["direction"] == "BEARISH" for p in data["top_losers"])


@patch("backend.app.api.routes.predictions.run_predictions")
def test_daily_predictions_503_when_no_model(mock_run):
    mock_run.return_value = []
    response = client.get("/api/predictions/daily")
    assert response.status_code == 503


# ── /api/predictions/{ticker}/explain ────────────────────────────────────────

@patch("backend.app.api.routes.predictions.fetch_news_headlines", return_value=[])
@patch("backend.app.api.routes.predictions.generate_explanation", return_value="Test explanation.")
@patch("backend.app.api.routes.predictions.run_predictions")
def test_explain_returns_explanation(mock_run, mock_explain, mock_news):
    mock_run.return_value = [MOCK_PREDICTIONS[0]]
    response = client.get("/api/predictions/HSBA.L/explain")
    assert response.status_code == 200
    assert response.json()["explanation"] == "Test explanation."


@patch("backend.app.api.routes.predictions.run_predictions")
def test_explain_404_for_unknown_ticker(mock_run):
    mock_run.return_value = MOCK_PREDICTIONS
    response = client.get("/api/predictions/FAKE/explain")
    assert response.status_code == 404
