# UK Stock Intelligence

AI-powered daily prediction engine for FTSE 350 stocks. Predicts next-day price direction (up/down) using a 49-feature XGBoost model with cross-sectional momentum, market regime analysis, and earnings proximity signals.

> **Personal use** — not financial advice. Predictions are for research purposes only.

---

## What It Does

- Analyses **346 FTSE 350 stocks** every trading day after 4:30pm UK time
- Predicts whether each stock will close **higher or lower tomorrow**
- Shows **confidence %, LBU score, cross-sectional rank, earnings flags**
- Tracks **prediction accuracy** day by day (current real-world: ~59.5%)
- Displays a **Market Regime meter** (BULLISH / NEUTRAL / BEARISH) using VIX + FTSE momentum
- Generates **plain-English AI explanations** via Claude (Anthropic API)

---

## Features & Signals

### Technical Model (49 features)
| Group | Signals |
|---|---|
| Price momentum | 1d, 5d, 10d, 20d, 60d, 120d, 252d returns |
| RSI / MACD / Bollinger | Standard oscillators + velocity (5-day delta) |
| Long Build-Up | LBU score (0–4), consecutive up-closes, HHHL pattern, volume trend |
| Breakout | Distance from 20-day and 52-week highs |
| Moving average crossovers | MA20/50, MA50/200 (golden cross) |
| New indicators | ATR ratio, ADX, Stochastic %K/%D, Williams %R, CMF, CCI, OBV slope |
| Candle structure | Gap %, body ratio, upper/lower wick ratios |
| Cross-sectional ranks | 1m, 3m, 6m, 12-1m momentum rank vs full FTSE 350 universe |

### Market Context
- **VIX + FTSE momentum** → BULLISH / NEUTRAL / BEARISH regime → confidence nudge
- **Earnings proximity** → pre-earnings uncertainty penalty, post-earnings PEAD boost
- **Analyst targets** (Phase 2 roadmap)

---

## Setup

### Prerequisites
- Python 3.11+
- macOS / Linux
- `brew install libomp` (macOS, for XGBoost)

### Install
```bash
git clone https://github.com/YOUR_USERNAME/uk-stock-intelligence.git
cd uk-stock-intelligence

pip3 install -r requirements.txt

cp .env.example .env
# Edit .env — set DATABASE_URL to absolute path, add ANTHROPIC_API_KEY
```

### First Run
```bash
# 1. Build stock list (FTSE 100 + 250 from Wikipedia)
python3 scripts/update_stock_list.py

# 2. Fetch 5 years of historical prices (takes ~30 min for 350 stocks)
python3 scripts/fetch_historical.py

# 3. Build earnings calendar cache
python3 scripts/update_earnings_cache.py

# 4. Train the XGBoost model
python3 scripts/train_model.py

# 5. Run daily pipeline (generates predictions for tomorrow)
python3 scripts/run_daily_pipeline.py
```

### Start the API + Dashboard
```bash
# Terminal 1 — API server
python3 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

# Terminal 2 (or just open the file)
open frontend/index.html
```

---

## Daily Usage

The pipeline runs **automatically at 17:15 Mon–Fri (London time)** via APScheduler when the API is running.

To run manually after market close (4:30pm UK):
```bash
python3 scripts/run_daily_pipeline.py
```

Then refresh `frontend/index.html` in your browser.

---

## Project Structure

```
uk-stock-intelligence/
├── backend/
│   └── app/
│       ├── api/routes/          # FastAPI endpoints
│       ├── data/stocks.json     # 350 FTSE tickers (auto-generated)
│       ├── ml/
│       │   ├── features.py      # 45 technical features + cross-sectional ranks
│       │   └── train.py         # XGBoost training + walk-forward CV
│       ├── models/              # SQLAlchemy + Pydantic schemas
│       └── services/
│           ├── predictor.py     # Prediction engine
│           ├── market_sentiment.py  # VIX + FTSE regime
│           ├── earnings_fetcher.py  # Earnings calendar cache
│           └── accuracy_checker.py # Daily prediction tracking
├── frontend/
│   └── index.html               # Dashboard (no build step — pure HTML)
├── models/                      # Saved model (gitignored, train locally)
├── scripts/
│   ├── run_daily_pipeline.py    # Main daily script
│   ├── train_model.py           # Train + walk-forward evaluate
│   ├── fetch_historical.py      # Seed 5yr price history
│   ├── update_stock_list.py     # Refresh FTSE 350 list
│   └── update_earnings_cache.py # Refresh earnings dates
└── .env.example
```

---

## Walk-Forward Accuracy Evaluation

```bash
python3 scripts/train_model.py --evaluate
```

Runs 5-fold rolling cross-validation (train 2yr → test 3mo → slide) for a statistically reliable accuracy estimate. More honest than a single backtest split.

---

## Tech Stack

- **Data**: Yahoo Finance (yfinance) — free, no API key
- **ML**: XGBoost with class-balance weighting
- **API**: FastAPI + SQLite (SQLAlchemy)
- **AI explanations**: Anthropic Claude (haiku-4-5)
- **Scheduler**: APScheduler (runs pipeline at 17:15 daily)
- **Frontend**: Vanilla HTML + Tailwind CDN

---

## Roadmap

- [ ] LightGBM ensemble (XGB + LGB averaged)
- [ ] FinBERT sentiment on RNS announcements
- [ ] Sector relative strength features
- [ ] Analyst price target upside feature
- [ ] Probability calibration (Platt scaling)
- [ ] Google Trends weekly signal

---

*Built with Claude Code · Not financial advice*
