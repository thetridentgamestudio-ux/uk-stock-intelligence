from pydantic import BaseModel
from typing import Optional


class StockOut(BaseModel):
    ticker: str
    name: str
    sector: Optional[str]
    index_name: Optional[str]


class PredictionOut(BaseModel):
    ticker: str
    name: str
    direction: str
    confidence: float
    prob_up: float
    technical_score: float
    volume_ratio: float
    return_5d: Optional[float] = None
    lbu_score: Optional[int] = None
    cs_rank_1m: Optional[float] = None
    market_regime: Optional[str] = None
    earnings_flag: Optional[str] = None
    explanation: Optional[str] = None


class DailyPredictionsOut(BaseModel):
    date: str
    top_gainers: list[PredictionOut]
    top_losers: list[PredictionOut]
    total_analysed: int


class PriceOut(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
