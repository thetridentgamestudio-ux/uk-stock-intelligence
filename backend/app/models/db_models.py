from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float,
    Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.sql import func
from ..database import Base


class Stock(Base):
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), unique=True, nullable=False)
    name = Column(String(200))
    sector = Column(String(100))
    index_name = Column(String(50))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class StockPrice(Base):
    __tablename__ = "stock_prices"
    __table_args__ = (UniqueConstraint("ticker", "date"),)

    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), nullable=False)
    prediction_date = Column(Date, nullable=False)
    target_date = Column(Date, nullable=False)
    direction = Column(String(10))
    confidence = Column(Float)
    predicted_change_pct = Column(Float)
    explanation = Column(Text)
    sentiment_score = Column(Float)
    technical_score = Column(Float)
    volume_score = Column(Float)
    actual_change_pct = Column(Float, nullable=True)
    was_correct = Column(Boolean, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class NewsArticle(Base):
    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(20))
    title = Column(String(500))
    source = Column(String(100))
    url = Column(String(500))
    published_at = Column(DateTime(timezone=True))
    sentiment_score = Column(Float)
    sentiment_label = Column(String(20))
    is_rns = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
