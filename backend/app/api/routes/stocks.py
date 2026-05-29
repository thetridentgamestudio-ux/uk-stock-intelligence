from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from ...database import get_db
from ...models.schemas import PriceOut, StockOut

router = APIRouter(prefix="/stocks", tags=["stocks"])


@router.get("/", response_model=list[StockOut])
def list_stocks(db: Session = Depends(get_db)):
    rows = db.execute(
        text("SELECT ticker, name, sector, index_name FROM stocks ORDER BY ticker")
    ).fetchall()
    return [{"ticker": r[0], "name": r[1], "sector": r[2], "index_name": r[3]} for r in rows]


@router.get("/{ticker}/prices", response_model=list[PriceOut])
def get_prices(ticker: str, days: int = 30, db: Session = Depends(get_db)):
    rows = db.execute(
        text(
            "SELECT date, open, high, low, close, volume FROM stock_prices "
            "WHERE ticker = :t ORDER BY date DESC LIMIT :d"
        ),
        {"t": ticker.upper(), "d": days},
    ).fetchall()
    return [
        {"date": str(r[0]), "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]}
        for r in rows
    ]
