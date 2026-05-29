"""Create all database tables and seed the stock list."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from sqlalchemy.orm import sessionmaker

from backend.app.database import Base, engine
from backend.app.models.db_models import NewsArticle, Prediction, Stock, StockPrice
from backend.app.services.data_fetcher import FTSE_STOCKS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    Base.metadata.create_all(bind=engine)
    logger.info("Tables created.")

    Session = sessionmaker(bind=engine)
    db = Session()

    added = 0
    for ticker, (name, sector, index_name) in FTSE_STOCKS.items():
        if not db.query(Stock).filter(Stock.ticker == ticker).first():
            db.add(Stock(ticker=ticker, name=name, sector=sector, index_name=index_name))
            added += 1

    db.commit()
    db.close()
    logger.info("Seeded %d stocks into the stocks table.", added)


if __name__ == "__main__":
    main()
