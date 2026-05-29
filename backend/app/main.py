import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import accuracy, predictions, stocks
from .services.scheduler import lifespan

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

app = FastAPI(
    title="UK Stock Intelligence API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(stocks.router, prefix="/api")
app.include_router(predictions.router, prefix="/api")
app.include_router(accuracy.router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}
