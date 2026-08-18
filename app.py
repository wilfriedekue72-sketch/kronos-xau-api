"""
Kronos XAU/USD Forecast API
----------------------------
Wraps the Kronos foundation model (NeoQuasar/Kronos-*) behind a small FastAPI
service so it can be called from a Netlify function. Designed to run on a
free Hugging Face Space (Docker SDK, CPU basic tier).

Endpoints:
  GET  /health              -> liveness + which model is loaded
  POST /forecast             -> given historical OHLCV candles, return predicted
                                 future candles (one or more sampled paths)

This service holds NO API keys, does NOT place trades, and does NOT know
about your account/risk settings. It only turns candles -> forecasted candles.
All risk/signal logic lives in the Netlify function.
"""

import os
import time
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model import Kronos, KronosTokenizer, KronosPredictor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("kronos-api")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# kronos-mini  = 4.1M params  -> fastest, lowest quality, context 2048
# kronos-small = 24.7M params -> good balance, context 512 (default)
# kronos-base  = 102.3M params -> best quality, noticeably slower on free CPU
MODEL_CHOICE = os.environ.get("KRONOS_MODEL", "kronos-mini")

MODEL_REGISTRY = {
    "kronos-mini": {
        "model_id": "NeoQuasar/Kronos-mini",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
        "max_context": 2048,
    },
    "kronos-small": {
        "model_id": "NeoQuasar/Kronos-small",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
    "kronos-base": {
        "model_id": "NeoQuasar/Kronos-base",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
}

if MODEL_CHOICE not in MODEL_REGISTRY:
    raise RuntimeError(f"Unknown KRONOS_MODEL '{MODEL_CHOICE}', choose one of {list(MODEL_REGISTRY)}")

API_KEY = os.environ.get("API_KEY", "")  # optional shared secret, checked on /forecast

# ---------------------------------------------------------------------------
# App + model loading (happens once at startup / cold start)
# ---------------------------------------------------------------------------
app = FastAPI(title="Kronos XAU/USD Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Netlify domain once deployed
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor: Optional[KronosPredictor] = None
_load_error: Optional[str] = None


def get_predictor() -> KronosPredictor:
    global _predictor, _load_error
    if _predictor is not None:
        return _predictor
    if _load_error is not None:
        raise HTTPException(status_code=503, detail=f"Model failed to load: {_load_error}")

    cfg = MODEL_REGISTRY[MODEL_CHOICE]
    log.info("Loading %s ...", cfg["model_id"])
    t0 = time.time()
    try:
        tokenizer = KronosTokenizer.from_pretrained(cfg["tokenizer_id"])
        model = Kronos.from_pretrained(cfg["model_id"])
        _predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=cfg["max_context"])
        log.info("Loaded %s in %.1fs", cfg["model_id"], time.time() - t0)
    except Exception as e:  # noqa: BLE001
        _load_error = str(e)
        log.exception("Model load failed")
        raise HTTPException(status_code=503, detail=f"Model failed to load: {e}") from e
    return _predictor


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class Candle(BaseModel):
    timestamp: str  # ISO8601
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class ForecastRequest(BaseModel):
    candles: List[Candle] = Field(..., description="Historical candles, oldest first")
    pred_len: int = Field(12, ge=1, le=120, description="How many future candles to forecast")
    interval_minutes: int = Field(15, description="Spacing between candles, used to build future timestamps")
    T: float = Field(1.0, description="Sampling temperature")
    top_p: float = Field(0.9)
    sample_count: int = Field(1, ge=1, le=5, description="Number of sampled forecast paths to average/return")


class ForecastCandle(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class ForecastResponse(BaseModel):
    model: str
    lookback_used: int
    forecast: List[ForecastCandle]
    forecast_paths: List[List[ForecastCandle]]  # all sampled paths, forecast = their mean


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok" if _load_error is None else "model_load_failed",
        "model": MODEL_CHOICE,
        "model_loaded": _predictor is not None,
        "error": _load_error,
    }


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest, x_api_key: str = ""):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    predictor = get_predictor()
    cfg = MODEL_REGISTRY[MODEL_CHOICE]

    if len(req.candles) < 30:
        raise HTTPException(status_code=400, detail="Need at least 30 historical candles for a meaningful forecast")

    lookback = min(len(req.candles), cfg["max_context"])
    candles = req.candles[-lookback:]

    df = pd.DataFrame([c.dict() for c in candles])
    df["timestamps"] = pd.to_datetime(df["timestamp"], utc=True)
    x_df = df[["open", "high", "low", "close", "volume"]].copy()
    x_df["amount"] = x_df["volume"] * x_df["close"]
    x_timestamp = df["timestamps"]

    last_ts = df["timestamps"].iloc[-1]
    future_ts = pd.Series(
        [last_ts + pd.Timedelta(minutes=req.interval_minutes * (i + 1)) for i in range(req.pred_len)]
    )

    try:
        paths = []
        for _ in range(req.sample_count):
            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=future_ts,
                pred_len=req.pred_len,
                T=req.T,
                top_p=req.top_p,
                sample_count=1,
                verbose=False,
            )
            path = [
                ForecastCandle(
                    timestamp=future_ts.iloc[i].isoformat(),
                    open=float(pred_df["open"].iloc[i]),
                    high=float(pred_df["high"].iloc[i]),
                    low=float(pred_df["low"].iloc[i]),
                    close=float(pred_df["close"].iloc[i]),
                    volume=float(pred_df["volume"].iloc[i]) if "volume" in pred_df else 0.0,
                )
                for i in range(req.pred_len)
            ]
            paths.append(path)
    except Exception as e:  # noqa: BLE001
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e

    # Average across sampled paths -> the "forecast" the dashboard will chart
    avg = []
    for i in range(req.pred_len):
        o = np.mean([p[i].open for p in paths])
        h = np.mean([p[i].high for p in paths])
        l = np.mean([p[i].low for p in paths])
        c = np.mean([p[i].close for p in paths])
        v = np.mean([p[i].volume for p in paths])
        avg.append(ForecastCandle(timestamp=paths[0][i].timestamp, open=o, high=h, low=l, close=c, volume=v))

    return ForecastResponse(
        model=MODEL_CHOICE,
        lookback_used=lookback,
        forecast=avg,
        forecast_paths=paths,
    )
