"""Prediction service for the Nagpur Weavers' Cluster.

POST /predict  -> metres woven in the shift
GET  /health   -> {"status": "ok"}
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).resolve().parents[1] / "models" / "model.joblib"))
FEATURES = ["loom_age_yr", "thread_count", "power_cuts_min", "weaver_exp_yr",
            "humidity_pct", "shift_hours", "fabric", "shed"]


class JsonFormatter(logging.Formatter):
    """One JSON object per log line."""

    def format(self, record):
        entry = {"time": self.formatTime(record), "level": record.levelname,
                 "logger": record.name, "message": record.getMessage()}
        entry.update(getattr(record, "extra_fields", {}))
        return json.dumps(entry)


handler = logging.StreamHandler(sys.stdout)  # stdout only, never a file
handler.setFormatter(JsonFormatter())
logger = logging.getLogger("mp088.api")
logger.handlers = [handler]
logger.setLevel(logging.INFO)
logger.propagate = False



class ShiftRequest(BaseModel):
    loom_age_yr: float = Field(..., le=55, description="client rule 1: above 55 is impossible")
    thread_count: float
    power_cuts_min: float = Field(..., ge=0, description="client rule 2: below 0 is a typing mistake")
    weaver_exp_yr: float
    humidity_pct: float
    shift_hours: float
    fabric: str
    shed: str


app = FastAPI(title="MP088 Power Loom Output")
model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(req: ShiftRequest):
    start = time.perf_counter()
    row = pd.DataFrame([req.model_dump()])[FEATURES]
    prediction = round(float(model.predict(row)[0]), 2)
    logger.info("prediction", extra={"extra_fields": {
        "event": "prediction",
        "input": req.model_dump(),  # loom features, no personal details
        "prediction": prediction,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2)
    }})
    return {"metres_per_shift": prediction}
