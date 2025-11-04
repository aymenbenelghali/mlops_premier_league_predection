import os
import logging
from typing import Dict, List
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import settings
from .main import compute_team_rolling, build_feature_vector, load_model_and_metadata, team_exists


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    date: str


app = FastAPI(title="PL Stats Predictor (Per-Target Models)")


MODELS: Dict[str, object] = {}
METADATA: Dict = {}
HISTORY: pd.DataFrame = pd.DataFrame()


@app.on_event("startup")
def startup_load():
    global MODELS, METADATA, HISTORY
    # Load per-target models and shared metadata
    targets = ["total_goals", "total_cards", "total_corners", "total_offsides"]
    for t in targets:
        path = os.path.join(settings.models_dir, f"{t}.pkl")
        if not os.path.exists(path):
            raise RuntimeError(f"Model not found for {t}: {path}")
        MODELS[t] = joblib.load(path)
    meta_path = os.path.join(settings.models_dir, "metadata.joblib")
    if not os.path.exists(meta_path):
        raise RuntimeError("metadata.joblib not found. Train first.")
    METADATA = joblib.load(meta_path)
    hist_path = os.path.join(settings.models_dir, "match_history.csv")
    if not os.path.exists(hist_path):
        raise RuntimeError("match_history.csv not found. Train first.")
    HISTORY = pd.read_csv(hist_path)
    HISTORY["Date"] = pd.to_datetime(HISTORY["Date"], errors="coerce", dayfirst=True, infer_datetime_format=True)
    logger.info("Models and metadata loaded for API.")


@app.post("/predict_match")
def predict_match(req: MatchRequest):
    # Validate date
    try:
        _ = pd.to_datetime(req.date)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format (YYYY-MM-DD)")
    # Validate teams
    if not team_exists(req.home_team) or not team_exists(req.away_team):
        raise HTTPException(status_code=400, detail="Unknown team(s)")

    # Build features using existing util from main.py (keeps feature set consistent)
    from .main import METADATA as META_APP, HISTORY as HIST_APP, build_feature_vector as build_row
    # Temporarily bind for reuse
    META_APP.clear(); META_APP.update(METADATA)
    HIST_APP.drop(HIST_APP.index, inplace=True)
    HIST_APP[HISTORY.columns] = HISTORY

    X = build_row(req)
    # Ensure features ordering
    feature_cols: List[str] = METADATA.get("feature_columns", list(X.columns))
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_cols].fillna(0)

    preds: Dict[str, float] = {}
    for t, model in MODELS.items():
        y = model.predict(X)
        preds[t] = float(np.asarray(y).ravel()[0])

    # Confidence proxy using metrics if available
    conf = {}
    try:
        import json
        with open(os.path.join(settings.models_dir, "metrics.json"), "r", encoding="utf-8") as f:
            m = json.load(f)
        for k, v in m.items():
            conf[k] = max(0.0, 1.0 - float(v.get("RMSE", 1.0)) / (float(v.get("MAE", 1.0)) + 1e-6))
    except Exception:
        pass

    return {"predictions": preds, "confidence": conf}


