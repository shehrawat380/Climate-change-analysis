"""
Modeling helpers: train/evaluate baseline models.
"""
from __future__ import annotations
import os, json, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, Tuple

def train_random_forest(Xtr: np.ndarray, ytr: pd.Series, Xv: np.ndarray, yv: pd.Series, random_state: int=42) -> Dict[str, Any]:
    model = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    model.fit(Xtr, ytr)
    preds = model.predict(Xv)
    metrics = {
        "mae": float(mean_absolute_error(yv, preds)),
        "rmse": float(mean_squared_error(yv, preds, squared=False)),
        "r2": float(r2_score(yv, preds))
    }
    return {"model": model, "val_metrics": metrics}

def evaluate(model, X, y) -> Dict[str, float]:
    preds = model.predict(X)
    return {
        "mae": float(mean_absolute_error(y, preds)),
        "rmse": float(mean_squared_error(y, preds, squared=False)),
        "r2": float(r2_score(y, preds))
    }

def save_artifacts(model, scaler, out_dir="artifacts", prefix="rf"):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{prefix}_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(out_dir, f"{prefix}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
