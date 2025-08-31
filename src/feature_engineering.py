"""
Feature engineering utilities.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

def add_time_lags(df: pd.DataFrame, target: str, lags=(1,2,4)) -> pd.DataFrame:
    df = df.copy()
    if target not in df.columns:
        return df
    for lag in lags:
        df[f"{target}_lag{lag}"] = df[target].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, target: str, windows=(3,7,12)) -> pd.DataFrame:
    df = df.copy()
    if target not in df.columns:
        return df
    for w in windows:
        df[f"{target}_rollmean_{w}"] = df[target].rolling(window=w, min_periods=1).mean()
        df[f"{target}_rollstd_{w}"] = df[target].rolling(window=w, min_periods=1).std()
    return df
