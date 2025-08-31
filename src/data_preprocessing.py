"""
Data preprocessing utilities.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

NUM_IMPUTE = {"strategy": "median"}

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def basic_cleaning(df: pd.DataFrame, drop_dupes: bool = True) -> pd.DataFrame:
    df = df.copy()
    if drop_dupes:
        df = df.drop_duplicates()
    # strip column names
    df.columns = [c.strip() for c in df.columns]
    # parse dates if present
    for col in df.columns:
        if "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
    return df

def train_val_test_split(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    assert target in df.columns, f"Target '{target}' not in columns."
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size+val_size, random_state=random_state)
    rel_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-rel_val, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_numeric(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train[numeric_cols])
    Xv = scaler.transform(X_val[numeric_cols])
    Xte = scaler.transform(X_test[numeric_cols])
    return Xtr, Xv, Xte, scaler
