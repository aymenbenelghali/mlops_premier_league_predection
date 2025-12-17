"""
Sequence MLP fallback (CPU-only) — flattens sliding windows and trains an MLPRegressor.
Saves model to `models/mlp_seq.joblib` and prints MAE/RMSE per target.
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
import joblib

from .data_preprocessing import load_and_prepare_data
from .train_model import select_features

TARGETS = ["total_goals", "total_cards", "total_corners"]


def chronological_split(df: pd.DataFrame):
    years = sorted(df["Date"].dt.year.unique())
    if len(years) < 2:
        cut = int(0.75 * len(df))
        return df.iloc[:cut].copy(), df.iloc[cut:].copy()
    return df[df["Date"].dt.year.isin(years[:-1])].copy(), df[df["Date"].dt.year == years[-1]].copy()


def add_time_features(X: pd.DataFrame, df_dates: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(df_dates)
    dow = dates.dt.weekday
    month = dates.dt.month
    dayofyear = dates.dt.dayofyear
    X = X.copy()
    X["dow"] = dow
    X["month"] = month
    X["sin_doy"] = np.sin(2 * np.pi * dayofyear / 365.25)
    X["cos_doy"] = np.cos(2 * np.pi * dayofyear / 365.25)
    return X


def to_sequences_flat(X_df: pd.DataFrame, y_df: pd.DataFrame, seq_len: int):
    X = X_df.values.astype(np.float32)
    y = y_df[TARGETS].values.astype(np.float32)
    n = max(0, X.shape[0] - seq_len)
    Xs = np.zeros((n, seq_len * X.shape[1]), dtype=np.float32)
    ys = np.zeros((n, y.shape[1]), dtype=np.float32)
    for i in range(n):
        Xs[i] = X[i:i+seq_len].reshape(-1)
        ys[i] = y[i+seq_len]
    return Xs, ys


def evaluate(y_true, y_pred):
    metrics = {}
    for i, t in enumerate(TARGETS):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = mean_squared_error(y_true[:, i], y_pred[:, i], squared=False)
        metrics[t] = {"MAE": float(mae), "RMSE": float(rmse)}
    return metrics


def main():
    print("Loading data...")
    df = load_and_prepare_data()
    train_df, test_df = chronological_split(df)
    train_df = train_df.sort_values("Date").reset_index(drop=True)
    test_df = test_df.sort_values("Date").reset_index(drop=True)

    X_train_df, y_train_df, feature_cols = select_features(train_df)
    X_test_df, y_test_df, _ = select_features(test_df)

    X_train_df = add_time_features(X_train_df, train_df["Date"])
    X_test_df = add_time_features(X_test_df, test_df["Date"])
    feature_cols = list(X_train_df.columns)

    seq_len = 7
    X_train, y_train = to_sequences_flat(X_train_df, y_train_df, seq_len)
    X_test, y_test = to_sequences_flat(X_test_df, y_test_df, seq_len)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training MLP on flattened sequences...")
    mlp = MLPRegressor(hidden_layer_sizes=(256,128), activation='relu', max_iter=200, random_state=42)
    mlp.fit(X_train, y_train)

    preds = mlp.predict(X_test)

    metrics = evaluate(y_test, preds)
    print("Metrics:")
    for t in TARGETS:
        print(f"{t}: MAE={metrics[t]['MAE']:.4f} RMSE={metrics[t]['RMSE']:.4f}")

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "mlp_seq.joblib"
    joblib.dump({"model": mlp, "scaler": scaler, "features": feature_cols, "seq_len": seq_len}, model_path)
    print(f"Saved model to {model_path}")

    meta = {"feature_columns": feature_cols, "targets": TARGETS, "metrics": metrics}
    with open(out_dir / "mlp_seq_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {out_dir / 'mlp_seq_metadata.json'}")

if __name__ == '__main__':
    main()
