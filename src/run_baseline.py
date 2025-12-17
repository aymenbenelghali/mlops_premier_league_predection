"""
CPU baseline trainer using XGBoost MultiOutputRegressor.
Saves model to `models/xgb_baseline.joblib` and prints MAE/RMSE per target.
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
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

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    y_train = y_train_df[TARGETS].values.astype(float)
    y_test = y_test_df[TARGETS].values.astype(float)

    print("Training XGBoost multi-output baseline...")
    base = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, verbosity=0, n_jobs=4)
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = evaluate(y_test, preds)
    print("Metrics:")
    for t in TARGETS:
        print(f"{t}: MAE={metrics[t]['MAE']:.4f} RMSE={metrics[t]['RMSE']:.4f}")

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "xgb_baseline.joblib"
    joblib.dump({"model": model, "scaler": scaler, "features": feature_cols}, model_path)
    print(f"Saved baseline model to {model_path}")

    meta = {
        "feature_columns": feature_cols,
        "targets": TARGETS,
        "metrics": metrics,
    }
    with open(out_dir / "xgb_baseline_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {out_dir / 'xgb_baseline_metadata.json'}")


if __name__ == '__main__':
    main()
