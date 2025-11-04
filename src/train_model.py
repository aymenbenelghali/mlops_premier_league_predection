import os
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import joblib

from .config import settings
from .data_preprocessing import load_and_prepare_data


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


TARGETS = [
    "total_goals",
    "total_cards",
    "total_corners",
    "total_offsides",
]


def chronological_train_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Use last year as test
    years = sorted(df["Date"].dt.year.unique())
    if len(years) < 2:
        # Fallback: 75/25 split chronologically
        cutoff = int(0.75 * len(df))
        return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()
    train_years = years[:-1]
    test_year = years[-1]
    train_df = df[df["Date"].dt.year.isin(train_years)].copy()
    test_df = df[df["Date"].dt.year == test_year].copy()
    return train_df, test_df


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    base_features = [
        "home_avg_goals",
        "home_avg_cards",
        "home_avg_corners",
        "home_avg_offsides",
        "away_avg_goals",
        "away_avg_cards",
        "away_avg_corners",
        "away_avg_offsides",
        "is_weekend",
        "year",
        "month",
        # advanced
        "home_avg_gs_5",
        "home_avg_gc_5",
        "away_avg_gs_5",
        "away_avg_gc_5",
        "rest_days_home",
        "rest_days_away",
        "h2h_avg_goals",
        "home_points_5",
        "home_points_10",
        "home_gd_5",
        "home_gd_10",
        "away_points_5",
        "away_points_10",
        "away_gd_5",
        "away_gd_10",
        "home_elo_like",
        "away_elo_like",
        "elo_diff",
        "points5_diff",
        "gd5_diff",
    ]
    spi_features = [
        "home_spi","away_spi","spi_diff",
        "home_off","away_off","off_diff",
        "home_def","away_def","def_diff",
    ]
    # take only available columns
    feature_cols = [c for c in base_features if c in df.columns]
    feature_cols += [c for c in spi_features if c in df.columns]
    # Add one-hot team columns
    team_cols = [c for c in df.columns if c.startswith("home_is_") or c.startswith("away_is_")]
    feature_cols.extend(sorted(team_cols))
    # Add referee one-hots if present
    ref_cols = [c for c in df.columns if c.startswith("ref_")]
    feature_cols.extend(sorted(ref_cols))

    X = df[feature_cols].fillna(0)
    y = df[TARGETS]
    return X, y, feature_cols


def build_model() -> MultiOutputRegressor:
    base = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        tree_method="hist",
        objective="reg:squarederror",
        n_jobs=0,
    )
    model = MultiOutputRegressor(base)
    return model


def maybe_tune_hyperparams(X: pd.DataFrame, y: pd.Series) -> Dict[str, object]:
    """Tune a single-output XGB on total_goals and return best params."""
    params = {
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "n_estimators": [300, 600, 900],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    base = XGBRegressor(
        random_state=42,
        tree_method="hist",
        objective="reg:squarederror",
        n_jobs=0,
    )
    grid = GridSearchCV(base, params, cv=3, scoring="neg_mean_absolute_error", n_jobs=1, verbose=1)
    grid.fit(X, y)
    logger.info(f"Best params: {grid.best_params_}")
    return grid.best_params_


def evaluate(y_true: pd.DataFrame, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for i, target in enumerate(TARGETS):
        true_i = y_true.iloc[:, i].values
        pred_i = y_pred[:, i]
        mae = mean_absolute_error(true_i, pred_i)
        rmse = mean_squared_error(true_i, pred_i, squared=False)
        metrics[target] = {"MAE": float(mae), "RMSE": float(rmse)}
    return metrics


def save_artifacts(model: MultiOutputRegressor, feature_cols: List[str], reference_df: pd.DataFrame) -> None:
    os.makedirs(settings.models_dir, exist_ok=True)
    joblib.dump(model, settings.model_path)
    # Save metadata for API feature building
    metadata = {
        "feature_columns": feature_cols,
        "team_columns": [c for c in feature_cols if c.startswith("home_is_") or c.startswith("away_is_")],
        "rolling_window": settings.rolling_window,
    }
    joblib.dump(metadata, settings.metadata_path)
    # Save a compact reference history used to compute rolling features on demand
    ref_cols = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "HY",
        "AY",
        "HR",
        "AR",
        "HC",
        "AC",
        "HO",
        "AO",
    ]
    history_path = os.path.join(settings.models_dir, "match_history.csv")
    if all(c in reference_df.columns for c in ref_cols):
        reference_df[ref_cols].to_csv(history_path, index=False)
    logger.info(f"Saved model to {settings.model_path} and metadata to {settings.metadata_path}")


def main() -> None:
    logger.info("Loading and preparing data...")
    df = load_and_prepare_data()
    logger.info("Splitting train/test chronologically...")
    train_df, test_df = chronological_train_test_split(df)

    X_train, y_train, feature_cols = select_features(train_df)
    X_test, y_test, _ = select_features(test_df)

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    # Optional tuning via env var
    tune = os.getenv("TUNE_HYPERPARAMS", "false").lower() in ("1", "true", "yes")
    if tune:
        best = maybe_tune_hyperparams(X_train, y_train["total_goals"])  # tune on goals proxy
        base = XGBRegressor(
            random_state=42,
            tree_method="hist",
            objective="reg:squarederror",
            n_jobs=0,
            **best,
        )
        model = MultiOutputRegressor(base)
    else:
        model = build_model()
    logger.info("Training XGBoost multi-output model...")
    model.fit(X_train, y_train)

    logger.info("Evaluating on test set...")
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)
    for target, vals in metrics.items():
        logger.info(f"{target}: MAE={vals['MAE']:.3f} RMSE={vals['RMSE']:.3f}")

    save_artifacts(model, feature_cols, reference_df=df)


if __name__ == "__main__":
    main()


