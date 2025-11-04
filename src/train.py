import logging
import json
import os
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

from .config import settings
from .data_preprocessing import load_and_prepare_data, data_quality_report
from .train_model import select_features


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


TARGETS = ["total_goals", "total_cards", "total_corners", "total_offsides"]


def get_model_pipeline(scale: bool) -> Pipeline:
    steps = []
    if scale and (os.getenv("SCALE_FEATURES", "false").lower() in ("1", "true", "yes")):
        steps.append(("scaler", StandardScaler(with_mean=False)))
    steps.append(("xgb", XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )))
    return Pipeline(steps)


def tune_params(X: pd.DataFrame, y: pd.Series) -> Dict:
    params = {
        "xgb__n_estimators": [200, 500, 800],
        "xgb__max_depth": [4, 6, 8],
        "xgb__learning_rate": [0.01, 0.03, 0.05],
        "xgb__subsample": [0.7, 0.9, 1.0],
        "xgb__colsample_bytree": [0.6, 0.8, 1.0],
        "xgb__reg_lambda": [0.5, 1.0, 2.0],
        "xgb__reg_alpha": [0.0, 0.5, 1.0],
    }
    base = get_model_pipeline(scale=False)
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(base, params, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=1)
    grid.fit(X, y)
    logger.info(f"Best params: {grid.best_params_}")
    return grid.best_params_


def plot_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, feature_names: List[str], model: XGBRegressor, target: str) -> None:
    os.makedirs("plots", exist_ok=True)
    # Actual vs Pred
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{target} - Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join("plots", f"{target}_actual_vs_pred.png"))
    plt.close()

    # Residuals
    resid = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(resid, bins=30)
    plt.title(f"{target} - Residuals")
    plt.tight_layout()
    plt.savefig(os.path.join("plots", f"{target}_residuals_hist.png"))
    plt.close()

    # Feature importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:20]
        plt.figure(figsize=(8, 6))
        plt.barh([feature_names[i] for i in idx][::-1], importances[idx][::-1])
        plt.title(f"{target} - Top Feature Importances")
        plt.tight_layout()
        plt.savefig(os.path.join("plots", f"{target}_feature_importance.png"))
        plt.close()

    # SHAP summary if available
    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(pd.DataFrame(columns=feature_names))
            # Use a small sample of X for plotting if needed (we skip here to keep runtime low)
        except Exception:
            pass


def main() -> None:
    df = load_and_prepare_data()
    data_quality_report(df)

    # Chronological split (last year test)
    years = sorted(df["Date"].dt.year.unique())
    train_df = df[df["Date"].dt.year.isin(years[:-1])]
    test_df = df[df["Date"].dt.year == years[-1]]

    X_train, y_train_all, feature_cols = select_features(train_df)
    X_test, y_test_all, _ = select_features(test_df)

    scale = os.getenv("SCALE_FEATURES", "false").lower() in ("1", "true", "yes")
    metrics: Dict[str, Dict[str, float]] = {}
    best_params_per_target: Dict[str, Dict] = {}

    os.makedirs(settings.models_dir, exist_ok=True)
    for target in TARGETS:
        y_train = y_train_all[target]
        y_test = y_test_all[target]

        params = None
        if os.getenv("TUNE_HYPERPARAMS", "false").lower() in ("1", "true", "yes"):
            params = tune_params(X_train, y_train)

        pipe = get_model_pipeline(scale)
        if params:
            pipe.set_params(**params)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(mean_squared_error(y_test, y_pred, squared=False))
        metrics[target] = {"MAE": mae, "RMSE": rmse}
        best_params_per_target[target] = params or {}

        # Save model
        joblib.dump(pipe, os.path.join(settings.models_dir, f"{target}.pkl"))
        plot_diagnostics(y_test.values, np.asarray(y_pred), feature_cols, pipe.named_steps["xgb"], target)

    # Save metadata and metrics
    meta = {
        "feature_columns": feature_cols,
        "best_params": best_params_per_target,
    }
    joblib.dump(meta, os.path.join(settings.models_dir, "metadata.joblib"))
    with open(os.path.join(settings.models_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    # Save history for API
    hist_cols = ["Date","HomeTeam","AwayTeam","FTHG","FTAG","HY","AY","HR","AR","HC","AC","HO","AO"]
    hist_cols = [c for c in hist_cols if c in df.columns]
    df[hist_cols].to_csv(os.path.join(settings.models_dir, "match_history.csv"), index=False)
    logger.info(f"Training complete. Metrics: {metrics}")


if __name__ == "__main__":
    main()


