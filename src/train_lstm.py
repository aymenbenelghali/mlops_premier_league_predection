"""
Improved LSTM training script for multi-target count regression (goals, cards, corners) with MLflow tracking.

What this adds compared to previous version:
- MLflow experiment/run creation and logging of params, metrics and artifacts
- Per-epoch logging of train/val loss into MLflow
- Logs final metrics and saves model artifact (both torch checkpoint and mlflow-pytorch)
- Logs scaler and feature_columns as a JSON artifact

Usage examples:
  # Run locally and track with default file store
  python lstm_training_improved.py --epochs 50 --batch_size 32 --loss_type log_mse --experiment "football_lstm"

  # Or configure MLflow server
  export MLFLOW_TRACKING_URI=http://localhost:5000
  python lstm_training_improved.py --experiment "football_lstm"

This is meant to be a drop-in replacement for your previous script. It assumes
`load_and_prepare_data()` and `select_features()` are available in your package.
"""

import os
import argparse
import logging
import json
import tempfile
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

try:
    import mlflow
    # mlflow.pytorch is optional and may trigger additional imports; import lazily below
    mlflow_pytorch = getattr(mlflow, "pytorch", None)
except Exception:
    mlflow = None
    mlflow_pytorch = None

from .data_preprocessing import load_and_prepare_data
from .train_model import select_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

TARGETS = ["total_goals", "total_cards", "total_corners"]


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, self.X.shape[0] - self.seq_len)

    def __getitem__(self, idx: int):
        x_seq = self.X[idx:idx + self.seq_len]
        y_next = self.y[idx + self.seq_len]
        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(y_next, dtype=torch.float32),
        )


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        out_dim: int = 3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, out_dim),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


def chronological_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def to_sequences(X_df: pd.DataFrame, y_df: pd.DataFrame, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X = X_df.values.astype(np.float32)
    y = y_df[TARGETS].values.astype(np.float32)
    return X, y


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def train_one_epoch(model, loader, optimizer, device, loss_fn, clip_grad=None):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_samples += xb.size(0)
    return total_loss / max(1, total_samples)


@torch.no_grad()
def predict_model(model, loader, device):
    model.eval()
    preds = []
    trues = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        pred = model(xb).detach().cpu().numpy()
        preds.append(pred)
        # yb comes from dataset as tensor on CPU
        if isinstance(yb, torch.Tensor):
            trues.append(yb.cpu().numpy())
        else:
            trues.append(np.asarray(yb))
    if len(preds) == 0:
        return np.empty((0, len(TARGETS))), np.empty((0, len(TARGETS)))
    preds_all = np.vstack(preds)
    trues_all = np.vstack(trues)
    min_len = min(len(preds_all), len(trues_all))
    return trues_all[:min_len], preds_all[:min_len]


def evaluate(y_true_transformed, y_pred_transformed, transform: str = "log1p"):
    if transform == "log1p":
        y_true = np.expm1(y_true_transformed)
        y_pred = np.expm1(y_pred_transformed)
    elif transform == "identity":
        y_true = y_true_transformed
        y_pred = y_pred_transformed
    else:
        raise ValueError("Unknown transform")
    metrics = {}
    for i, t in enumerate(TARGETS):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        # Compute RMSE via sqrt(MSE) to avoid relying on potentially incompatible
        # sklearn `squared` keyword across versions.
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        rmse = float(np.sqrt(mse))
        metrics[t] = {"MAE": float(mae), "RMSE": float(rmse)}
    return metrics


def main(args):
    set_seed(args.seed)
    logger.info("Loading data...")
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
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_df), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_df), columns=feature_cols)

    seq_len = args.seq_len
    Xtr, ytr = to_sequences(X_train_scaled, y_train_df, seq_len)
    Xte, yte = to_sequences(X_test_scaled, y_test_df, seq_len)

    if args.loss_type == "log_mse":
        ytr_trans = np.log1p(ytr)
        yte_trans = np.log1p(yte)
    elif args.loss_type == "identity":
        ytr_trans = ytr
        yte_trans = yte
    else:
        ytr_trans = ytr
        yte_trans = yte

    train_ds = SeqDataset(Xtr, ytr_trans, seq_len)
    test_ds = SeqDataset(Xte, yte_trans, seq_len)

    # DataLoader settings
    num_workers = args.num_workers if args.num_workers is not None else 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    # Device selection
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model = LSTMRegressor(
        num_features=len(feature_cols),
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        out_dim=len(TARGETS),
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)

    val_cut = max(1, int(len(train_ds) * 0.1))
    if val_cut > 1:
        val_subset = torch.utils.data.Subset(train_ds, range(len(train_ds) - val_cut, len(train_ds)))
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    else:
        val_loader = None

    # MLflow setup (resilient and opt-in)
    # To disable MLflow entirely, set `MLFLOW_NO_TRACKING=1` in your environment.
    mlflow_available = False
    if os.getenv("MLFLOW_NO_TRACKING", "false").lower() in ("1", "true", "yes"):
        logger.info("MLflow tracking disabled via MLFLOW_NO_TRACKING environment variable")
    elif mlflow is None:
        logger.info("mlflow package not available; continuing without tracking")
    else:
        try:
            # If no tracking URI is provided, use MLflow's default file-based
            # tracking (the `mlruns/` directory in the working directory). This
            # avoids creating a sqlite DB and running alembic migrations.
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            # else: leave default (file-based mlruns) — safe and avoids DB migrations

            try:
                mlflow.set_experiment(args.experiment)
            except Exception as ex:
                # If the experiments folder is corrupted or an experiment meta file
                # is missing, create a fresh experiment name to avoid failures.
                logger.warning(f"mlflow.set_experiment failed: {ex}; creating fallback experiment name")
                import time

                fallback_name = f"{args.experiment}_{int(time.time())}"
                try:
                    # create_experiment returns an id; set_experiment accepts name
                    mlflow.create_experiment(fallback_name)
                    mlflow.set_experiment(fallback_name)
                    logger.info(f"Created fallback experiment '{fallback_name}'")
                except Exception as ex2:
                    logger.warning(f"Failed to create fallback MLflow experiment: {ex2}")
                    raise
            mlflow_available = True
        except Exception as e:
            logger.warning(f"MLflow unavailable, continuing without tracking: {e}")
            mlflow_available = False

    # Start MLflow run if available; otherwise use a dummy context manager
    if mlflow_available:
        try:
            run_ctx = mlflow.start_run(run_name=args.run_name)
        except Exception as e:
            logger.warning(f"Failed to start MLflow run, continuing without tracking: {e}")
            mlflow_available = False
            run_ctx = None
    else:
        run_ctx = None

    class _DummyCtx:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    ctx = run_ctx if run_ctx is not None else _DummyCtx()
    with ctx as run:
        # log params (guarded)
        if mlflow_available:
            try:
                mlflow.log_params({
                    "seq_len": seq_len,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "hidden": args.hidden,
                    "layers": args.layers,
                    "dropout": args.dropout,
                    "loss_type": args.loss_type,
                    "weight_decay": args.weight_decay,
                    "seed": args.seed,
                })
            except Exception as e:
                logger.warning(f"mlflow.log_params failed: {e}")

        # save and log feature & scaler metadata
        meta = {
            "feature_columns": feature_cols,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "targets": TARGETS,
        }
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tf:
            json.dump(meta, tf)
            tf.flush()
            if mlflow_available:
                try:
                    mlflow.log_artifact(tf.name, artifact_path="metadata")
                except Exception as e:
                    logger.warning(f"mlflow.log_artifact failed: {e}")

        best_val = float("inf")
        best_state = None
        bad = 0

        logger.info(
            f"Training: epochs={args.epochs}, seq_len={seq_len}, hidden={args.hidden}, layers={args.layers}, device={device}"
        )
        for ep in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optim, device, loss_fn, clip_grad=args.clip_grad)
            val_loss = None
            if val_loader is not None:
                vt, vp = predict_model(model, val_loader, device)
                val_loss = float(((vp - vt) ** 2).mean())
                # Step the scheduler once per epoch (avoid passing epoch/metric
                # as that use is being deprecated in newer PyTorch versions).
                scheduler.step()
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    bad = 0
                    best_state = model.state_dict()
                else:
                    bad += 1
            # log per-epoch metrics to MLflow (guarded)
            if mlflow_available:
                try:
                    mlflow.log_metric("train_loss", train_loss, step=ep)
                    if val_loss is not None:
                        mlflow.log_metric("val_loss", val_loss, step=ep)
                except Exception as e:
                    logger.warning(f"mlflow.log_metric failed: {e}")

            if ep % args.log_every == 0 or ep == 1 or ep == args.epochs:
                logger.info(f"Epoch {ep}/{args.epochs} - train_loss={train_loss:.6f}{' val_loss='+str(round(val_loss,6)) if val_loss is not None else ''}")
            if bad >= args.patience:
                logger.info("Early stopping triggered")
                break

        # restore best
        if best_state is not None:
            model.load_state_dict(best_state)

        # predict on test set
        y_true_t, y_pred_t = predict_model(model, test_loader, device)

        # Align test ground-truth to sequence windows: targets start at index `seq_len`
        yte_aligned = yte_trans[seq_len: seq_len + len(y_true_t)]
        # If predict_model returned fewer samples than available, truncate accordingly
        min_test_len = min(len(y_true_t), len(yte_aligned))
        y_true_t = y_true_t[:min_test_len]
        y_pred_t = y_pred_t[:min_test_len]
        yte_test = yte_aligned[:min_test_len]

        if args.loss_type == "poisson":
            y_pred_post = torch.nn.functional.softplus(torch.tensor(y_pred_t)).numpy()
            # we used raw counts as target for poisson during training, compare counts
            metrics = evaluate(yte_test, np.log1p(y_pred_post), transform="log1p")
        elif args.loss_type == "log_mse":
            metrics = evaluate(yte_test, y_pred_t, transform="log1p")
        else:
            metrics = evaluate(yte_test, y_pred_t, transform="identity")

        for t in TARGETS:
            logger.info(f"{t}: MAE={metrics[t]['MAE']:.3f} RMSE={metrics[t]['RMSE']:.3f}")
            if mlflow_available:
                try:
                    mlflow.log_metric(f"MAE/{t}", metrics[t]["MAE"]) 
                    mlflow.log_metric(f"RMSE/{t}", metrics[t]["RMSE"]) 
                except Exception as e:
                    logger.warning(f"mlflow.log_metric failed: {e}")

        # save model checkpoint locally and log as artifact
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, "lstm_multi_targets_best.pt")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "feature_columns": feature_cols,
                "seq_len": seq_len,
                "targets": TARGETS,
                "hidden_size": args.hidden,
                "num_layers": args.layers,
                "dropout": args.dropout,
                "metrics": metrics,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
            },
            model_path,
        )
        logger.info(f"Model saved to {model_path}")
        # log checkpoint file (guarded)
        if mlflow_available:
            try:
                mlflow.log_artifact(model_path, artifact_path="checkpoint")
            except Exception as e:
                logger.warning(f"mlflow.log_artifact failed: {e}")
        # log pytorch model via mlflow.pytorch (stores a proper MLflow PyTorch model)
        try:
            # Record exact pip requirements for reproducibility (include torch with
            # local version label when possible). MLflow may strip local version
            # labels when auto-detecting, so we pass them explicitly here.
            pip_reqs = [f"torch=={torch.__version__}", f"numpy=={np.__version__}"]
            if mlflow_available:
                        try:
                            # Use mlflow.pytorch if available; fall back to logging the checkpoint
                            if mlflow_pytorch is not None:
                                mlflow_pytorch.log_model(model, artifact_path="pytorch_model", pip_requirements=pip_reqs)
                            else:
                                logger.info("mlflow.pytorch not available; skipping MLflow PyTorch model log")
                        except Exception as e:
                            logger.warning(f"mlflow.pytorch.log_model failed: {e}")
        except Exception as e:
            logger.warning(f"mlflow.pytorch.log_model failed: {e}")

    # end of MLflow run


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seq_len", type=int, default=int(os.getenv("LSTM_SEQ_LEN", "7")))
    p.add_argument("--batch_size", type=int, default=int(os.getenv("LSTM_BATCH_SIZE", "32")))
    p.add_argument("--epochs", type=int, default=int(os.getenv("LSTM_EPOCHS", "50")))
    p.add_argument("--lr", type=float, default=float(os.getenv("LSTM_LR", "0.0005")))
    p.add_argument("--hidden", type=int, default=int(os.getenv("LSTM_HIDDEN", "64")))
    p.add_argument("--layers", type=int, default=int(os.getenv("LSTM_LAYERS", "1")))
    p.add_argument("--dropout", type=float, default=float(os.getenv("LSTM_DROPOUT", "0.1")))
    p.add_argument("--weight_decay", type=float, default=float(os.getenv("LSTM_WD", "1e-3")))
    p.add_argument("--loss_type", type=str, default=os.getenv("LSTM_LOSS", "log_mse"),
                   choices=["log_mse", "identity", "poisson"])  # log_mse = MSE on log1p(y)
    p.add_argument("--patience", type=int, default=int(os.getenv("LSTM_EARLY_PATIENCE", "10")))
    p.add_argument("--clip_grad", type=float, default=float(os.getenv("LSTM_CLIP", "1.0")))
    p.add_argument("--output_dir", type=str, default=os.getenv("MODEL_DIR", "models"))
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--use_cuda", action="store_true", help="Enable CUDA if available")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=3)
    p.add_argument("--experiment", type=str, default=os.getenv("MLFLOW_EXPERIMENT", "lstm_experiment"))
    p.add_argument("--run_name", type=str, default=os.getenv("MLFLOW_RUN_NAME", "run_1"))
    args = p.parse_args()
    main(args)
