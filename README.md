# Premier League Match Stats Prediction (XGBoost + FastAPI)

Predict upcoming Premier League match stats with a production-grade, modular MLOps setup.

Targets per match:
- total_goals = FTHG + FTAG
- total_cards = HY + AY + HR + AR
- total_corners = HC + AC
- total_offsides = HO + AO

## Project Structure

```
.
├── data/                     # raw and combined CSVs (auto-downloaded)
├── models/                   # trained model + metadata + match_history
├── src/
│   ├── __init__.py
│   ├── config.py             # env-driven settings
│   ├── data_preprocessing.py # download + clean + rolling features
│   ├── train_model.py        # train & evaluate XGBoost model
│   ├── main.py               # FastAPI app (/predict_match)
│   └── retrain.py            # optional scheduled retraining entrypoint
├── requirements.txt
└── README.md
```

## Environment & Setup

1) Python 3.10+ recommended.

2) Optional env vars (defaults shown):
```
DATA_DIR=data
MODELS_DIR=models
ROLLING_WINDOW=5
SEASONS_BACK=4
LEAGUE_CODE=E0  # Premier League
MODEL_PATH=models/xgb_multioutput.joblib
METADATA_PATH=models/feature_metadata.joblib
```

3) Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Pipeline

- Source: `https://www.football-data.co.uk/englandm.php` via season CSVs on `mmz4281/{season}/{league}.csv`.
- Automatically downloads last N seasons (default 4) based on current date.
- Parses: Date, HomeTeam, AwayTeam, FTHG, FTAG, HY, AY, HR, AR, HC, AC, HO, AO
- Builds targets and rolling averages over each team’s last 5 games.
- One-hot encodes team names (home/away).
- Saves combined dataset to `data/combined_clean.csv`.

Run preprocessing implicitly via training (next section), or directly:
```bash
python -m src.data_preprocessing
```

## Training

- Chronological split: first 3 years train, last year test (fallback 75/25).
- Model: `MultiOutputRegressor(XGBRegressor)` predicting 4 targets.
- Metrics: MAE and RMSE per target.

Run training:
```bash
python -m src.train_model
```
Artifacts written to `models/`:
- `xgb_multioutput.joblib` – trained model
- `feature_metadata.joblib` – feature columns, team columns, rolling window
- `match_history.csv` – compact history used by API to compute rolling features for new dates

## FastAPI Inference API

Start server:
```bash
uvicorn src.main:app --reload --port 8000
```

Predict endpoint:
- POST `/predict_match`
- Body:
```json
{
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "date": "2025-12-15"
}
```
- Response:
```json
{
  "total_goals": 2.31,
  "total_cards": 3.85,
  "total_corners": 10.12,
  "total_offsides": 2.04
}
```

Notes:
- Team names must match those seen during training for one-hot flags to activate.
- Rolling features are computed from `models/match_history.csv` before the provided date; if insufficient history, zeros are used.

## Retraining (Optional)

Run once (intended for weekly scheduler):
```bash
python -m src.retrain
```

Schedule weekly via Windows Task Scheduler or cron.

## Logging

- Each step uses Python `logging` for observability.
- Check your console output during runs for progress and metrics.

## Troubleshooting

- Empty dataset: ensure network access to `football-data.co.uk`.
- Missing columns across seasons: some files vary; the pipeline warns and proceeds when possible.
- Unknown team at inference: one-hot features default to 0; rolling stats default to 0.

## License

For educational and personal use. Data courtesy of football-data.co.uk.


