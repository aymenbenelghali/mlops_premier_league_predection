import os
import logging
from datetime import datetime
from typing import Dict, List

import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field

from .config import settings


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


class MatchRequest(BaseModel):
    home_team: str = Field(..., description="Home team name")
    away_team: str = Field(..., description="Away team name")
    date: str = Field(..., description="Match date in YYYY-MM-DD format")


class MatchPrediction(BaseModel):
    total_goals: float
    total_cards: float
    total_corners: float
    total_offsides: float


app = FastAPI(title="Premier League Match Stat Predictor")

# Templates and static setup
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(["html", "xml"]),
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def load_model_and_metadata():
    if not os.path.exists(settings.model_path) or not os.path.exists(settings.metadata_path):
        raise RuntimeError("Model or metadata not found. Train the model first.")
    model = joblib.load(settings.model_path)
    metadata = joblib.load(settings.metadata_path)
    history_path = os.path.join(settings.models_dir, "match_history.csv")
    if not os.path.exists(history_path):
        raise RuntimeError("match_history.csv not found. Re-train to generate artifacts.")
    history = pd.read_csv(history_path)
    # Parse dates
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce", dayfirst=True, infer_datetime_format=True)
    return model, metadata, history


MODEL = None
METADATA: Dict = {}
HISTORY: pd.DataFrame = pd.DataFrame()

# Common alias -> canonical mapping to match football-data team names
ALIAS_TO_CANON = {
    "Manchester United": "Man United",
    "Man Utd": "Man United",
    "Man United": "Man United",
    "Manchester City": "Man City",
    "Tottenham Hotspur": "Tottenham",
    "Spurs": "Tottenham",
    "Wolverhampton": "Wolves",
    "Wolverhampton Wanderers": "Wolves",
    "Newcastle United": "Newcastle",
    "Brighton and Hove Albion": "Brighton",
    "West Bromwich Albion": "West Brom",
    "Nottingham Forest": "Nott'm Forest",
    "Nottm Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield United",
    "AFC Bournemouth": "Bournemouth",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
}


def normalize_team_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    return ALIAS_TO_CANON.get(name.strip(), name.strip())


@app.on_event("startup")
def load_model():
    global MODEL, METADATA, HISTORY
    MODEL, METADATA, HISTORY = load_model_and_metadata()
    logger.info("Model and metadata loaded.")


@app.get("/", response_class=HTMLResponse)
def home():
    template = env.get_template("index.html")
    return template.render()


def compute_team_rolling(history: pd.DataFrame, team: str, cutoff_date: pd.Timestamp, window: int) -> Dict[str, float]:
    # Build per-team events like in preprocessing
    df = history.copy()
    df = df[df["Date"] < cutoff_date]

    home = df[df["HomeTeam"] == team][["Date", "FTHG", "HY", "HR", "HC", "HO"]].rename(
        columns={"FTHG": "goals", "HY": "yc", "HR": "rc", "HC": "corners", "HO": "offsides"}
    )
    away = df[df["AwayTeam"] == team][["Date", "FTAG", "AY", "AR", "AC", "AO"]].rename(
        columns={"FTAG": "goals", "AY": "yc", "AR": "rc", "AC": "corners", "AO": "offsides"}
    )
    home["cards"] = home["yc"].fillna(0) + home["rc"].fillna(0)
    away["cards"] = away["yc"].fillna(0) + away["rc"].fillna(0)

    sel_cols = ["Date", "goals", "cards", "corners", "offsides"]
    team_games = pd.concat([home[sel_cols], away[sel_cols]], ignore_index=True).sort_values("Date")

    if team_games.empty:
        return {
            "avg_goals": np.nan,
            "avg_cards": np.nan,
            "avg_corners": np.nan,
            "avg_offsides": np.nan,
        }

    # take last window games
    recent = team_games.tail(window)
    return {
        "avg_goals": float(recent["goals"].mean()),
        "avg_cards": float(recent["cards"].mean()),
        "avg_corners": float(recent["corners"].mean()),
        "avg_offsides": float(recent["offsides"].mean()),
    }


def team_exists(team: str) -> bool:
    if HISTORY is None or HISTORY.empty:
        return False
    team_norm = normalize_team_name(team)
    teams = pd.unique(pd.concat([HISTORY["HomeTeam"], HISTORY["AwayTeam"]], ignore_index=True).astype(str))
    return team_norm in set(teams)


def validate_request_teams(home_team: str, away_team: str) -> None:
    missing = []
    if not team_exists(home_team):
        missing.append(home_team)
    if not team_exists(away_team):
        missing.append(away_team)
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unknown team(s)",
                "unknown": missing,
                "hint": "Team names are normalized (e.g., 'Manchester United' -> 'Man United'). Check spelling or try short names.",
            },
        )


def build_feature_vector(req: MatchRequest) -> pd.DataFrame:
    # Normalize team names to canonical forms used in training
    req.home_team = normalize_team_name(req.home_team)
    req.away_team = normalize_team_name(req.away_team)
    # Validate teams exist in history; ensures model uses each team's last games
    validate_request_teams(req.home_team, req.away_team)
    cutoff = pd.to_datetime(req.date)
    window = int(METADATA.get("rolling_window", settings.rolling_window))

    home_stats = compute_team_rolling(HISTORY, req.home_team, cutoff, window)
    away_stats = compute_team_rolling(HISTORY, req.away_team, cutoff, window)

    # Construct a single-row feature frame
    data = {
        "home_avg_goals": home_stats["avg_goals"],
        "home_avg_cards": home_stats["avg_cards"],
        "home_avg_corners": home_stats["avg_corners"],
        "home_avg_offsides": home_stats["avg_offsides"],
        "away_avg_goals": away_stats["avg_goals"],
        "away_avg_cards": away_stats["avg_cards"],
        "away_avg_corners": away_stats["avg_corners"],
        "away_avg_offsides": away_stats["avg_offsides"],
        "is_weekend": int(cutoff.weekday() in (5, 6)),
        "year": cutoff.year,
        "month": cutoff.month,
    }

    # One-hot team columns as in training
    feature_cols: List[str] = METADATA["feature_columns"]
    team_cols: List[str] = METADATA["team_columns"]
    for col in team_cols:
        data[col] = 0
    home_col = f"home_is_{req.home_team}"
    away_col = f"away_is_{req.away_team}"
    if home_col in data:
        data[home_col] = 1
    if away_col in data:
        data[away_col] = 1

    row = pd.DataFrame([data])
    # Ensure all features exist and in correct order
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_cols]
    # Handle NaNs from lack of history
    row = row.fillna(0)
    return row


def build_feature_vector_lenient(req: MatchRequest) -> pd.DataFrame:
    """Lenient builder for upcoming fixtures listing: normalize teams, do not
    enforce history presence; default missing rolling/team one-hots to 0."""
    # Normalize
    req.home_team = normalize_team_name(req.home_team)
    req.away_team = normalize_team_name(req.away_team)
    cutoff = pd.to_datetime(req.date)
    window = int(METADATA.get("rolling_window", settings.rolling_window))

    # Try to compute rolling; if team has no games yet, return NaNs -> fill 0
    home_stats = compute_team_rolling(HISTORY, req.home_team, cutoff, window)
    away_stats = compute_team_rolling(HISTORY, req.away_team, cutoff, window)

    data = {
        "home_avg_goals": home_stats["avg_goals"],
        "home_avg_cards": home_stats["avg_cards"],
        "home_avg_corners": home_stats["avg_corners"],
        "home_avg_offsides": home_stats["avg_offsides"],
        "away_avg_goals": away_stats["avg_goals"],
        "away_avg_cards": away_stats["avg_cards"],
        "away_avg_corners": away_stats["avg_corners"],
        "away_avg_offsides": away_stats["avg_offsides"],
        "is_weekend": int(cutoff.weekday() in (5, 6)),
        "year": cutoff.year,
        "month": cutoff.month,
    }
    feature_cols: List[str] = METADATA.get("feature_columns", [])
    team_cols: List[str] = METADATA.get("team_columns", [])
    # Initialize one-hots to 0; set if present in metadata
    for col in team_cols:
        data[col] = 0
    home_col = f"home_is_{req.home_team}"
    away_col = f"away_is_{req.away_team}"
    if home_col in team_cols:
        data[home_col] = 1
    if away_col in team_cols:
        data[away_col] = 1

    row = pd.DataFrame([data])
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_cols] if feature_cols else row
    return row.fillna(0)


@app.post("/predict_match", response_model=MatchPrediction)
def predict_match(req: MatchRequest):
    try:
        _ = pd.to_datetime(req.date)  # validate date format
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    X = build_feature_vector(req)
    # Ensure we have some history for both teams; otherwise, instruct the user
    if X[["home_avg_goals", "away_avg_goals"]].isna().any(axis=None):
        raise HTTPException(
            status_code=400,
            detail="Insufficient match history for one or both teams prior to the given date.",
        )
    try:
        pred = MODEL.predict(X)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    pred = np.asarray(pred).reshape(1, -1)
    return MatchPrediction(
        total_goals=float(pred[0, 0]),
        total_cards=float(pred[0, 1]),
        total_corners=float(pred[0, 2]),
        total_offsides=float(pred[0, 3]),
    )


@app.get("/predict_upcoming_week")
def predict_upcoming_week(days: int = 14):
    # 1) FPL public API (no auth) – reliable free source
    try:
        import requests as _req
        today = datetime.utcnow().date()
        horizon = today + pd.Timedelta(days=int(days))
        # Map team IDs to names
        bs = _req.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=20)
        bs.raise_for_status()
        teams = {t["id"]: (t.get("name") or t.get("short_name") or str(t["id"])) for t in bs.json().get("teams", [])}
        # Fetch all fixtures then filter by date window and status (SCHEDULED)
        fx = _req.get("https://fantasy.premierleague.com/api/fixtures/", timeout=25)
        fx.raise_for_status()
        results_fpl = []
        for m in fx.json():
            kt = m.get("kickoff_time")
            if not kt:
                continue
            try:
                d = pd.to_datetime(kt).date()
            except Exception:
                continue
            if not (today <= d <= (horizon.date() if hasattr(horizon, 'date') else horizon)):
                continue
            if m.get("finished_provisional") or m.get("finished"):
                continue
            home = teams.get(m.get("team_h"))
            away = teams.get(m.get("team_a"))
            if not home or not away:
                continue
            req = MatchRequest(home_team=str(home), away_team=str(away), date=str(d))
            X = build_feature_vector_lenient(req)
            pred = np.asarray(MODEL.predict(X)).reshape(1, -1)
            results_fpl.append(
                {
                    "date": str(d),
                    "home_team": req.home_team,
                    "away_team": req.away_team,
                    "total_goals": float(pred[0, 0]),
                    "total_cards": float(pred[0, 1]),
                    "total_corners": float(pred[0, 2]),
                    "total_offsides": float(pred[0, 3]),
                }
            )
        if results_fpl:
            return {"fixtures": sorted(results_fpl, key=lambda x: x["date"]) }
    except Exception as exc:
        logger.warning(f"FPL fixtures fetch failed: {exc}")

    # 2) Try external football-data.org if token available
    token = os.getenv("FOOTBALL_DATA_API_TOKEN")
    if token:
        try:
            base = "https://api.football-data.org/v4/competitions/PL/matches"
            today = datetime.utcnow().date()
            date_from = today.isoformat()
            date_to = (today + pd.Timedelta(days=int(days))).date().isoformat()
            import requests as _req
            r = _req.get(
                base,
                params={"dateFrom": date_from, "dateTo": date_to, "status": "SCHEDULED,TIMED", "limit": 200},
                headers={"X-Auth-Token": token},
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            results = []
            for m in data.get("matches", []):
                utc_date = m.get("utcDate")
                try:
                    d = pd.to_datetime(utc_date).date()
                except Exception:
                    d = today
                home = m.get("homeTeam", {}).get("shortName") or m.get("homeTeam", {}).get("name")
                away = m.get("awayTeam", {}).get("shortName") or m.get("awayTeam", {}).get("name")
                if not home or not away:
                    continue
                # Build request row and predict using lenient builder
                req = MatchRequest(home_team=str(home), away_team=str(away), date=str(d))
                X = build_feature_vector_lenient(req)
                pred = np.asarray(MODEL.predict(X)).reshape(1, -1)
                results.append(
                    {
                        "date": str(d),
                        "home_team": req.home_team,
                        "away_team": req.away_team,
                        "total_goals": float(pred[0, 0]),
                        "total_cards": float(pred[0, 1]),
                        "total_corners": float(pred[0, 2]),
                        "total_offsides": float(pred[0, 3]),
                    }
                )
            # If we got any from external API, return immediately
            if results:
                return {"fixtures": results}
        except Exception as exc:
            logger.warning(f"External fixtures API failed: {exc}; falling back to local CSV.")

    # Secondary source: TheSportsDB (public key defaults to '1')
    try:
        tsdb_key = os.getenv("THESPORTSDB_API_KEY", "1")
        import requests as _req
        r2 = _req.get(
            f"https://www.thesportsdb.com/api/v1/json/{tsdb_key}/eventsnextleague.php",
            params={"id": 4328},  # Premier League league id
            timeout=20,
        )
        if r2.ok:
            data2 = r2.json() or {}
            events = data2.get("events") or []
            results_tsdb = []
            today = datetime.utcnow().date()
            horizon = today + pd.Timedelta(days=int(days))
            for e in events:
                dstr = e.get("dateEvent")
                if not dstr:
                    continue
                try:
                    d = pd.to_datetime(dstr).date()
                except Exception:
                    continue
                if not (today <= d <= horizon.date() if hasattr(horizon, 'date') else horizon):
                    continue
                home = e.get("strHomeTeam")
                away = e.get("strAwayTeam")
                if not home or not away:
                    continue
                req = MatchRequest(home_team=str(home), away_team=str(away), date=str(d))
                X = build_feature_vector_lenient(req)
                pred = np.asarray(MODEL.predict(X)).reshape(1, -1)
                results_tsdb.append(
                    {
                        "date": str(d),
                        "home_team": req.home_team,
                        "away_team": req.away_team,
                        "total_goals": float(pred[0, 0]),
                        "total_cards": float(pred[0, 1]),
                        "total_corners": float(pred[0, 2]),
                        "total_offsides": float(pred[0, 3]),
                    }
                )
            if results_tsdb:
                return {"fixtures": results_tsdb}

        # Season-wide fallback: fetch entire season schedule then filter next N days
        # Derive season string like 2025-2026
        today_dt = datetime.utcnow()
        start_year = today_dt.year if today_dt.month >= 8 else today_dt.year - 1
        season_str = f"{start_year}-{start_year+1}"
        r3 = _req.get(
            f"https://www.thesportsdb.com/api/v1/json/{tsdb_key}/eventsseason.php",
            params={"id": 4328, "s": season_str},
            timeout=25,
        )
        if r3.ok:
            data3 = r3.json() or {}
            events3 = data3.get("events") or []
            results_tsdb2 = []
            today = datetime.utcnow().date()
            horizon = today + pd.Timedelta(days=int(days))
            for e in events3:
                dstr = e.get("dateEvent")
                if not dstr:
                    continue
                try:
                    d = pd.to_datetime(dstr).date()
                except Exception:
                    continue
                if not (today <= d <= (horizon.date() if hasattr(horizon, 'date') else horizon)):
                    continue
                home = e.get("strHomeTeam")
                away = e.get("strAwayTeam")
                if not home or not away:
                    continue
                req = MatchRequest(home_team=str(home), away_team=str(away), date=str(d))
                X = build_feature_vector_lenient(req)
                pred = np.asarray(MODEL.predict(X)).reshape(1, -1)
                results_tsdb2.append(
                    {
                        "date": str(d),
                        "home_team": req.home_team,
                        "away_team": req.away_team,
                        "total_goals": float(pred[0, 0]),
                        "total_cards": float(pred[0, 1]),
                        "total_corners": float(pred[0, 2]),
                        "total_offsides": float(pred[0, 3]),
                    }
                )
            if results_tsdb2:
                return {"fixtures": sorted(results_tsdb2, key=lambda x: x["date"]) }
    except Exception as exc:
        logger.warning(f"TheSportsDB fallback failed: {exc}")

    # Final fallback: derive from latest downloaded season CSVs (may be empty if future fixtures not present)
    data_files = [f for f in os.listdir(settings.data_dir) if f.startswith("E0_") and f.endswith(".csv")]
    if not data_files:
        raise HTTPException(status_code=400, detail="No season CSVs found. Train the model first to download data.")
    # pick most recent by season code
    latest = sorted(data_files)[-1]
    path = os.path.join(settings.data_dir, latest)
    df = pd.read_csv(path)
    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True, infer_datetime_format=True)
    # Normalize to date (drop time part) to avoid timezone/clock drift issues
    df["Date"] = df["Date"].dt.date
    today = datetime.utcnow().date()
    horizon = today + pd.Timedelta(days=int(days))
    # filter fixtures in the next N days
    fixtures = df[(df["Date"] >= today) & (df["Date"] <= horizon)][["Date", "HomeTeam", "AwayTeam"]].dropna()

    # If empty, progressively expand window or pick next available matchday in file
    if fixtures.empty:
        for extra in (28, 60):
            ext_horizon = today + pd.Timedelta(days=extra)
            fx = df[(df["Date"] >= today) & (df["Date"] <= ext_horizon)][["Date", "HomeTeam", "AwayTeam"]].dropna()
            if not fx.empty:
                fixtures = fx
                break
    if fixtures.empty:
        # pick the next date available in the file after today; if none, choose last available date
        future_dates = sorted({d for d in df["Date"].dropna().unique() if d >= today})
        if future_dates:
            next_date = future_dates[0]
            fixtures = df[df["Date"] == next_date][["Date", "HomeTeam", "AwayTeam"]].dropna()
        else:
            past_dates = sorted({d for d in df["Date"].dropna().unique() if d <= today})
            if past_dates:
                last_date = past_dates[-1]
                fixtures = df[df["Date"] == last_date][["Date", "HomeTeam", "AwayTeam"]].dropna()
    results = []
    for _, row in fixtures.iterrows():
        req = MatchRequest(home_team=str(row["HomeTeam"]), away_team=str(row["AwayTeam"]), date=str(row["Date"]))
        X = build_feature_vector(req)
        pred = np.asarray(MODEL.predict(X)).reshape(1, -1)
        results.append(
            {
                "date": str(row["Date"]),
                "home_team": req.home_team,
                "away_team": req.away_team,
                "total_goals": float(pred[0, 0]),
                "total_cards": float(pred[0, 1]),
                "total_corners": float(pred[0, 2]),
                "total_offsides": float(pred[0, 3]),
            }
        )
    return {"fixtures": results}


@app.get("/recent_results")
def recent_results(days: int = 30):
    # Find latest season CSV in data directory
    data_files = [f for f in os.listdir(settings.data_dir) if f.startswith("E0_") and f.endswith(".csv")]
    if not data_files:
        raise HTTPException(status_code=400, detail="No season CSVs found. Train the model first to download data.")
    latest = sorted(data_files)[-1]
    path = os.path.join(settings.data_dir, latest)
    df = pd.read_csv(path)
    # Parse dates and normalize to date only
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True, infer_datetime_format=True)
    df["Date"] = df["Date"].dt.date
    today = datetime.utcnow().date()
    start = today - pd.Timedelta(days=int(days))
    played = df[(df["Date"] >= start) & (df["Date"] <= today)].copy()
    # Keep only rows with full-time stats available
    for c in ["FTHG","FTAG","HY","AY","HR","AR","HC","AC"]:
        if c not in played.columns:
            played[c] = np.nan
    results = []
    for _, row in played.iterrows():
        total_goals = float(pd.to_numeric(row.get("FTHG", 0), errors="coerce") + pd.to_numeric(row.get("FTAG", 0), errors="coerce"))
        total_cards = float(pd.to_numeric(row.get("HY", 0), errors="coerce") + pd.to_numeric(row.get("AY", 0), errors="coerce") + pd.to_numeric(row.get("HR", 0), errors="coerce") + pd.to_numeric(row.get("AR", 0), errors="coerce"))
        total_corners = float(pd.to_numeric(row.get("HC", 0), errors="coerce") + pd.to_numeric(row.get("AC", 0), errors="coerce"))
        # Offsides if available
        ho = pd.to_numeric(row.get("HO", 0), errors="coerce") if "HO" in df.columns else 0
        ao = pd.to_numeric(row.get("AO", 0), errors="coerce") if "AO" in df.columns else 0
        total_offsides = float((0 if np.isnan(ho) else ho) + (0 if np.isnan(ao) else ao))
        results.append({
            "date": str(row["Date"]),
            "home_team": str(row.get("HomeTeam")),
            "away_team": str(row.get("AwayTeam")),
            "total_goals": total_goals,
            "total_cards": total_cards,
            "total_corners": total_corners,
            "total_offsides": total_offsides,
        })
    # Sort most recent first
    results = sorted(results, key=lambda r: r["date"], reverse=True)
    return {"results": results}


# Local dev entrypoint: uvicorn src.main:app --reload


