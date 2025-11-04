import os
import io
import logging
from datetime import datetime
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
import requests

from .config import settings


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


FOOTBALL_DATA_URL_TEMPLATE = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"


def derive_last_n_seasons(n: int) -> List[str]:
    """Return last n season codes like '2324', '2425'. Includes current season.

    If current date is Aug or later, current season is YY(YY+1). Otherwise previous season.
    """
    today = datetime.utcnow()
    year = today.year
    month = today.month

    # English PL season starts around August
    start_year = year if month >= 8 else year - 1
    seasons = []
    for i in range(n):
        sy = (start_year - i) % 100
        ey = (start_year - i + 1) % 100
        seasons.append(f"{sy:02d}{ey:02d}")
    return seasons


def download_season_csv(season_code: str, league_code: str, data_dir: str) -> Tuple[str, pd.DataFrame]:
    url = FOOTBALL_DATA_URL_TEMPLATE.format(season=season_code, league=league_code)
    os.makedirs(data_dir, exist_ok=True)
    local_path = os.path.join(data_dir, f"{league_code}_{season_code}.csv")
    try:
        logger.info(f"Downloading {url}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)
        df = pd.read_csv(io.BytesIO(resp.content))
        logger.info(f"Saved {local_path} with shape {df.shape}")
        return local_path, df
    except Exception as exc:
        logger.warning(f"Failed to fetch {url}: {exc}")
        return local_path, pd.DataFrame()


RELEVANT_COLUMNS = [
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


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    # football-data sometimes uses different date formats; try parse flexibly
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True, infer_datetime_format=True)
    return df


TEAM_NORMALIZATION: Dict[str, str] = {
    # Common aliases to canonical names used by football-data
    "Man United": "Man United",
    "Manchester United": "Man United",
    "Man Utd": "Man United",
    "Manchester Utd": "Man United",
    "Man City": "Man City",
    "Manchester City": "Man City",
    "Spurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Wolves": "Wolves",
    "Wolverhampton": "Wolves",
    "Newcastle United": "Newcastle",
    "Brighton & Hove Albion": "Brighton",
    "West Bromwich Albion": "West Brom",
}


def normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["HomeTeam", "AwayTeam"]:
        df[col] = df[col].astype(str).map(lambda x: TEAM_NORMALIZATION.get(x, x))
    return df


# Map football-data names to FiveThirtyEight names
FDC_TO_SPI: Dict[str, str] = {
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Newcastle": "Newcastle United",
    "Tottenham": "Tottenham Hotspur",
    "Wolves": "Wolverhampton",
    "West Brom": "West Bromwich Albion",
    "Brighton": "Brighton and Hove Albion",
    "Bournemouth": "AFC Bournemouth",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Sheffield United": "Sheffield United",
    "Nott'm Forest": "Nottingham Forest",
    "Nottm Forest": "Nottingham Forest",
}


def download_spi_rankings(data_dir: str) -> str:
    url = "https://projects.fivethirtyeight.com/soccer-api/club/spi_global_rankings.csv"
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "spi_global_rankings.csv")
    try:
        logger.info(f"Downloading SPI rankings from {url}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
        return out_path
    except Exception as exc:
        logger.warning(f"Failed to fetch SPI rankings: {exc}")
        return out_path


def add_spi_features(df: pd.DataFrame) -> pd.DataFrame:
    spi_path = download_spi_rankings(settings.data_dir)
    if not os.path.exists(spi_path):
        return df
    try:
        spi = pd.read_csv(spi_path, encoding="utf-8", on_bad_lines="skip")
    except Exception as exc:
        logger.warning(f"Failed to read SPI file: {exc}")
        return df

    # Expected columns: name, league, date, spi, off, def
    if not set(["name", "league", "date", "spi", "off", "def"]).issubset(spi.columns):
        return df

    spi["date"] = pd.to_datetime(spi["date"], errors="coerce")
    # Filter to English leagues only
    mask = spi["league"].astype(str).str.contains("Premier League", case=False, na=False)
    spi = spi[mask].copy()
    spi = spi.sort_values(["name", "date"])  # chronological per team

    # Prepare match frame with mapped team names to SPI naming
    enriched = df.copy()
    enriched["spi_home_name"] = enriched["HomeTeam"].map(lambda x: FDC_TO_SPI.get(x, x))
    enriched["spi_away_name"] = enriched["AwayTeam"].map(lambda x: FDC_TO_SPI.get(x, x))

    # Merge-asof style: for each team, take latest SPI before match date
    def merge_spi(side_col: str, prefix: str) -> pd.DataFrame:
        left = enriched[["Date", side_col]].rename(columns={side_col: "name"}).copy()
        left = left.sort_values(["name", "Date"])  # required for asof by group
        right = spi[["name", "date", "spi", "off", "def"]].copy()
        # group-wise asof by name
        merged_list = []
        for name, group in left.groupby("name", sort=False):
            sp = right[right["name"] == name].sort_values("date")
            if sp.empty:
                tmp = group.copy()
                tmp[[f"{prefix}_spi", f"{prefix}_off", f"{prefix}_def"]] = np.nan
                merged_list.append(tmp)
                continue
            tmp = pd.merge_asof(
                group.sort_values("Date"),
                sp.rename(columns={"date": "spi_date"}).sort_values("spi_date"),
                left_on="Date",
                right_on="spi_date",
                direction="backward",
            )
            tmp = tmp.drop(columns=["spi_date"]) if "spi_date" in tmp.columns else tmp
            tmp = tmp.rename(columns={
                "spi": f"{prefix}_spi",
                "off": f"{prefix}_off",
                "def": f"{prefix}_def",
            })
            merged_list.append(tmp)
        out = pd.concat(merged_list, ignore_index=True)
        return out

    home_spi = merge_spi("spi_home_name", "home")
    away_spi = merge_spi("spi_away_name", "away")

    # Join back on Date and names
    enriched = enriched.merge(
        home_spi[["Date", "name", "home_spi", "home_off", "home_def"]],
        left_on=["Date", "spi_home_name"], right_on=["Date", "name"], how="left"
    ).drop(columns=["name"])
    enriched = enriched.merge(
        away_spi[["Date", "name", "away_spi", "away_off", "away_def"]],
        left_on=["Date", "spi_away_name"], right_on=["Date", "name"], how="left"
    ).drop(columns=["name"])

    # Diffs
    enriched["spi_diff"] = enriched["home_spi"].fillna(0) - enriched["away_spi"].fillna(0)
    enriched["off_diff"] = enriched["home_off"].fillna(0) - enriched["away_off"].fillna(0)
    enriched["def_diff"] = enriched["home_def"].fillna(0) - enriched["away_def"].fillna(0)

    return enriched.drop(columns=["spi_home_name", "spi_away_name"])


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure missing numeric columns exist (some seasons omit offsides)
    for col in ["FTHG", "FTAG", "HY", "AY", "HR", "AR", "HC", "AC", "HO", "AO"]:
        if col not in df.columns:
            df[col] = np.nan
    df["total_goals"] = df["FTHG"].fillna(0) + df["FTAG"].fillna(0)
    df["total_cards"] = (
        df[["HY", "AY", "HR", "AR"]].fillna(0).sum(axis=1)
    )
    df["total_corners"] = df[["HC", "AC"]].fillna(0).sum(axis=1)
    # Offsides strict validation
    if ("HO" in df.columns) and ("AO" in df.columns):
        df["total_offsides"] = df[["HO", "AO"]].fillna(0).sum(axis=1)
        # Warn if offsides variance is near zero
        if df["total_offsides"].var(skipna=True) < 1e-6:
            logger.warning("total_offsides variance is near zero; learning may be impaired.")
    else:
        allow_fb = os.getenv("ALLOW_OFFSIDES_FALLBACK", "false").lower() in ("1", "true", "yes")
        if not allow_fb:
            raise RuntimeError("HO/AO (offsides) columns missing. Set ALLOW_OFFSIDES_FALLBACK=true to proceed with proxy.")
        # Fallback proxy using shots on target and corners if available
        proxy = None
        shots_cols = [c for c in df.columns if c.upper() in ("HST", "AST")]
        if ("HST" in df.columns) and ("AST" in df.columns):
            proxy = df[["HST", "AST"]].fillna(0).sum(axis=1) * 0.1
        elif ("HC" in df.columns) and ("AC" in df.columns):
            proxy = df[["HC", "AC"]].fillna(0).sum(axis=1) * 0.05
        else:
            proxy = 0
        logger.warning("Using offsides proxy due to missing HO/AO; set ALLOW_OFFSIDES_FALLBACK=false to abort instead.")
        df["total_offsides"] = proxy
    return df


def _long_format_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Convert match rows to per-team rows to compute rolling stats.

    Returns columns: [team, date, goals, cards, corners, offsides, is_home]
    """
    home = pd.DataFrame(
        {
            "team": df["HomeTeam"],
            "date": df["Date"],
            "goals": df["FTHG"],
            "cards": df[["HY", "HR"]].sum(axis=1),
            "corners": df["HC"],
            "offsides": df.get("HO", 0),
            "is_home": 1,
        }
    )
    away = pd.DataFrame(
        {
            "team": df["AwayTeam"],
            "date": df["Date"],
            "goals": df["FTAG"],
            "cards": df[["AY", "AR"]].sum(axis=1),
            "corners": df["AC"],
            "offsides": df.get("AO", 0),
            "is_home": 0,
        }
    )
    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.sort_values(["team", "date"])  # chronological
    return long_df


def compute_rolling_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute rolling averages per team, then merge back to match-level features."""
    long_df = _long_format_team_stats(df)

    long_df[["goals", "cards", "corners", "offsides"]] = long_df[
        ["goals", "cards", "corners", "offsides"]
    ].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Rolling means excluding current match (shift)
    long_df["avg_goals_rolling"] = long_df.groupby("team")["goals"].transform(
        lambda s: s.shift(1).rolling(window).mean()
    )
    long_df["avg_cards_rolling"] = long_df.groupby("team")["cards"].transform(
        lambda s: s.shift(1).rolling(window).mean()
    )
    long_df["avg_corners_rolling"] = long_df.groupby("team")["corners"].transform(
        lambda s: s.shift(1).rolling(window).mean()
    )
    long_df["avg_offsides_rolling"] = long_df.groupby("team")["offsides"].transform(
        lambda s: s.shift(1).rolling(window).mean()
    )

    # Prepare for merge
    home_feats = long_df[long_df["is_home"] == 1][
        [
            "team",
            "date",
            "avg_goals_rolling",
            "avg_cards_rolling",
            "avg_corners_rolling",
            "avg_offsides_rolling",
        ]
    ].rename(columns={
        "team": "HomeTeam",
        "date": "Date",
        "avg_goals_rolling": "home_avg_goals",
        "avg_cards_rolling": "home_avg_cards",
        "avg_corners_rolling": "home_avg_corners",
        "avg_offsides_rolling": "home_avg_offsides",
    })

    away_feats = long_df[long_df["is_home"] == 0][
        [
            "team",
            "date",
            "avg_goals_rolling",
            "avg_cards_rolling",
            "avg_corners_rolling",
            "avg_offsides_rolling",
        ]
    ].rename(columns={
        "team": "AwayTeam",
        "date": "Date",
        "avg_goals_rolling": "away_avg_goals",
        "avg_cards_rolling": "away_avg_cards",
        "avg_corners_rolling": "away_avg_corners",
        "avg_offsides_rolling": "away_avg_offsides",
    })

    # Merge on (Date, HomeTeam/AwayTeam) with original df
    merged = df.merge(home_feats, on=["Date", "HomeTeam"], how="left")
    merged = merged.merge(away_feats, on=["Date", "AwayTeam"], how="left")

    return merged


def one_hot_encode_teams(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]], ignore_index=True))
    teams = sorted([t for t in teams if isinstance(t, str)])
    team_cols = []

    encoded = df.copy()
    for team in teams:
        hcol = f"home_is_{team}"
        acol = f"away_is_{team}"
        encoded[hcol] = (encoded["HomeTeam"] == team).astype(int)
        encoded[acol] = (encoded["AwayTeam"] == team).astype(int)
        team_cols.extend([hcol, acol])
    return encoded, team_cols


def load_and_prepare_data() -> pd.DataFrame:
    seasons = derive_last_n_seasons(settings.seasons_back)
    all_dfs: List[pd.DataFrame] = []

    for season in seasons:
        _, df = download_season_csv(season, settings.league_code, settings.data_dir)
        if df.empty:
            continue
        missing_cols = [c for c in RELEVANT_COLUMNS if c not in df.columns]
        if missing_cols:
            logger.warning(f"Season {season} missing columns {missing_cols}; attempting to continue.")
        use_cols = [c for c in RELEVANT_COLUMNS if c in df.columns]
        df = df[use_cols].copy()
        df = parse_dates(df)
        df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])  # essential fields
        # Coerce numeric columns
        for col in [c for c in RELEVANT_COLUMNS if c not in ["Date", "HomeTeam", "AwayTeam"] and c in df.columns]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("No data downloaded. Please check network or source availability.")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = normalize_team_names(combined)
    combined = combined.sort_values("Date").reset_index(drop=True)
    combined = build_targets(combined)

    # Compute rolling features
    combined = compute_rolling_features(combined, window=settings.rolling_window)

    # External enrichment (SPI)
    combined = add_spi_features(combined)

    # One-hot encode teams
    combined, team_cols = one_hot_encode_teams(combined)

    # Add simple recency features
    combined["is_weekend"] = combined["Date"].dt.weekday.isin([5, 6]).astype(int)
    combined["year"] = combined["Date"].dt.year
    combined["month"] = combined["Date"].dt.month

    # Advanced features: home/away specific forms and rest days
    combined = add_advanced_features(combined)

    # Drop rows without rolling history (first few games)
    rolling_cols = [
        "home_avg_goals",
        "home_avg_cards",
        "home_avg_corners",
        "home_avg_offsides",
        "away_avg_goals",
        "away_avg_cards",
        "away_avg_corners",
        "away_avg_offsides",
    ]
    combined = combined.dropna(subset=rolling_cols)

    # Save combined raw for reference
    out_path = os.path.join(settings.data_dir, "combined_clean.csv")
    os.makedirs(settings.data_dir, exist_ok=True)
    combined.to_csv(out_path, index=False)
    logger.info(f"Combined dataset saved to {out_path} shape={combined.shape}")

    return combined


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("Date")

    # Per-team last home and away rolling for goals scored and conceded
    def team_side_form(frame: pd.DataFrame, side: str, window: int) -> pd.DataFrame:
        is_home = 1 if side == "home" else 0
        team_col = "HomeTeam" if is_home else "AwayTeam"
        goals_scored = "FTHG" if is_home else "FTAG"
        goals_conceded = "FTAG" if is_home else "FTHG"
        sub = frame[["Date", team_col, goals_scored, goals_conceded]].rename(columns={team_col: "team"})
        sub = sub.sort_values(["team", "Date"])  # chronological
        sub[f"{side}_avg_gs_5"] = sub.groupby("team")[goals_scored].transform(lambda s: s.shift(1).rolling(5).mean())
        sub[f"{side}_avg_gc_5"] = sub.groupby("team")[goals_conceded].transform(lambda s: s.shift(1).rolling(5).mean())
        sub = sub[["Date", "team", f"{side}_avg_gs_5", f"{side}_avg_gc_5"]]
        return sub

    home_form = team_side_form(df, "home", 5).rename(columns={"team": "HomeTeam"})
    away_form = team_side_form(df, "away", 5).rename(columns={"team": "AwayTeam"})
    df = df.merge(home_form, on=["Date", "HomeTeam"], how="left")
    df = df.merge(away_form, on=["Date", "AwayTeam"], how="left")

    # Rest days for each team since last match
    def rest_days(frame: pd.DataFrame, team_col: str) -> pd.Series:
        tmp = frame[["Date", team_col]].copy()
        tmp = tmp.sort_values([team_col, "Date"])  # chronological
        last_date = tmp.groupby(team_col)["Date"].shift(1)
        return (tmp["Date"] - last_date).dt.days

    df["rest_days_home"] = rest_days(df, "HomeTeam")
    df["rest_days_away"] = rest_days(df, "AwayTeam")

    # Head-to-head previous average goals (last 5 meetings before current date)
    df["h2h_avg_goals"] = np.nan
    # build key for teams independent of home/away order
    pairs = pd.DataFrame({
        "Date": df["Date"],
        "pair_key": df.apply(lambda r: "::".join(sorted([str(r["HomeTeam"]), str(r["AwayTeam"])])), axis=1),
        "total_goals": df["total_goals"],
    })
    pairs = pairs.sort_values("Date")
    # rolling previous 5 totals per pair
    prev_mean = pairs.groupby("pair_key")["total_goals"].transform(lambda s: s.shift(1).rolling(5).mean())
    pairs["h2h_avg_goals"] = prev_mean
    df["h2h_avg_goals"] = pairs["h2h_avg_goals"].values

    # Results and points
    df["home_points"] = np.select(
        [df["FTHG"] > df["FTAG"], df["FTHG"] == df["FTAG"]], [3, 1], default=0
    )
    df["away_points"] = np.select(
        [df["FTAG"] > df["FTHG"], df["FTAG"] == df["FTHG"]], [3, 1], default=0
    )
    df["goal_diff"] = df["FTHG"].fillna(0) - df["FTAG"].fillna(0)

    # Rolling points and goal diff by team overall (last 5 and 10)
    def team_roll(frame: pd.DataFrame) -> pd.DataFrame:
        # Build per team match rows with points and goal diff signed per team
        home_rows = frame[["Date", "HomeTeam", "home_points", "goal_diff"]].rename(
            columns={"HomeTeam": "team", "home_points": "points", "goal_diff": "gd"}
        )
        away_rows = frame[["Date", "AwayTeam", "away_points", "goal_diff"]].rename(
            columns={"AwayTeam": "team", "away_points": "points", "goal_diff": "gd"}
        )
        away_rows["gd"] = -away_rows["gd"]  # goal diff from away perspective
        tdf = pd.concat([home_rows, away_rows], ignore_index=True).sort_values(["team", "Date"]).copy()
        for w in (5, 10):
            tdf[f"roll_points_{w}"] = tdf.groupby("team")["points"].transform(lambda s: s.shift(1).rolling(w).mean())
            tdf[f"roll_gd_{w}"] = tdf.groupby("team")["gd"].transform(lambda s: s.shift(1).rolling(w).mean())
        # EMA-like rating
        tdf["rating_signal"] = 0.6 * tdf["points"].fillna(0) + 0.4 * tdf["gd"].fillna(0)
        tdf["elo_like"] = tdf.groupby("team")["rating_signal"].transform(lambda s: s.shift(1).ewm(alpha=0.1).mean())
        return tdf

    team_stats = team_roll(df)
    # Merge back for home and away
    home_merge = team_stats[["Date", "team", "roll_points_5", "roll_points_10", "roll_gd_5", "roll_gd_10", "elo_like"]].rename(
        columns={
            "team": "HomeTeam",
            "roll_points_5": "home_points_5",
            "roll_points_10": "home_points_10",
            "roll_gd_5": "home_gd_5",
            "roll_gd_10": "home_gd_10",
            "elo_like": "home_elo_like",
        }
    )
    away_merge = team_stats[["Date", "team", "roll_points_5", "roll_points_10", "roll_gd_5", "roll_gd_10", "elo_like"]].rename(
        columns={
            "team": "AwayTeam",
            "roll_points_5": "away_points_5",
            "roll_points_10": "away_points_10",
            "roll_gd_5": "away_gd_5",
            "roll_gd_10": "away_gd_10",
            "elo_like": "away_elo_like",
        }
    )
    df = df.merge(home_merge, on=["Date", "HomeTeam"], how="left")
    df = df.merge(away_merge, on=["Date", "AwayTeam"], how="left")

    # Derived diffs
    df["elo_diff"] = df["home_elo_like"].fillna(0) - df["away_elo_like"].fillna(0)
    df["points5_diff"] = df["home_points_5"].fillna(0) - df["away_points_5"].fillna(0)
    df["gd5_diff"] = df["home_gd_5"].fillna(0) - df["away_gd_5"].fillna(0)

    # Referee encoding if available
    if "Referee" in df.columns:
        ref = df[["Referee"]].copy()
        ref_cols = pd.get_dummies(ref["Referee"], prefix="ref", dummy_na=False)
        df = pd.concat([df.reset_index(drop=True), ref_cols.reset_index(drop=True)], axis=1)

    return df


def data_quality_report(df: pd.DataFrame) -> None:
    os.makedirs(settings.data_dir, exist_ok=True)
    # Summaries
    info_buf = []
    nulls = df.isnull().sum().to_frame(name="null_count")
    desc = df.describe(include="all").transpose()
    # Targets variance
    var_targets = df[[c for c in ["total_goals", "total_cards", "total_corners", "total_offsides"] if c in df.columns]].var()
    # Save consolidated report
    report_path = os.path.join(settings.data_dir, "report.csv")
    out = pd.concat([nulls, desc], axis=1)
    out["variance_targets"] = np.nan
    for k, v in var_targets.items():
        if k in out.index:
            out.loc[k, "variance_targets"] = v
    out.to_csv(report_path)
    logger.info(f"Data quality report saved to {report_path}")


if __name__ == "__main__":
    df = load_and_prepare_data()
    logger.info(df.head().to_string())


