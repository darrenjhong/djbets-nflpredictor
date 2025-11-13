# data_loader.py
# Handles loading of schedule, historical, merging with ESPN, and safety-cleaning

import pandas as pd
import numpy as np
import json
import os
from utils import safe_request_json
from team_logo_map import TEAM_NAME_MAP

ESPN_SCOREBOARD_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/types/2/weeks/{week}/events"


# ---------------------------------------------------------------------
# Load historical archive (local)
# ---------------------------------------------------------------------
def load_historical(path="data/nfl_archive_10Y.json"):
    if not os.path.exists(path):
        print(f"[load_historical] No file found at {path}, returning empty DataFrame")
        return pd.DataFrame()

    try:
        with open(path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)

        # Normalize column names
        df.columns = [c.lower().strip() for c in df.columns]

        # Ensure required columns exist
        for col in ["home_team", "away_team", "home_score", "away_score"]:
            if col not in df:
                df[col] = np.nan

        return df

    except Exception as e:
        print(f"[load_historical] Error loading: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------
# Load local schedule.csv (fallback)
# ---------------------------------------------------------------------
def load_local_schedule(path="data/schedule.csv"):
    if not os.path.exists(path):
        print("[load_local_schedule] schedule.csv not found")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]

        required = ["season", "week", "home_team", "away_team"]
        for c in required:
            if c not in df.columns:
                df[c] = np.nan

        return df

    except Exception as e:
        print(f"[load_local_schedule] Error loading CSV: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------
# Fetch ESPN schedule for a single week
# ---------------------------------------------------------------------
def fetch_espn_week(season: int, week: int) -> pd.DataFrame:
    url = ESPN_SCOREBOARD_URL.format(season=season, week=week)
    data = safe_request_json(url)

    if not data or "events" not in data:
        return pd.DataFrame()

    events = data.get("events", [])
    out = []

    for game in events:
        try:
            competitions = game.get("competitions", [])
            if not competitions:
                continue

            comp = competitions[0]
            competitors = comp.get("competitors", [])
            if len(competitors) != 2:
                continue

            team1 = competitors[0]
            team2 = competitors[1]

            home = team1 if team1.get("homeAway") == "home" else team2
            away = team2 if home is team1 else team1

            out.append({
                "season": season,
                "week": week,
                "home_team": TEAM_NAME_MAP.get(
                    home["team"]["displayName"].lower(), home["team"]["displayName"]
                ),
                "away_team": TEAM_NAME_MAP.get(
                    away["team"]["displayName"].lower(), away["team"]["displayName"]
                ),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "status": comp.get("status", {}).get("type", {}).get("name", "unknown"),
            })
        except Exception:
            continue

    return pd.DataFrame(out)


# ---------------------------------------------------------------------
# Fetch full ESPN schedule for current year
# ---------------------------------------------------------------------
def fetch_espn_season(season: int, weeks=18) -> pd.DataFrame:
    frames = []
    for w in range(1, weeks + 1):
        df = fetch_espn_week(season, w)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------
# Choose best available schedule (ESPN > local CSV)
# ---------------------------------------------------------------------
def load_or_fetch_schedule(season: int):
    # Load local CSV first
    local = load_local_schedule()

    # Fetch ESPN schedule
    espn = fetch_espn_season(season)

    if not espn.empty:
        return espn

    return local


# ---------------------------------------------------------------------
# Safely merge schedule + historical
# ---------------------------------------------------------------------
def merge_schedule_and_history(schedule_df, hist_df):
    if schedule_df.empty:
        return pd.DataFrame()

    df = schedule_df.copy()

    # Ensure required columns
    for col in ["home_team", "away_team"]:
        df[col] = df[col].astype(str).str.lower().str.strip()

    # historical team name normalization
    if not hist_df.empty:
        hist_df["home_team"] = hist_df["home_team"].astype(str).str.lower().str.strip()
        hist_df["away_team"] = hist_df["away_team"].astype(str).str.lower().str.strip()

    # We DO NOT merge history onto schedule (different structure),
    # Instead we provide schedule; history is used separately for training only.
    return df