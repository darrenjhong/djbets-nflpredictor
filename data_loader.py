# data_loader.py
import pandas as pd
import numpy as np
import os
from utils import safe_request_json, normalize_team_name
from datetime import datetime
import logging

logger = logging.getLogger("djbets.data")

# safer ESPN scoreboard endpoint
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

def load_historical(path="data/nfl_archive_10Y.json"):
    if not os.path.exists(path):
        logger.info("[load_historical] no historical file at %s", path)
        return pd.DataFrame()
    try:
        df = pd.read_json(path)
        df.columns = [c.lower().strip() for c in df.columns]
        # ensure columns
        for c in ["home_team","away_team","home_score","away_score","season","week"]:
            if c not in df.columns:
                df[c] = np.nan
        # canonicalize team names
        df["home_team"] = df["home_team"].apply(lambda x: normalize_team_name(x) or x)
        df["away_team"] = df["away_team"].apply(lambda x: normalize_team_name(x) or x)
        return df
    except Exception as e:
        logger.exception("load_historical error: %s", e)
        return pd.DataFrame()

def load_local_schedule(path="data/schedule.csv"):
    if not os.path.exists(path):
        logger.info("[load_local_schedule] schedule.csv not found at %s", path)
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        for c in ["season","week","home_team","away_team"]:
            if c not in df.columns:
                df[c] = np.nan
        df["home_team"] = df["home_team"].apply(lambda x: normalize_team_name(x) or x)
        df["away_team"] = df["away_team"].apply(lambda x: normalize_team_name(x) or x)
        return df
    except Exception as e:
        logger.exception("load_local_schedule error: %s", e)
        return pd.DataFrame()

def fetch_espn_week(season:int, week:int):
    params = {"season": season, "week": week, "seasontype": 2}
    data = safe_request_json(ESPN_SCOREBOARD_URL, params=params)
    if not data or "events" not in data:
        return pd.DataFrame()
    rows = []
    for ev in data.get("events", []):
        comps = ev.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue
        home_obj = None
        away_obj = None
        for c in competitors:
            if c.get("homeAway") == "home":
                home_obj = c
            else:
                away_obj = c
        if home_obj is None or away_obj is None:
            continue
        home_name = home_obj.get("team", {}).get("displayName")
        away_name = away_obj.get("team", {}).get("displayName")
        home_score = home_obj.get("score")
        away_score = away_obj.get("score")
        status = comp.get("status", {}).get("type", {}).get("name", "unknown")
        rows.append({
            "season": season,
            "week": week,
            "home_team": normalize_team_name(home_name) or home_name,
            "away_team": normalize_team_name(away_name) or away_name,
            "home_score": home_score,
            "away_score": away_score,
            "status": status
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def fetch_espn_season(season:int, weeks=18):
    frames = []
    for w in range(1, weeks+1):
        df = fetch_espn_week(season, w)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def load_or_fetch_schedule(season:int=2025, weeks=18):
    # Prefer ESPN season fetch, fallback to local schedule.csv
    espn = fetch_espn_season(season, weeks=weeks)
    if not espn.empty:
        return espn
    local = load_local_schedule()
    return local

def merge_schedule_and_history(schedule_df, hist_df):
    # keep schedule normalized; history is used for model training separately
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame()
    df = schedule_df.copy()
    # ensure canonical strings
    df["home_team"] = df["home_team"].astype(str).apply(lambda x: normalize_team_name(x) or x)
    df["away_team"] = df["away_team"].astype(str).apply(lambda x: normalize_team_name(x) or x)
    return df