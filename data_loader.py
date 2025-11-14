# data_loader.py
import os
import json
import time
import requests
import pandas as pd
from typing import Optional

from team_logo_map import canonical_team_name
from utils import safe_request_json

DATA_DIR = "data"
ESPN_SCOREBOARD_WEEK = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?season={season}&week={week}&seasontype=2"

def load_historical(path=os.path.join(DATA_DIR, "nfl_archive_10Y.json")) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        # normalize names to canonical
        for col in ["home_team", "away_team"]:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(lambda s: canonical_team_name(s.lower()))
        # ensure scores
        for c in ["home_score", "away_score"]:
            if c not in df.columns:
                df[c] = pd.NA
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()

def load_local_schedule(path=os.path.join(DATA_DIR, "schedule.csv")) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        # canonicalize team names
        for col in ["home_team", "away_team"]:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(lambda s: canonical_team_name(s.lower()))
        return df
    except Exception:
        return pd.DataFrame()

def fetch_espn_week(season: int, week: int) -> pd.DataFrame:
    url = ESPN_SCOREBOARD_WEEK.format(season=season, week=week)
    data = safe_request_json(url)
    if not data:
        return pd.DataFrame()
    events = data.get("events", [])
    rows = []
    for ev in events:
        try:
            comp = ev.get("competitions", [])[0]
            competitors = comp.get("competitors", [])
            if len(competitors) != 2:
                continue
            home = next((t for t in competitors if t.get("homeAway") == "home"), competitors[0])
            away = next((t for t in competitors if t.get("homeAway") == "away"), competitors[1])
            home_name = home.get("team", {}).get("displayName", "")
            away_name = away.get("team", {}).get("displayName", "")
            rows.append({
                "season": season,
                "week": week,
                "home_team": canonical_team_name(home_name.lower()),
                "away_team": canonical_team_name(away_name.lower()),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "status": comp.get("status", {}).get("type", {}).get("name", "unknown"),
            })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def fetch_espn_season(season: int, weeks: int = 18) -> pd.DataFrame:
    frames = []
    for w in range(1, weeks + 1):
        try:
            df = fetch_espn_week(season, w)
            if not df.empty:
                frames.append(df)
            time.sleep(0.2)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def load_or_fetch_schedule(season: int) -> pd.DataFrame:
    # local CSV is used as fallback
    local = load_local_schedule()
    try:
        espn = fetch_espn_season(season)
    except Exception:
        espn = pd.DataFrame()
    if not espn.empty:
        return espn
    return local
