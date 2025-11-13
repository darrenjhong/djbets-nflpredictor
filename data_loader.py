# data_loader.py
import os
import json
import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime
from utils import safe_request_json
from team_logo_map import lookup_logo, canonical_from_string

ESPN_SCOREBOARD_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/types/2/weeks/{week}/events"

def load_historical(path="data/nfl_archive_10Y.json") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        with open(path, "r", encoding="utf8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        # normalize lowercase columns
        df.columns = [c.lower().strip() for c in df.columns]
        # ensure columns
        for col in ["home_team", "away_team", "home_score", "away_score"]:
            if col not in df.columns:
                df[col] = np.nan
        return df
    except Exception:
        return pd.DataFrame()

def load_local_schedule(path="data/schedule.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        for c in ["season", "week", "home_team", "away_team"]:
            if c not in df.columns:
                df[c] = np.nan
        # normalize team strings
        df["home_team"] = df["home_team"].astype(str).str.strip()
        df["away_team"] = df["away_team"].astype(str).str.strip()
        return df
    except Exception:
        return pd.DataFrame()

def _normalize_competitor_name(cobj):
    # ESPN returns team object -> displayName or team.shortDisplayName
    if not cobj:
        return ""
    try:
        t = cobj.get("team", {})
        name = t.get("displayName") or t.get("shortDisplayName") or t.get("name")
        return str(name).strip()
    except Exception:
        return ""

def fetch_espn_week(season: int, week: int) -> pd.DataFrame:
    url = ESPN_SCOREBOARD_URL.format(season=season, week=week)
    data = safe_request_json(url)
    if not data or "events" not in data:
        return pd.DataFrame()
    rows = []
    for ev in data.get("events", []):
        comps = ev.get("competitions", []) or []
        if not comps:
            continue
        comp = comps[0]
        competitors = comp.get("competitors", []) or []
        if len(competitors) < 2:
            continue
        try:
            # find home/away
            home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
            away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
            home_name = _normalize_competitor_name(home)
            away_name = _normalize_competitor_name(away)
            rows.append({
                "season": season,
                "week": week,
                "home_team": canonical_from_string(home_name),
                "away_team": canonical_from_string(away_name),
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
        df = fetch_espn_week(season, w)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def load_or_fetch_schedule(season: Optional[int] = None):
    # prefer ESPN season if available (current season)
    season = season or datetime.now().year
    local = load_local_schedule()
    espn = fetch_espn_season(season, weeks=18)
    if not espn.empty:
        return espn
    return local