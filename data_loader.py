# data_loader.py
"""
Schedule + history loader and week prepare helper
- Prefer local data/schedule.csv if present
- Try ESPN scoreboard for current season (best-effort, handles 500s)
- Merge in covers spreads if provided
"""

import os
import json
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime
import time

from team_logo_map import canonical_team_name
from utils import safe_request_json

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

def load_historical(path="data/nfl_archive_10Y.json") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        with open(path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df.columns = [c.lower().strip() for c in df.columns]
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
        return df
    except Exception:
        return pd.DataFrame()

def fetch_espn_week(season:int, week:int) -> pd.DataFrame:
    params = {"season": season, "seasontype": 2, "week": week}
    try:
        resp = safe_request_json(ESPN_SCOREBOARD_URL, params=params)
        if not resp or "events" not in resp:
            return pd.DataFrame()
        events = resp.get("events", [])
        rows=[]
        for e in events:
            comps = e.get("competitions", [])
            if not comps: continue
            comp = comps[0]
            competitors = comp.get("competitors", [])
            if len(competitors) != 2: continue
            # find home/away
            home = [c for c in competitors if c.get("homeAway")=="home"][0]
            away = [c for c in competitors if c.get("homeAway")=="away"][0]
            rows.append({
                "season": season,
                "week": week,
                "home_team": canonical_team_name(home.get("team", {}).get("displayName","")),
                "away_team": canonical_team_name(away.get("team", {}).get("displayName","")),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "status": comp.get("status", {}).get("type", {}).get("name", "unknown")
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def fetch_espn_season(season:int, weeks=18) -> pd.DataFrame:
    frames=[]
    for w in range(1, weeks+1):
        df = fetch_espn_week(season, w)
        if not df.empty:
            frames.append(df)
        time.sleep(0.3)
    if not frames: return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def load_or_fetch_schedule(season:int) -> pd.DataFrame:
    # local CSV first
    local = load_local_schedule()
    if not local.empty:
        return local
    # try ESPN
    espn = fetch_espn_season(season)
    if not espn.empty:
        return espn
    return pd.DataFrame()

def prepare_week_schedule(season:int, week:int, schedule_df:Optional[pd.DataFrame]=None, covers_df:Optional[pd.DataFrame]=None) -> pd.DataFrame:
    """
    Returns a DataFrame with rows for the week. Fields: season, week, home_team, away_team, spread, over_under, status
    `covers_df` expected columns: home/away/spread/over_under (strings; canonical names or plain)
    """
    out = pd.DataFrame()
    sched = schedule_df if schedule_df is not None else load_or_fetch_schedule(season)
    if sched is None or sched.empty:
        return out

    # normalize
    df = sched.copy()
    # ensure week col exists
    if "week" not in df.columns:
        return out
    df = df[df["week"] == int(week)].copy()
    if df.empty:
        return out

    # Normalize team names
    def canonical_display(x):
        try:
            return canonical_team_name(str(x).lower())
        except Exception:
            return str(x).lower().replace(" ", "_")

    df["home_team"] = df["home_team"].astype(str).apply(canonical_display)
    df["away_team"] = df["away_team"].astype(str).apply(canonical_display)

    # attach covers spreads if available (try to match by team names, defensive)
    if covers_df is not None and not covers_df.empty:
        cov = covers_df.copy()
        # canonicalize covers team names similarly
        cov["home_canon"] = cov["home"].astype(str).apply(canonical_display)
        cov["away_canon"] = cov["away"].astype(str).apply(canonical_display)
        # join by (home, away)
        merged = pd.merge(df, cov, left_on=["home_team","away_team"], right_on=["home_canon","away_canon"], how="left")
        merged["spread"] = merged["spread"].astype(float, errors="ignore")
        merged["over_under"] = merged["over_under"].astype(float, errors="ignore")
        merged["status"] = merged.get("status", "upcoming")
        return merged[["season","week","home_team","away_team","spread","over_under","status","home_score","away_score"]]
    else:
        df["spread"] = np.nan
        df["over_under"] = np.nan
        df["status"] = df.get("status", "upcoming")
        # ensure columns present
        for c in ["home_score","away_score"]:
            if c not in df.columns:
                df[c] = np.nan
        return df[["season","week","home_team","away_team","spread","over_under","status","home_score","away_score"]]