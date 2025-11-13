# data_loader.py
import pandas as pd
import numpy as np
import os
import json
from utils import safe_request_json
from team_logo_map import canonical_team_name

ESPN_WEEK_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/types/2/weeks/{week}/events"


def load_historical(path="data/nfl_archive_10Y.json"):
    if not os.path.exists(path):
        return pd.DataFrame()

    with open(path, "r") as f:
        js = json.load(f)

    df = pd.DataFrame(js)
    df.columns = [c.lower().strip() for c in df.columns]

    # required
    for c in ["home_team", "away_team", "home_score", "away_score"]:
        if c not in df:
            df[c] = np.nan

    df["home_team"] = df["home_team"].astype(str).str.lower().str.strip()
    df["away_team"] = df["away_team"].astype(str).str.lower().str.strip()

    return df


def load_local_schedule(path="data/schedule.csv"):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    for c in ["season", "week", "home_team", "away_team"]:
        if c not in df.columns:
            df[c] = np.nan

    df["home_team"] = df["home_team"].astype(str)
    df["away_team"] = df["away_team"].astype(str)
    return df


def fetch_espn_week(season, week):
    url = ESPN_WEEK_URL.format(season=season, week=week)
    js = safe_request_json(url)

    if not js or "events" not in js:
        return pd.DataFrame()

    rows = []
    for ev in js["events"]:
        comp = ev["competitions"][0]
        teams = comp["competitors"]

        home = next(t for t in teams if t["homeAway"] == "home")
        away = next(t for t in teams if t["homeAway"] == "away")

        rows.append({
            "season": season,
            "week": week,
            "home_team": canonical_team_name(home["team"]["displayName"]),
            "away_team": canonical_team_name(away["team"]["displayName"]),
            "home_score": home.get("score"),
            "away_score": away.get("score"),
            "status": comp.get("status", {}).get("type", {}).get("name", "unknown"),
        })

    return pd.DataFrame(rows)


def fetch_espn_season(season, weeks=18):
    frames = []
    for w in range(1, weeks + 1):
        df = fetch_espn_week(season, w)
        if not df.empty:
            frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def load_or_fetch_schedule(season):
    espn = fetch_espn_season(season)
    if not espn.empty:
        return espn
    return load_local_schedule()