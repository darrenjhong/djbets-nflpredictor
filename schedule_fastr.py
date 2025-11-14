# schedule_fastr.py
import pandas as pd
import requests
from io import StringIO

FASTR_URL = "https://raw.githubusercontent.com/nflverse/nflverse-data/master/games/games.csv"

def load_fastr_schedule(season: int):
    """
    Loads complete NFL schedule for the given season using the NFLverse dataset.
    This file ALWAYS contains full schedules (past + future) for all seasons.
    """
    try:
        r = requests.get(FASTR_URL, timeout=15)
        r.raise_for_status()

        df = pd.read_csv(StringIO(r.text))

        # Normalize
        df.columns = [c.lower().strip() for c in df.columns]

        # Keep only regular season
        df = df[(df["season"] == season) & (df["week"].between(1, 18))]

        if df.empty:
            return pd.DataFrame()

        # Canonical fields our app expects
        cleaned = pd.DataFrame({
            "season": df["season"],
            "week": df["week"],
            "home_team": df["home_team"].astype(str),
            "away_team": df["away_team"].astype(str),
            "home_score": df.get("home_score", pd.NA),
            "away_score": df.get("away_score", pd.NA),
            "game_id": df.get("game_id", pd.NA),
            "gameday": df.get("gameday", pd.NA),
        })

        return cleaned.reset_index(drop=True)

    except Exception as e:
        print("FAST-R LOAD ERROR:", e)
        return pd.DataFrame()