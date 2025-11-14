# schedule_fastr.py
import pandas as pd
import requests
from io import StringIO

FASTR_URL = "https://raw.githubusercontent.com/nflverse/nflverse-data/master/data/games.csv"

def load_fastr_schedule(season: int):
    """
    Loads full NFL schedule from NFLverse (fastR data source).
    This endpoint is stable and always contains week, home_team, away_team,
    scores, game_id, and more.
    """
    try:
        r = requests.get(FASTR_URL, timeout=10)
        r.raise_for_status()

        df = pd.read_csv(StringIO(r.text))

        # Standardize columns
        df.columns = [c.lower().strip() for c in df.columns]

        # Ensure required columns exist
        needed = {"season", "week", "home_team", "away_team"}
        if not needed.issubset(set(df.columns)):
            print("[load_fastr_schedule] Missing expected columns:", df.columns)
            return pd.DataFrame()

        # Filter season + NFL regular season
        df = df[(df["season"] == season) & (df["week"].between(1, 18))]

        keep = [
            "season",
            "week",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "game_id",
            "game_type",
            "gameday",
        ]
        df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)

        return df

    except Exception as e:
        print("[load_fastr_schedule] ERROR:", e)
        return pd.DataFrame()