# schedule_fastr.py
import pandas as pd
import requests
from io import StringIO

FASTR_URL = "https://raw.githubusercontent.com/nflverse/nflverse-data/master/schedules/csv/schedules.csv"

def load_fastr_schedule(season: int):
    """
    Loads full NFL schedule for the season using NFLverse schedules.csv.
    Reliable for past + current + future games.
    """
    try:
        r = requests.get(FASTR_URL, timeout=10)
        r.raise_for_status()

        df = pd.read_csv(StringIO(r.text))
        df.columns = [c.lower().strip() for c in df.columns]

        # Filter by year + regular season only
        df = df[(df["season"] == season) & (df["game_type"] == "REG")]

        # Keep needed columns
        keep = [
            "season", "week",
            "home_team", "away_team",
            "home_score", "away_score",
            "game_id", "gameday"
        ]
        df = df[[c for c in keep if c in df.columns]]

        return df.reset_index(drop=True)

    except Exception as e:
        print("[load_fastr_schedule] ERROR:", e)
        return pd.DataFrame()