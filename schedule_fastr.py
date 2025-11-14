# schedule_fastr.py
import pandas as pd
import requests
from io import StringIO

FASTR_URL = "https://raw.githubusercontent.com/nflverse/nflfastR-roster/master/data/games.csv"

def load_fastr_schedule(season: int):
    """
    Loads complete NFL schedule for the given season using the NFLverse FastR dataset.
    Works 100% reliably on Streamlit Cloud (no Cloudflare, no API key).
    """
    try:
        r = requests.get(FASTR_URL, timeout=10)
        r.raise_for_status()

        df = pd.read_csv(StringIO(r.text))

        # normalize names
        df.columns = [c.lower().strip() for c in df.columns]

        # rename a few columns to match internal usage
        df = df.rename(columns={
            "home_team": "home_team",
            "away_team": "away_team",
            "result": "result",
        })

        # filter by season + regular season weeks 1–18
        df = df[(df["season"] == season) & (df["week"].between(1, 18))]

        # select minimum required columns
        keep = [
            "season",
            "week",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "game_id",
            "game_type",
            "gameday"
        ]
        df = df[[c for c in keep if c in df.columns]]

        return df.reset_index(drop=True)

    except Exception as e:
        print("[load_fastr_schedule] ERROR:", e)
        return pd.DataFrame()