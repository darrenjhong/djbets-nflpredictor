# schedule_fastr.py
import pandas as pd
import requests
from io import StringIO

FASTR_URL = "https://raw.githubusercontent.com/nflverse/nflverse-data/master/releases/games.csv"

@st.cache_data(show_spinner=False)
def load_fastr_schedule(season: int):
    """Loads fastR schedule and normalizes all column names safely."""
    try:
        r = requests.get(FASTR_URL, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))

        df.columns = [c.lower().strip() for c in df.columns]

        # ---- FIX: Identify proper week column ----
        week_col = None
        for w in ["week", "game_week", "week_number"]:
            if w in df.columns:
                week_col = w
                break

        if week_col is None:
            print("[FASTR] ERROR: no week column found")
            return pd.DataFrame()

        # ---- FIX: Identify team columns ----
        home_col = None
        away_col = None
        for h in ["home_team", "team_home", "home"]:
            if h in df.columns:
                home_col = h
                break
        for a in ["away_team", "team_away", "away"]:
            if a in df.columns:
                away_col = a
                break

        if home_col is None or away_col is None:
            print("[FASTR] Missing home/away team columns")
            return pd.DataFrame()

        # ---- FIX: Normalize column names ----
        df = df.rename(columns={
            home_col: "home_team",
            away_col: "away_team",
            week_col: "week"
        })

        # ---- FIX: Filter
        df = df[df["season"] == season]
        df = df[df["week"].between(1, 18)]

        # ---- FIX: Score columns
        score_cols = {
            "home_score": ["home_score", "score_home", "points_home"],
            "away_score": ["away_score", "score_away", "points_away"]
        }

        for standard, candidates in score_cols.items():
            for c in candidates:
                if c in df.columns:
                    df = df.rename(columns={c: standard})
                    break
            if standard not in df.columns:
                df[standard] = np.nan

        # Final columns
        keep = [
            "season", "week",
            "home_team", "away_team",
            "home_score", "away_score"
        ]
        df = df[[c for c in keep if c in df.columns]]

        return df.reset_index(drop=True)

    except Exception as e:
        print("[load_fastr_schedule] ERROR:", e)
        return pd.DataFrame()