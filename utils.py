# utils.py
import requests, time, logging
import pandas as pd
import numpy as np
from datetime import datetime
from team_logo_map import canonical_from_string, lookup_logo as lookup_logo_internal

logger = logging.getLogger("djbets")
logger.setLevel(logging.INFO)

def safe_request_json(url, params=None, headers=None, timeout=10, max_retries=2, backoff=0.5):
    """
    Robust JSON fetch with retries. Returns dict or None on failure.
    """
    headers = headers or {"User-Agent": "DJBetsBot/1.0"}
    for i in range(max_retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.debug(f"[safe_request_json] attempt {i} failed: {e}")
            if i < max_retries:
                time.sleep(backoff * (2 ** i))
            else:
                return None

def normalize_team_name(s):
    """
    Return canonical team name (City + Mascot) or None.
    Uses team_logo_map.canonical_from_string.
    """
    return canonical_from_string(s)

def get_logo_path(team_name):
    """
    team_name: canonical (City + Mascot) or any raw string.
    Returns local path (string) or None.
    """
    if not team_name:
        return None
    # if not canonical, try to canonicalize
    canonical = canonical_from_string(team_name) or team_name
    return lookup_logo_internal(canonical)

# --- SIMPLE ELO helper (very lightweight) ---
def compute_simple_elo(history_df, k=20, base=1500):
    """
    Accepts a historical dataframe with columns:
       season, week, home_team, away_team, home_score, away_score
    Returns dataframe with elo_home and elo_away columns (for each game)
    Very small, incremental ELO calculation per season.
    """
    if history_df is None or history_df.empty:
        return pd.DataFrame()

    df = history_df.copy()
    df = df.sort_values(["season", "week"]).reset_index(drop=True)
    teams = {}
    out_rows = []
    for _, r in df.iterrows():
        home = str(r.get("home_team", "")).strip()
        away = str(r.get("away_team", "")).strip()
        home_score = r.get("home_score")
        away_score = r.get("away_score")
        if pd.isna(home_score) or pd.isna(away_score):
            # skip unlabeled game for ELO update, but still return current elos
            h_elo = teams.get(home, base)
            a_elo = teams.get(away, base)
            out_rows.append({"home":home, "away":away, "elo_home":h_elo, "elo_away":a_elo})
            continue
        h_elo = teams.get(home, base)
        a_elo = teams.get(away, base)
        # expected
        exp_home = 1.0 / (1 + 10 ** ((a_elo - h_elo) / 400))
        # result
        if home_score > away_score:
            s_home = 1.0
        elif home_score < away_score:
            s_home = 0.0
        else:
            s_home = 0.5
        # update
        h_new = h_elo + k * (s_home - exp_home)
        a_new = a_elo + k * ((1 - s_home) - (1 - exp_home))
        teams[home] = h_new
        teams[away] = a_new
        out_rows.append({"home":home, "away":away, "elo_home":h_elo, "elo_away":a_elo})
    return pd.DataFrame(out_rows)

def compute_roi(history_df):
    """
    Very simple ROI calc: expects 'recommended' boolean and 'result' (1 correct, 0 wrong), 'bet_size' column.
    Returns pnl, bets_made, roi_percent
    """
    if history_df is None or history_df.empty:
        return 0.0, 0, 0.0
    df = history_df.copy()
    if "recommended" not in df.columns or "result" not in df.columns:
        return 0.0, 0, 0.0
    bets = df[df["recommended"] == True]
    if bets.empty:
        return 0.0, 0, 0.0
    bets["bet_size"] = bets.get("bet_size", 1.0)
    bets["pnl"] = np.where(bets["result"]==1, bets["bet_size"]*1.0, -bets["bet_size"])
    pnl = bets["pnl"].sum()
    bets_made = len(bets)
    total_staked = (bets["bet_size"]).sum()
    roi = (pnl / total_staked) * 100 if total_staked>0 else 0.0
    return float(pnl), int(bets_made), float(roi)