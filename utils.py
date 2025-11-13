# utils.py
import numpy as np
import pandas as pd

def compute_simple_elo(hist_df: pd.DataFrame, k:float=20.0, base=1500):
    """
    Compute rolling Elo per team using historical results (home_score/away_score).
    Returns DataFrame with columns home_elo_pre, away_elo_pre for each historical row.
    """
    if hist_df.empty:
        return hist_df
    teams = {}
    home_elo_pre = []
    away_elo_pre = []
    for _, r in hist_df.iterrows():
        home = r.get("home_team")
        away = r.get("away_team")
        h_elo = teams.get(home, base)
        a_elo = teams.get(away, base)
        home_elo_pre.append(h_elo)
        away_elo_pre.append(a_elo)
        # compute expected
        exp_h = 1.0 / (1 + 10 ** ((a_elo - h_elo)/400))
        # result
        hs = r.get("home_score"); ascore = r.get("away_score")
        if pd.isna(hs) or pd.isna(ascore):
            continue
        result_h = 1.0 if hs>ascore else (0.5 if hs==ascore else 0.0)
        teams[home] = h_elo + k*(result_h - exp_h)
        teams[away] = a_elo + k*((1-result_h) - (1-exp_h))
    hist_df = hist_df.copy()
    hist_df["home_elo_pre"] = home_elo_pre
    hist_df["away_elo_pre"] = away_elo_pre
    hist_df["elo_diff"] = hist_df["home_elo_pre"] - hist_df["away_elo_pre"]
    return hist_df

def compute_roi(df):
    """
    Compute a simple ROI on recommended bets (flag 'recommended' True) with unit = 1
    Expected to have columns: recommended (bool), result (1 if won, 0 if lost), spread_hit (bool), pnl
    """
    if df is None or df.empty:
        return 0.0, 0, 0.0
    bets = df[df.get("recommended", False)]
    if bets.empty:
        return 0.0, 0, 0.0
    bets_made = bets.shape[0]
    # we expect 'pnl' column
    if "pnl" in bets.columns:
        total = bets["pnl"].sum()
    else:
        # approximate: winner pays +1 on win, -1 on loss (not realistic)
        total = ((bets["result"]==1).sum() - (bets["result"]==0).sum())
    roi = total / bets_made
    return float(total), int(bets_made), float(roi)