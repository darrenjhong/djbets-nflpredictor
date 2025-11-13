# utils.py
import requests
import json
import pandas as pd
import numpy as np
import os

def safe_request_json(url, params=None, headers=None):
    try:
        r = requests.get(url, params=params, headers=headers, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def compute_simple_elo(hist_df):
    """
    If historical scores missing, fallback to dummy ELO = 1500 baseline.
    """
    teams = pd.unique(hist_df[["home_team", "away_team"]].values.ravel("K"))
    return {t: 1500 for t in teams}


def normalize_team(s):
    return str(s).lower().strip()


def compute_roi(hist_df):
    if "recommended" not in hist_df.columns:
        return 0, 0, 0

    bets = hist_df[hist_df["recommended"] == True]
    if bets.empty:
        return 0, 0, 0

    pnl = bets["edge"].sum()
    roi = pnl / len(bets)
    return pnl, len(bets), roi


def get_logo_path(team):
    """
    Teams are in canonical format (ex: 'chicago_bears').
    Logo must exist in /logos folder.
    """
    t = normalize_team(team).replace(" ", "_")
    path = f"logos/{t}.png"
    if os.path.exists(path):
        return path
    return "logos/default.png"