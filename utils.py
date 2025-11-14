# utils.py
<<<<<<< HEAD
import os
import json
import requests
import math
from collections import defaultdict

import numpy as np
import pandas as pd

def safe_request_json(url, params=None, timeout=8):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; DJBetsBot/1.0)"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
=======
import requests
import json
import pandas as pd
import numpy as np
import os

def safe_request_json(url, params=None, headers=None):
    try:
        r = requests.get(url, params=params, headers=headers, timeout=8)
>>>>>>> 83e4cd8c405192af3350849f65cd9e3058c42b44
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

<<<<<<< HEAD
def get_logo_path(canonical_name: str):
    if not canonical_name:
        return None
    path = os.path.join("public", "logos", f"{canonical_name}.png")
    if os.path.exists(path):
        return path
    # try jpg
    path2 = os.path.join("public", "logos", f"{canonical_name}.jpg")
    if os.path.exists(path2):
        return path2
    return None

def compute_simple_elo(hist_df: pd.DataFrame, k: float = 20.0):
    # compute per-team Elo from historical boxed scores
    # returns dict team -> elo
    elo = {}
    teams = set()
    for _, r in hist_df.iterrows():
        ht = str(r.get("home_team", "")).lower()
        at = str(r.get("away_team", "")).lower()
        teams.add(ht)
        teams.add(at)
    for t in teams:
        elo[t] = 1500.0
    # iterate through games (assumes chronological order if available)
    for _, r in hist_df.iterrows():
        try:
            ht = str(r.get("home_team", "")).lower()
            at = str(r.get("away_team", "")).lower()
            hs = float(r.get("home_score", 0) or 0)
            as_ = float(r.get("away_score", 0) or 0)
            if ht == "" or at == "":
                continue
            Ra = 10 ** (elo[ht] / 400.0)
            Rb = 10 ** (elo[at] / 400.0)
            Ea = Ra / (Ra + Rb)
            # outcome
            Sa = 1.0 if hs > as_ else 0.5 if hs == as_ else 0.0
            # update
            elo[ht] = elo[ht] + k * (Sa - Ea)
            elo[at] = elo[at] + k * ((1 - Sa) - (1 - Ea))
        except Exception:
            continue
    return elo

def compute_roi(hist_df: pd.DataFrame, model):
    # very basic ROI simulation: bet 1 unit when model edge >= 5pp
    bets = 0
    pnl = 0.0
    if hist_df is None or hist_df.empty:
        return 0.0, 0, 0.0
    # we need spreads and final scores in hist
    for _, r in hist_df.iterrows():
        try:
            spread = r.get("spread", None)
            if spread is None or pd.isna(spread):
                continue
            # predict prob for the row
            elo_diff = r.get("elo_home", 0) - r.get("elo_away", 0)
            X = {"elo_diff": elo_diff, "spread": float(spread), "over_under": float(r.get("over_under", 0) or 0)}
            prob, _, _ = (None, None, None)
            try:
                prob, _, _ = model.predict_row if False else (model.predict_proba([[elo_diff]])[0][1], None, None)
            except Exception:
                # fallback: use 0.5
                prob = 0.5
            # brute force: if prob > 0.6, bet home at -110
            if prob > 0.6:
                bets += 1
                # determine winner
                home_score = float(r.get("home_score", 0) or 0)
                away_score = float(r.get("away_score", 0) or 0)
                # compute result vs spread
                home_cover = (home_score - away_score) > float(spread)
                if home_cover:
                    pnl += 0.909  # $1 at -110 profit
                else:
                    pnl -= 1.0
        except Exception:
            continue
    roi = (pnl / bets * 100.0) if bets > 0 else 0.0
    return pnl, bets, roi
=======

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
>>>>>>> 83e4cd8c405192af3350849f65cd9e3058c42b44
