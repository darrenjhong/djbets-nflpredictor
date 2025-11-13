# utils.py
# Shared helpers for DJBets NFL Predictor (safe HTTP, Elo, ROI, misc utils)

import requests
import pandas as pd
import numpy as np
import time
import traceback

# -----------------------------
# Safe HTTP request wrapper
# -----------------------------
def safe_request_json(url, params=None, retries=3, timeout=10):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            time.sleep(1)
        except Exception:
            time.sleep(1)
    return None


# -----------------------------
# Simple Elo Calculation
# -----------------------------
def compute_simple_elo(df):
    """
    Computes simple ELO values based on game outcomes.
    df must contain:
    - home_team, away_team
    - home_score, away_score
    """

    if "home_score" not in df.columns or "away_score" not in df.columns:
        # Fallback for incomplete historical archives (your case)
        df["home_elo"] = 1500
        df["away_elo"] = 1500
        return df

    elo = {}

    def get_elo(team):
        return elo.get(team, 1500)

    def update_elo(team, new_rating):
        elo[team] = new_rating

    K = 20

    home_elo_list = []
    away_elo_list = []

    for _, row in df.iterrows():
        h = row["home_team"]
        a = row["away_team"]

        h_elo = get_elo(h)
        a_elo = get_elo(a)

        home_elo_list.append(h_elo)
        away_elo_list.append(a_elo)

        # Score
        if row["home_score"] > row["away_score"]:
            s_home, s_away = 1, 0
        elif row["home_score"] < row["away_score"]:
            s_home, s_away = 0, 1
        else:
            s_home, s_away = 0.5, 0.5

        # Expected probabilities
        expected_home = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
        expected_away = 1 - expected_home

        # Updates
        new_h = h_elo + K * (s_home - expected_home)
        new_a = a_elo + K * (s_away - expected_away)

        update_elo(h, new_h)
        update_elo(a, new_a)

    df["home_elo"] = home_elo_list
    df["away_elo"] = away_elo_list

    return df


# -----------------------------
# ROI + Model Record
# -----------------------------
def compute_roi(df):
    """
    Computes:
    - ROI
    - PnL
    - model record: wins/losses
    Expects df to contain:
    - recommended (bool)
    - actual_result (1 if correct, 0 if wrong)
    """
    if "recommended" not in df.columns:
        return 0, 0, 0

    bets = df[df["recommended"] == True]

    if bets.empty:
        return 0, 0, 0

    wins = (bets["actual_result"] == 1).sum()
    losses = (bets["actual_result"] == 0).sum()

    pnl = wins * 1 - losses * 1  # flat unit betting
    roi = pnl / max(len(bets), 1)

    return pnl, len(bets), roi