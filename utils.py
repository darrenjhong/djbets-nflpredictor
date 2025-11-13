# utils.py
import requests
import time
import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Simple HTTP JSON helper with backoff
def safe_request_json(url, params=None, headers=None, timeout=8, retries=2, backoff=0.8):
    headers = headers or {"User-Agent": "DJBetsBot/1.0"}
    for i in range(retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logging.debug("safe_request_json failed (%s) attempt=%d: %s", url, i, e)
            if i == retries:
                return None
            time.sleep(backoff * (i + 1))
    return None


# canonicalize team name -> filename-friendly
def normalize_team_name(name: str) -> str:
    if not name or pd.isna(name):
        return ""
    s = str(name).lower().strip()
    # remove punctuation
    for ch in [".", ",", "’", "'", ":", ";", "®", "™", "(", ")"]:
        s = s.replace(ch, "")
    s = s.replace(" ", "_")
    s = s.replace("--", "-")
    return s


def get_logo_path(team_name: str, logos_dir="public/logos") -> str:
    """
    Return local path for a team logo file. Does not raise.
    If not exists, returns empty string.
    """
    if not team_name:
        return ""
    fname = normalize_team_name(team_name)
    # try png then jpg
    for ext in ("png", "jpg", "jpeg", "webp"):
        p = os.path.join(logos_dir, f"{fname}.{ext}")
        if os.path.exists(p):
            return p
    # maybe user stored canonical names like 'CHI_bears.png' — try direct
    for ext in ("png", "jpg", "jpeg", "webp"):
        p = os.path.join(logos_dir, f"{team_name}.{ext}")
        if os.path.exists(p):
            return p
    return ""


# Simple Elo aggregator — returns dict of team->current_elo
def compute_simple_elo(historical_df, base_elo=1500, k=20):
    """
    historical_df expected to include columns: season, week, home_team, away_team,
    home_score, away_score (scores may be numeric).
    Returns dict mapping canonical team string -> elo value.
    This is intentionally simple and robust to missing columns.
    """
    elos = {}
    def ensure(t):
        if pd.isna(t) or t == "":
            return
        if t not in elos:
            elos[t] = base_elo

    cols = set(historical_df.columns)
    if not {"home_team", "away_team", "home_score", "away_score"}.issubset(cols):
        return elos

    df = historical_df.copy()
    df = df.dropna(subset=["home_team", "away_team"])
    # sort by season/week if available
    if "season" in df.columns and "week" in df.columns:
        try:
            df = df.sort_values(["season", "week"])
        except Exception:
            pass

    for _, r in df.iterrows():
        h = r.get("home_team")
        a = r.get("away_team")
        hs = r.get("home_score")
        as_ = r.get("away_score")
        if pd.isna(h) or pd.isna(a):
            continue
        ensure(h); ensure(a)
        try:
            hs = float(hs)
            as_ = float(as_)
        except Exception:
            continue
        if hs == as_:
            # treat as half-win
            res_h = 0.5
            res_a = 0.5
        elif hs > as_:
            res_h = 1.0
            res_a = 0.0
        else:
            res_h = 0.0
            res_a = 1.0
        eh = elos.get(h, base_elo)
        ea = elos.get(a, base_elo)
        expected_h = 1.0 / (1.0 + 10 ** ((ea - eh) / 400.0))
        expected_a = 1.0 - expected_h
        elos[h] = eh + k * (res_h - expected_h)
        elos[a] = ea + k * (res_a - expected_a)
    return elos


def compute_roi(hist_df, edge_pp_threshold=3.0, stake=1.0):
    """
    hist_df expected to include predicted_label, actual result, and market implied probability (or spread)
    This function is a simple placeholder producing (pnl, bets_count, roi_percent)
    We'll be defensive: if required columns missing, return zeros.
    """
    # required columns are flexible — try to find recommended flag
    if hist_df is None or hist_df.empty:
        return 0.0, 0, 0.0

    s = hist_df.copy()
    # if a 'recommended' boolean column exists, use it; otherwise attempt to infer from 'edge_pp'
    if "recommended" in s.columns:
        bets = s[s["recommended"] == True]
    elif "edge_pp" in s.columns:
        bets = s[s["edge_pp"].abs() >= edge_pp_threshold]
    else:
        # nothing to bet on
        return 0.0, 0, 0.0

    # must have 'profit' column or compute from actual result vs market — fallback to zeros
    if "profit" in bets.columns:
        pnl = bets["profit"].sum()
    else:
        pnl = 0.0
    bets_made = len(bets)
    roi = (pnl / (bets_made * stake)) * 100 if bets_made > 0 else 0.0
    return pnl, bets_made, roi