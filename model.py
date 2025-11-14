# model.py
"""
Lightweight model helpers:
- build_model_from_history(hist_df): compute simple per-team Elo-like average and return structure
- predict_game_row(row, model_info): returns home_prob, model_spread, pred_home_score, pred_away_score
"""

import numpy as np
import pandas as pd
import math
from collections import defaultdict

def build_model_from_history(hist_df: pd.DataFrame):
    """
    Compute per-team average points scored and allowed as a tiny 'model'.
    If hist_df empty/insufficient, we return fallback structure (trained=False).
    """
    info = {"trained": False, "notes": "", "team_avg": {}, "season": None}
    if hist_df is None or hist_df.empty or "home_team" not in hist_df.columns or "home_score" not in hist_df.columns:
        info["notes"] = "insufficient_history"
        return info

    # compute per-team averages
    team_stats = {}
    teams = set(hist_df["home_team"].dropna().unique()).union(set(hist_df["away_team"].dropna().unique()))
    for t in teams:
        # home offense
        home_points = hist_df.loc[hist_df["home_team"]==t, "home_score"].dropna().astype(float)
        away_points = hist_df.loc[hist_df["away_team"]==t, "away_score"].dropna().astype(float)
        scored = pd.concat([home_points, away_points]) if not home_points.empty or not away_points.empty else pd.Series(dtype=float)
        # points allowed
        allowed_home = hist_df.loc[hist_df["home_team"]==t, "away_score"].dropna().astype(float)
        allowed_away = hist_df.loc[hist_df["away_team"]==t, "home_score"].dropna().astype(float)
        allowed = pd.concat([allowed_home, allowed_away]) if not allowed_home.empty or not allowed_away.empty else pd.Series(dtype=float)
        avg_scored = float(scored.mean()) if not scored.empty else 21.0
        avg_allowed = float(allowed.mean()) if not allowed.empty else 21.0
        team_stats[t] = {"scored": avg_scored, "allowed": avg_allowed}
    info["trained"] = True
    info["team_avg"] = team_stats
    info["notes"] = f"teams:{len(team_stats)}"
    return info

def predict_game_row(row: pd.Series, model_info: dict):
    """
    Returns dictionary:
      home_prob (0-1)
      model_spread (home - away, positive => home favored)
      pred_home_score, pred_away_score
    Logic:
      - if team averages exist use them to predict scores
      - otherwise default to 21 - 21
      - home_field_advantage ~ 2.5 points
      - simple logistic function on spread -> prob
    """
    home = row.get("home_team", "")
    away = row.get("away_team", "")
    team_avg = model_info.get("team_avg", {}) if model_info else {}

    home_off = team_avg.get(home, {}).get("scored", 21.0)
    home_def = team_avg.get(home, {}).get("allowed", 21.0)
    away_off = team_avg.get(away, {}).get("scored", 21.0)
    away_def = team_avg.get(away, {}).get("allowed", 21.0)

    # naive predicted points: average of offense and opponent allowed
    pred_home = (home_off + away_def) / 2.0 + 2.5  # home field advantage
    pred_away = (away_off + home_def) / 2.0

    model_spread = pred_home - pred_away  # positive -> home favored
    # convert spread to probability via logistic
    def spread_to_prob(sp):
        # parameter scale ~ 7 => 7 points ~ 70/30 split
        k = 0.18
        prob = 1.0 / (1.0 + math.exp(-k * sp))
        return prob

    home_prob = spread_to_prob(model_spread)

    return {
        "home_prob": home_prob,
        "model_spread": round(model_spread, 1),
        "pred_home_score": round(pred_home, 1),
        "pred_away_score": round(pred_away, 1)
    }