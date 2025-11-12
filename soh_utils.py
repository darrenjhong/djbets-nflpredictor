# soh_utils.py
# Helper utilities for ESPN + SOH merging and logo lookups.
# IMPORTANT: do NOT call streamlit functions here (no st.*). Keep this file purely utility functions.

import os
import re
import json
import time
import math
import requests
import pandas as pd
from typing import Tuple

DATA_DIR = "./data"

# -------------------------
# ESPN schedule scraper (simple)
# -------------------------
def _normalize_team_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", name.lower())

def load_espn_schedule(season: int) -> pd.DataFrame:
    """
    Load schedule from ESPN public endpoints by season.
    This is a resilient best-effort function and returns an empty DataFrame on failure.
    """
    try:
        # ESPN has per-week pages; but to keep lightweight we try ESPN scoreboard API for current season
        # Fallback: if offline, attempt to load ./data/schedule_{season}.csv or json
        # --- simple fallback check ---
        local_csv = os.path.join(DATA_DIR, f"schedule_{season}.csv")
        local_json = os.path.join(DATA_DIR, f"schedule_{season}.json")
        if os.path.exists(local_csv):
            df = pd.read_csv(local_csv)
            return df
        if os.path.exists(local_json):
            with open(local_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            return pd.DataFrame(data)
        # Try ESPN scoreboard API for current week range (best-effort)
        # NOTE: ESPN endpoints change; we attempt basic scoreboard for season and week range 1..18
        rows = []
        for wk in range(1, 19):
            url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?week={wk}&season={season}"
            try:
                resp = requests.get(url, timeout=6)
                if resp.status_code != 200:
                    continue
                js = resp.json()
                events = js.get("events", [])
                for e in events:
                    competitions = e.get("competitions", [])
                    if not competitions:
                        continue
                    comp = competitions[0]
                    # determine teams
                    home = None; away = None; home_score=None; away_score=None
                    for competitor in comp.get("competitors", []):
                        if competitor.get("homeAway") == "home":
                            home = competitor.get("team", {}).get("abbreviation") or competitor.get("team", {}).get("displayName")
                            home_score = competitor.get("score")
                        else:
                            away = competitor.get("team", {}).get("abbreviation") or competitor.get("team", {}).get("displayName")
                            away_score = competitor.get("score")
                    kickoff = comp.get("date")
                    # try extracting spreads and totals from lines (ESPN may include in 'odds' field)
                    spread = None; over_under = None
                    odds = comp.get("odds", [])
                    if odds:
                        o = odds[0]
                        spread = o.get("spread")
                        over_under = o.get("total")
                    rows.append({
                        "season": season,
                        "week": wk,
                        "home_team": home,
                        "away_team": away,
                        "home_score": home_score,
                        "away_score": away_score,
                        "kickoff_et": kickoff,
                        "spread": spread,
                        "over_under": over_under
                    })
            except Exception:
                continue
            time.sleep(0.1)
        df = pd.DataFrame(rows)
        return df
    except Exception:
        return pd.DataFrame()

# -------------------------
# SOH (SportsOddsHistory) loader — best-effort local or remote
# -------------------------
def load_soh_data(season:int) -> pd.DataFrame:
    """
    Load SportsOddsHistory (SOH) pre-downloaded CSV/JSON in /data. If not found or malformed,
    return empty DataFrame.
    """
    possible = [
        os.path.join(DATA_DIR, f"soh_{season}.csv"),
        os.path.join(DATA_DIR, f"soh_{season}.json"),
        os.path.join(DATA_DIR, "soh_all.csv"),
        os.path.join(DATA_DIR, "soh_all.json")
    ]
    for p in possible:
        if os.path.exists(p):
            try:
                if p.endswith(".csv"):
                    df = pd.read_csv(p)
                else:
                    with open(p,"r",encoding="utf-8") as f:
                        df = pd.DataFrame(json.load(f))
                # minimal validation
                req_cols = {"week","home","away","spread","total"}
                if not req_cols.intersection(set(df.columns)):
                    # malformed
                    return pd.DataFrame()
                return df
            except Exception:
                continue
    return pd.DataFrame()

def merge_espn_soh(espn_df: pd.DataFrame, soh_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge ESPN schedule with SOH spreads when possible.
    If SOH is empty or missing columns for a given week, fall back to ESPN only.
    This function tries to be conservative and avoid raising.
    """
    if espn_df is None or espn_df.empty:
        return pd.DataFrame()
    esp = espn_df.copy()
    soh = soh_df.copy() if soh_df is not None else pd.DataFrame()
    # unify column names
    esp = esp.rename(columns={
        "home_team":"home_team","away_team":"away_team","kickoff_et":"kickoff_et"
    })
    # If SOH present and has week/home/away/spread, merge
    if not soh.empty:
        # normalize team abbreviations for join
        def norm(x):
            try:
                if pd.isna(x): return ""
                return re.sub(r"[^A-Za-z0-9]","", str(x)).lower()
            except Exception:
                return ""
        soh["home_norm"] = soh.get("home", soh.get("home_team", "")).apply(norm) if "home" in soh.columns or "home_team" in soh.columns else soh.get("home", "").apply(norm)
        soh["away_norm"] = soh.get("away", soh.get("away_team", "")).apply(norm) if "away" in soh.columns or "away_team" in soh.columns else soh.get("away", "").apply(norm)
        esp["home_norm"] = esp["home_team"].fillna("").apply(norm)
        esp["away_norm"] = esp["away_team"].fillna("").apply(norm)
        # merge on week + normalized teams
        try:
            merged = pd.merge(esp, soh[["week","home_norm","away_norm","spread","total"]], left_on=["week","home_norm","away_norm"], right_on=["week","home_norm","away_norm"], how="left")
            # rename total->over_under
            if "total" in merged.columns:
                merged = merged.rename(columns={"total":"over_under"})
            return merged
        except Exception:
            # if merge fails, return espn only
            return esp
    return esp

# -------------------------
# Logos & filenames
# -------------------------
def get_logo_path(team: str) -> str:
    """
    Given a team abbreviation / name, return the first local logo path found.
    Look in common folders (public/logos, public, logos).
    """
    if team is None:
        return None
    name = str(team).strip()
    # try common abbr -> file
    candidates = []
    abbr = re.sub(r"[^A-Za-z0-9]", "", name).lower()
    # typical filenames: pats.png, NE.png, newengland.png, patriots.png
    paths_to_check = [
        f"./public/logos/{abbr}.png",
        f"./public/{abbr}.png",
        f"./logos/{abbr}.png",
        f"./public/logos/{name}.png",
        f"./public/{name}.png",
        f"./logos/{name}.png",
    ]
    # Also attempt uppercase abbr
    paths_to_check += [p.replace(".png", ".svg") for p in paths_to_check]
    for p in paths_to_check:
        if os.path.exists(p):
            return p
    # not found
    return None

# -------------------------
# Helpers to coerce numeric columns
# -------------------------
def ensure_numeric_cols(df: pd.DataFrame, cols: list):
    """Ensure columns exist and are numeric (inplace)."""
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        except Exception:
            df[c] = 0.0