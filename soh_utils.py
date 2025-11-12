# soh_utils.py
"""
Helpers for ingesting / patching SportsOddsHistory-style data (SOH) and
merging it with ESPN schedule data.

Behavior:
- Loads local /data/nfl_archive_10Y.json (if present) or other csv/json in /data
- Normalizes column names to a predictable schema:
    ['date','season','week','home_team','away_team','home_score','away_score','spread','over_under']
  where 'spread' is the home-side spread (positive => home favored).
- fill_missing_spreads(df): fills missing spread/over_under with plausible defaults
- merge_espn_soh(espn_df, soh_df, week): returns espn rows augmented with SOH spreads/OUs.
"""

import os, json, glob
import pandas as pd
import numpy as np
from datetime import datetime

DATA_DIR = os.path.join(os.getcwd(), "data")

def _slug(s):
    if pd.isna(s): return ""
    return "".join([c for c in str(s).lower().replace("&","and").replace(".","").replace(" ","_") if c.isalnum() or c=="_"])

def load_soh_data():
    """
    Attempt to load a local historical odds dataset from /data.
    Accepts:
      - nfl_archive_10Y.json (your uploaded archive)
      - any .json/.csv in /data
    Returns a cleaned DataFrame (may be empty if nothing found).
    """
    files = []
    if os.path.isdir(DATA_DIR):
        files = glob.glob(os.path.join(DATA_DIR, "*"))
    else:
        files = []
    df = pd.DataFrame()
    tried = []
    # Prefer specific file if present
    prefer = os.path.join(DATA_DIR, "nfl_archive_10Y.json")
    if os.path.exists(prefer):
        try:
            df = pd.read_json(prefer)
            tried.append(prefer)
        except Exception:
            # try loading as lines
            try:
                df = pd.read_json(prefer, lines=True)
                tried.append(prefer + " (lines)")
            except Exception:
                df = pd.DataFrame()
    # fallback: find any .json or .csv
    if df.empty:
        for p in files:
            if p.endswith(".json") or p.endswith(".csv"):
                try:
                    if p.endswith(".json"):
                        tmp = pd.read_json(p)
                    else:
                        tmp = pd.read_csv(p)
                    df = tmp if df.empty else pd.concat([df,tmp], ignore_index=True)
                    tried.append(p)
                except Exception:
                    try:
                        tmp = pd.read_json(p, lines=True)
                        df = tmp if df.empty else pd.concat([df,tmp], ignore_index=True)
                        tried.append(p + " (lines)")
                    except Exception:
                        continue
    # If still empty, return empty df
    if df.empty:
        return pd.DataFrame()

    # Normalize columns - try to map common names
    df_cols = {c.lower(): c for c in df.columns}
    # create working df with lowercase keys
    working = df.copy()
    # create normalized columns
    def pick(cols):
        for c in cols:
            if c in df_cols:
                return df_cols[c]
        return None

    # map detection
    date_col = pick(["date","game_date","day"])
    home_col = pick(["home_team","home","team_home","homeabbr","home_short"])
    away_col = pick(["away_team","away","team_away","awayabbr","away_short"])
    home_score_col = pick(["home_score","homepoints","points_home","h_score"])
    away_score_col = pick(["away_score","awaypoints","points_away","a_score"])
    spread_col = pick(["spread","line","home_spread","closing_spread"])
    ou_col = pick(["over_under","total","ou"])

    # create standardized df
    out = pd.DataFrame()
    if date_col:
        out["date"] = pd.to_datetime(working[ df_cols[date_col].lower() ] if date_col in df.columns else working[date_col], errors="coerce")
    else:
        out["date"] = pd.NaT

    for k, cand in [("season", ["season","year"]), ("week", ["week","gweek"]), ("home_team",[home_col]), ("away_team",[away_col])]:
        if cand and cand[0] and cand[0] in df.columns:
            out[k] = working[cand[0]]
        else:
            # safe default
            out[k] = np.nan

    # scores
    out["home_score"] = working[ home_score_col ] if home_score_col in df_cols else working.get(home_score_col, np.nan)
    out["away_score"] = working[ away_score_col ] if away_score_col in df_cols else working.get(away_score_col, np.nan)
    # spreads: try to create as numeric (home_spread)
    if spread_col and spread_col in df_cols:
        out["spread"] = pd.to_numeric(working[ df_cols[spread_col].lower() ], errors="coerce")
    elif "spread" in working.columns:
        out["spread"] = pd.to_numeric(working["spread"], errors="coerce")
    else:
        out["spread"] = np.nan
    # over/under
    if ou_col and ou_col in df_cols:
        out["over_under"] = pd.to_numeric(working[ df_cols[ou_col].lower() ], errors="coerce")
    elif "over_under" in working.columns:
        out["over_under"] = pd.to_numeric(working["over_under"], errors="coerce")
    else:
        out["over_under"] = np.nan

    # normalize team strings
    out["home_team"] = out["home_team"].astype(str).str.lower().str.replace(r"[^a-z0-9 ]","",regex=True).str.strip()
    out["away_team"] = out["away_team"].astype(str).str.lower().str.replace(r"[^a-z0-9 ]","",regex=True).str.strip()

    # season/week numeric
    out["season"] = pd.to_numeric(out.get("season", None), errors="coerce").fillna(method="ffill").fillna(datetime.now().year).astype(int)
    out["week"] = pd.to_numeric(out.get("week", None), errors="coerce").fillna(0).astype(int)

    # final clean
    out = out[["date","season","week","home_team","away_team","home_score","away_score","spread","over_under"]]

    # If some values are lists or dicts because of odd JSON shapes, try to flatten common cases
    # Keep as-is otherwise.

    return out

def fill_missing_spreads(df, seed=42):
    """
    Ensure df has numeric 'spread' and 'over_under'. For rows missing this info,
    create plausible default values:
      - spread: normal centered 0, sigma 6
      - over_under: uniform 40-50
    Returns a copy.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    if "spread" not in out.columns:
        out["spread"] = np.nan
    if "over_under" not in out.columns:
        out["over_under"] = np.nan

    # ensure numeric where possible
    out["spread"] = pd.to_numeric(out["spread"], errors="coerce")
    out["over_under"] = pd.to_numeric(out["over_under"], errors="coerce")

    missing_spread = out["spread"].isna()
    missing_ou = out["over_under"].isna()

    out.loc[missing_spread, "spread"] = rng.normal(0, 6, missing_spread.sum()).round(1)
    out.loc[missing_ou, "over_under"] = rng.uniform(40, 48, missing_ou.sum()).round(1)

    return out

def merge_espn_soh(espn_df, soh_df, week=None, season=None):
    """
    Merge ESPN schedule dataframe with soh_df by season/week/home/away.
    espn_df expected to contain: ['season','week','home_team','away_team','kickoff_ts',...]
    Returns espn_df augmented with 'spread','over_under','soh_source' (soh/local/sim).
    If soh_df is empty or missing week, uses fill_missing_spreads on espn_df.
    """
    espn = espn_df.copy()
    # normalize team strings
    espn["home_team_norm"] = espn["home_team"].astype(str).str.lower().str.replace(r"[^a-z0-9 ]","",regex=True).str.strip()
    espn["away_team_norm"] = espn["away_team"].astype(str).str.lower().str.replace(r"[^a-z0-9 ]","",regex=True).str.strip()

    # filter soh to season/week if provided
    soh = soh_df.copy() if soh_df is not None and not soh_df.empty else pd.DataFrame()
    if not soh.empty:
        if season is not None:
            soh = soh[soh["season"]==int(season)]
        if week is not None:
            soh = soh[soh["week"]==int(week)]
    # normalize soh teams
    if not soh.empty:
        soh["home_team_norm"] = soh["home_team"].astype(str).str.lower().str.replace(r"[^a-z0-9 ]","",regex=True).str.strip()
        soh["away_team_norm"] = soh["away_team"].astype(str).str.lower().str.replace(r"[^a-z0-9 ]","",regex=True).str.strip()

    if soh.empty:
        # nothing from soh => return espn with simulated spreads
        res = espn.copy()
        res = fill_missing_spreads(res)
        res["soh_source"] = "simulated"
        return res

    # Try merge by home/away normalized and season/week
    try:
        merged = pd.merge(
            espn,
            soh[["season","week","home_team_norm","away_team_norm","spread","over_under","home_score","away_score","date"]],
            left_on=["season","week","home_team_norm","away_team_norm"],
            right_on=["season","week","home_team_norm","away_team_norm"],
            how="left",
            suffixes=("","_soh")
        )
    except Exception:
        # fallback: merge by home/away only
        merged = pd.merge(
            espn,
            soh[["home_team_norm","away_team_norm","spread","over_under","home_score","away_score","date"]],
            left_on=["home_team_norm","away_team_norm"],
            right_on=["home_team_norm","away_team_norm"],
            how="left"
        )

    # If still missing many spreads, fill them
    merged = fill_missing_spreads(merged)
    merged["soh_source"] = np.where(merged["spread"].notna(), "soh", "simulated")
    return merged
