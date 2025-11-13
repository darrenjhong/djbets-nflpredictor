# streamlit_app.py
# DJBets — NFL Predictor (Streamlit)
# Sidebar style: Version B (compact with icons)
#
# NOTE: replace or merge this file into your repo. This file is defensive:
# - expects data/ and public/logos/ directories
# - will not crash the app when optional data (spreads, elo files, etc.) are missing
# - trains on available historical games; falls back to a simple model if needed
#
# Git push instructions: see top-of-file comment or the assistant message.

import os
import io
import sys
import json
import time
import math
import requests
import traceback
from datetime import datetime, timedelta
from typing import Tuple, List

# Streamlit
import streamlit as st
st.set_page_config(page_title="DJBets — NFL Predictor", layout="wide")

# Data
import pandas as pd
import numpy as np

# ML - try xgboost first, fallback to sklearn LogisticRegression
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# PIL for logo validation
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# ---------------------------
# Configuration
# ---------------------------
DATA_DIR = "data"
LOGOS_DIR = os.path.join("public", "logos")  # location you said you put logos
HIST_JSON = os.path.join(DATA_DIR, "nfl_archive_10Y.json")
SCHEDULE_CSV = os.path.join(DATA_DIR, "schedule.csv")
SOH_LOCAL = os.path.join(DATA_DIR, "soh.csv")  # optional pre-scraped spreads
ODDS_API_KEY_PATH = os.path.join(DATA_DIR, "odds_api_key.txt")

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

THIS_YEAR = int(os.getenv("SEASON_YEAR", datetime.now().year))

# Model features we prefer
PREFERRED_FEATURES = ["elo_diff", "inj_diff", "temp_c", "wind_kph", "precip_prob", "spread", "over_under"]

# For UI
MAX_WEEKS = 18
DEBUG = False  # set True to show more info on UI

# ---------------------------
# Helpers
# ---------------------------

def safe_read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def try_read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return None

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

ensure_dir(DATA_DIR)

# ---------------------------
# Logo utilities
# ---------------------------

def normalize_team_key(s: str) -> str:
    """Normalize a team string (from ESPN or archive) to a filename-friendly key."""
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    # common conversions
    s = s.replace(" ", "_").replace(".", "").replace("'", "").replace("-", "_")
    # remove characters
    s = "".join(ch for ch in s if ch.isalnum() or ch == "_")
    return s

def get_logo_candidates(team_name: str) -> List[str]:
    """
    Return list of candidate filenames for a team, in order.
    Examples: bears -> ['bears.png', 'chi_bears.png', 'chicago_bears.png', 'CHI.png']
    """
    base = normalize_team_key(team_name)
    if not base:
        return []
    candidates = []
    # basic
    candidates.append(f"{base}.png")
    candidates.append(f"{base}.jpg")
    # with nfl_ prefix
    candidates.append(f"nfl_{base}.png")
    # short uppercase versions (common abbrs might be inside team_name already)
    abbr = "".join([p[0] for p in team_name.split() if p])
    if len(abbr) >= 2:
        candidates.append(f"{abbr.upper()}.png")
    # variations
    variants = [
        base.replace("_", ""),
        base.replace("_", "-"),
        base.replace("_", ""),
        base + "_logo",
        base + "-logo",
    ]
    for v in variants:
        candidates.append(f"{v}.png")
        candidates.append(f"{v}.jpg")
    # fallback to team_name as-is
    candidates.append(team_name + ".png")
    return list(dict.fromkeys(candidates))  # unique keep order

def find_logo_path(team_name: str) -> str:
    """Return a usable path to a logo or None."""
    # If team_name is already a path, try it
    if not team_name:
        return None
    if os.path.isabs(team_name) and os.path.exists(team_name):
        return team_name
    # Check in LOGOS_DIR
    for cand in get_logo_candidates(team_name):
        p = os.path.join(LOGOS_DIR, cand)
        if os.path.exists(p):
            return p
    # also try directly under public/
    for cand in get_logo_candidates(team_name):
        p = os.path.join("public", cand)
        if os.path.exists(p):
            return p
    return None

def safe_st_image(path_or_bytes, width=None):
    """Display image but don't crash on invalid image file."""
    try:
        st.image(path_or_bytes, width=width)
    except Exception as e:
        if DEBUG:
            st.text(f"[logo error] {e}")

# ---------------------------
# Data loaders
# ---------------------------

@st.cache_data(ttl=60*60)
def load_local_history() -> pd.DataFrame:
    """Load historical games from local JSON archive if exists."""
    if os.path.exists(HIST_JSON):
        try:
            with open(HIST_JSON, "r", encoding="utf-8") as f:
                obj = json.load(f)
            # Many archives are list-of-dicts
            df = pd.json_normalize(obj)
            # Normalize common columns
            # Attempt to map common field names to our canonical ones
            colmap = {}
            if "home_team" not in df.columns and "home" in df.columns:
                colmap["home"] = "home_team"
            if "away_team" not in df.columns and "away" in df.columns:
                colmap["away"] = "away_team"
            df.rename(columns=colmap, inplace=True)
            return df
        except Exception as e:
            if DEBUG:
                st.text(f"Failed to load local history: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data(ttl=30*60)
def load_local_schedule() -> pd.DataFrame:
    """Load schedule.csv if present. If not present, return empty df."""
    df = try_read_csv(SCHEDULE_CSV)
    if df is None:
        return pd.DataFrame()
    # Normalize common columns
    df.columns = [c.strip() for c in df.columns]
    # expected: season,week,home,away,kickoff,date,time,spread,over_under,home_score,away_score
    return df

def safe_request(url, params=None, timeout=8):
    """HTTP wrapper to make ESPN requests safe."""
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception as e:
        if DEBUG:
            st.text(f"HTTP request failed: {e}")
        return None

@st.cache_data(ttl=15*60)
def fetch_scoreboard_week(season:int, week:int) -> pd.DataFrame:
    """
    Fetch ESPN scoreboard for a particular season/week.
    Returns DataFrame with canonical columns if possible.
    """
    params = {"season": season, "seasontype": 2, "week": week}
    r = safe_request(ESPN_SCOREBOARD_URL, params=params)
    if not r:
        return pd.DataFrame()
    try:
        js = r.json()
        games = []
        for evt in js.get("events", []):
            # parse teams
            competitions = evt.get("competitions", [])
            if not competitions:
                continue
            comp = competitions[0]
            status = comp.get("status", {}).get("type", {}).get("description", "")
            kickoff = comp.get("date", None)
            # teams
            teams = comp.get("competitors", [])
            if len(teams) < 2:
                continue
            # determine home/away
            home = None; away = None
            for t in teams:
                if t.get("homeAway") == "home":
                    home = t
                else:
                    away = t
            # fallback if not labeled
            if home is None or away is None:
                home = teams[0]; away = teams[1]
            # extract common fields
            rec = {
                "season": season,
                "week": week,
                "home_team": home.get("team", {}).get("displayName") or home.get("team", {}).get("shortDisplayName"),
                "away_team": away.get("team", {}).get("displayName") or away.get("team", {}).get("shortDisplayName"),
                "home_score": None,
                "away_score": None,
                "status": status,
                "kickoff_utc": kickoff,
                "espn_game_id": evt.get("id"),
            }
            # scores if present
            try:
                if "score" in home:
                    rec["home_score"] = int(home.get("score", 0))
                if "score" in away:
                    rec["away_score"] = int(away.get("score", 0))
            except Exception:
                pass
            games.append(rec)
        df = pd.DataFrame(games)
        return df
    except Exception:
        if DEBUG:
            st.text("Failed to parse ESPN JSON")
            traceback.print_exc()
        return pd.DataFrame()

@st.cache_data(ttl=60*60)
def fetch_scoreboard_season(season:int) -> pd.DataFrame:
    """Fetch scoreboard for whole season by iterating weeks 1..MAX_WEEKS"""
    rows = []
    for w in range(1, MAX_WEEKS+1):
        dfw = fetch_scoreboard_week(season, w)
        if dfw is None or dfw.empty:
            continue
        rows.append(dfw)
        # be considerate
        time.sleep(0.2)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()

# ---------------------------
# SOH / spread helpers (local or remote)
# ---------------------------

@st.cache_data(ttl=60*60)
def load_soh_local() -> pd.DataFrame:
    """Load a local SOH file if present; otherwise empty DataFrame."""
    df = try_read_csv(SOH_LOCAL)
    if df is None:
        return pd.DataFrame()
    return df

def merge_spreads(espn_df: pd.DataFrame, soh_df: pd.DataFrame) -> pd.DataFrame:
    """Merge espn schedule with available spreads data (SOH); forgiving merges."""
    if espn_df is None or espn_df.empty:
        return espn_df
    merged = espn_df.copy()
    if soh_df is None or soh_df.empty:
        merged["spread"] = np.nan
        merged["over_under"] = np.nan
        return merged
    # try to merge on (season, week, home_team/away_team) - be forgiving about names
    soh = soh_df.copy()
    # normalize names in both
    def n(x): return normalize_team_key(x)
    soh["home_norm"] = soh.get("home_team", soh.get("home", soh.get("home_team_name", ""))).astype(str).apply(n)
    soh["away_norm"] = soh.get("away_team", soh.get("away", soh.get("away_team_name", ""))).astype(str).apply(n)
    merged["home_norm"] = merged["home_team"].astype(str).apply(n)
    merged["away_norm"] = merged["away_team"].astype(str).apply(n)
    # attempt left join
    try:
        merged = merged.merge(soh[["season","week","home_norm","away_norm","spread","over_under"]],
                              left_on=["season","week","home_norm","away_norm"],
                              right_on=["season","week","home_norm","away_norm"],
                              how="left")
    except Exception:
        # fallback: set NaNs
        merged["spread"] = np.nan
        merged["over_under"] = np.nan
    # drop helper columns
    merged.drop(columns=[c for c in ["home_norm","away_norm"] if c in merged.columns], inplace=True, errors="ignore")
    return merged

# ---------------------------
# Elo & features
# ---------------------------

def ensure_feature_cols(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Ensure DataFrame has the required feature columns; fill missing with 0/NaN defaults."""
    for f in features:
        if f not in df.columns:
            if f in ("spread","over_under","temp_c","wind_kph","precip_prob","inj_diff","elo_diff"):
                df[f] = np.nan
            else:
                df[f] = 0
    return df

@st.cache_data(ttl=60*60)
def compute_simple_elo(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a simple season-agnostic Elo for teams based on historical scores if present.
    This is lightweight fallback if no external Elo file is available.
    """
    if hist is None or hist.empty:
        return pd.DataFrame()
    # expect columns home_team, away_team, home_score, away_score
    h = hist.copy()
    if not all(c in h.columns for c in ["home_team","away_team","home_score","away_score"]):
        return pd.DataFrame()
    # initialize elo dict
    elo = {}
    K = 20
    base = 1500
    def get_e(t): return elo.get(t, base)
    for _, row in h.iterrows():
        ht = row["home_team"]; at = row["away_team"]
        try:
            hs = float(row["home_score"]); ascore = float(row["away_score"])
        except Exception:
            continue
        eh = get_e(ht); ea = get_e(at)
        # expected
        exp_h = 1 / (1 + 10 ** ((ea - eh) / 400))
        # actual
        if hs > ascore:
            s_h = 1.0
        elif hs < ascore:
            s_h = 0.0
        else:
            s_h = 0.5
        # update
        elo[ht] = eh + K * (s_h - exp_h)
        elo[at] = ea + K * ((1 - s_h) - (1 - exp_h))
    # convert to df
    rows = [{"team": t, "elo": v} for t,v in elo.items()]
    return pd.DataFrame(rows)

# ---------------------------
# Model training / loading
# ---------------------------

@st.cache_resource(ttl=60*60*6)
def train_model_from_history(hist: pd.DataFrame) -> Tuple[object, List[str]]:
    """
    Train a model from historical data. Returns (model, features_used).
    It will try to use XGBoost (if available) or fall back to sklearn LogisticRegression.
    This routine is defensive about missing columns and will simulate minimal training if necessary.
    """
    # Required label columns
    if hist is None:
        hist = pd.DataFrame()
    h = hist.copy()
    # Ensure canonical fields
    # We want a binary label: home_win (1/0) if scores exist
    if "home_score" in h.columns and "away_score" in h.columns:
        h = h.dropna(subset=["home_score","away_score"])
        h["home_win"] = (pd.to_numeric(h["home_score"], errors="coerce") > pd.to_numeric(h["away_score"], errors="coerce")).astype(int)
    else:
        # no labels available -> insufficient data; produce a fallback model (weak)
        # We'll simulate a tiny dataset using Elo differences if available else random
        h = pd.DataFrame({
            "elo_diff": np.random.normal(0, 100, size=200),
            "inj_diff": np.random.normal(0,1,size=200),
            "temp_c": np.random.normal(15,5,size=200),
            "wind_kph": np.random.normal(8,3,size=200),
            "precip_prob": np.random.uniform(0,1,size=200),
        })
        h["home_win"] = (h["elo_diff"] > 0).astype(int)

    # Determine features available
    # We try PREFERRED_FEATURES; if not available use subset
    features = [f for f in PREFERRED_FEATURES if f in h.columns]
    if not features:
        # fallback to numeric columns excluding labels
        features = [c for c in h.select_dtypes(include=[np.number]).columns if c not in ("home_score","away_score","home_win")]
    if not features:
        # last resort
        features = ["elo_diff"]
        h["elo_diff"] = h.get("elo_diff", np.random.normal(0,100,size=len(h)))

    X = h[features].fillna(0)
    y = h["home_win"].astype(int)

    # Train/test split just to create a model
    try:
        if XGB_AVAILABLE:
            # Use XGBoost classifier
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=1)
            model.fit(X, y)
            return model, features
        elif SKLEARN_AVAILABLE:
            model = LogisticRegression(max_iter=200)
            model.fit(X, y)
            return model, features
        else:
            # No ML libs available — return a tiny callable fallback
            class DummyModel:
                def __init__(self):
                    pass
                def predict_proba(self, X):
                    # synthetic: convert elo_diff to probability if present else 0.5
                    if "elo_diff" in (X.columns if hasattr(X,"columns") else []):
                        ed = np.array(X["elo_diff"], dtype=float)
                        p = 1 / (1 + np.exp(-ed/50.0))
                        return np.vstack([1-p, p]).T
                    return np.vstack([np.ones(len(X))*0.5, np.ones(len(X))*0.5]).T
            return DummyModel(), features
    except Exception:
        # If training XGBoost fails for some reason, fallback to logistic regression or dummy
        if SKLEARN_AVAILABLE:
            try:
                model = LogisticRegression(max_iter=200)
                model.fit(X, y)
                return model, features
            except Exception:
                return None, features
        return None, features

# ---------------------------
# Utility functions for UI / predictions
# ---------------------------

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def pretty_dt(utc_str):
    if pd.isna(utc_str):
        return ""
    try:
        dt = pd.to_datetime(utc_str)
        # show in local-ish format (no timezone conversion)
        return dt.strftime("%a %b %d %H:%M")
    except Exception:
        return str(utc_str)

def format_team_display(team_name):
    # Shorten long names if required
    if not team_name:
        return ""
    parts = team_name.split()
    if len(parts) > 3:
        return " ".join(parts[:2])
    return team_name

# ---------------------------
# ROI & record computations (not cached with DataFrames)
# ---------------------------

def compute_model_record_simple(history: pd.DataFrame, model, features: List[str]):
    """
    Compute simple model record: how many completed games was model correct.
    Avoid caching DataFrame/Model combo to prevent hashing issues.
    """
    if history is None or history.empty or model is None:
        return 0, 0, 0.0
    df = history.copy()
    # require final scores
    if not all(c in df.columns for c in ("home_score","away_score")):
        return 0, 0, 0.0
    df = df.dropna(subset=["home_score","away_score"])
    if df.empty:
        return 0,0,0.0
    # construct feature matrix for those games
    ensure_feature_cols(df, features)
    X = df[features].fillna(0)
    try:
        probs = model.predict_proba(X)[:,1]
    except Exception:
        return 0,0,0.0
    preds = (probs >= 0.5).astype(int)
    actual = (pd.to_numeric(df["home_score"]) > pd.to_numeric(df["away_score"])).astype(int)
    correct = int((preds == actual).sum())
    total = int(len(df))
    pct = correct / total if total else 0.0
    return correct, total-correct, pct

# ---------------------------
# Main page
# ---------------------------

st.title("🏈 DJBets — NFL Predictor")
st.markdown("Local/ESPN schedule + local historical archive. Logos from `public/logos/`")

# Sidebar B (compact with icons)
with st.sidebar:
    st.markdown("## 🏈 DJBets NFL Predictor")
    # Season selector
    season = st.selectbox("🗂 Season", options=[THIS_YEAR, THIS_YEAR-1, THIS_YEAR-2], index=0, key="season")
    # Week selector at top per request
    # Build weeks from schedule if available later; for now show 1..18
    week = st.selectbox("🗓 Week", options=list(range(1, MAX_WEEKS+1)), index=0, key="week")
    st.markdown("---")
    st.markdown("⚙️ Model Settings")
    market_weight = st.slider("🎚 Market weight (0 = ignore market)", 0.0, 1.0, 0.5, step=0.05, help="Blend between model and market probabilities.")
    bet_threshold = st.slider("🎯 Bet threshold (pp)", 0.0, 10.0, 3.0, step=0.5, help="Minimum edge (percentage points) to recommend a bet.")
    st.markdown("---")
    st.markdown("📈 Performance")
    # We'll fill these later after we compute model record
    perf_col1, perf_col2 = st.columns(2)
    perf_col1.metric("✔️ Correct", "—")
    perf_col2.metric("❌ Incorrect", "—")
    perf_col3, perf_col4 = st.columns(2)
    perf_col3.metric("📉 ROI", "—")
    perf_col4.metric("🧾 Bets", "—")

# ---------------------------
# Load data
# ---------------------------

# 1) History
hist_df = load_local_history()
if hist_df is None:
    hist_df = pd.DataFrame()
if not hist_df.empty:
    st.success(f"✅ Loaded historical data with {len(hist_df)} rows from local archive.")
else:
    st.warning("⚠️ No local historical data found. Model will use simulated fallback.")

# 2) Schedule
local_sched = load_local_schedule()
espn_sched = fetch_scoreboard_season(season) if local_sched.empty else pd.DataFrame()
if not local_sched.empty:
    st.success(f"✅ Loaded schedule from {SCHEDULE_CSV}")
    sched_df = local_sched.copy()
else:
    if not espn_sched.empty:
        st.success("✅ Loaded schedule from ESPN scoreboard")
        sched_df = espn_sched.copy()
    else:
        sched_df = pd.DataFrame()
        st.warning("⚠️ No schedule file found locally and ESPN failed. Upload schedule.csv to data/ for offline mode.")


# 3) SOH spreads local
soh_df = load_soh_local()
if soh_df is None or soh_df.empty:
    if DEBUG:
        st.info("SOH local not available.")
else:
    st.success("✅ Loaded local SOH spreads (optional)")

# 4) Merge spreads (SOH preferred; ESPN-only allowed)
sched_df = merge_spreads(sched_df, soh_df)

# Normalize columns
if "week" not in sched_df.columns:
    # some ESPN JSON may not include week; try to infer from espn events by mapping id->week unknown
    sched_df["week"] = sched_df.get("week", week)

# Ensure season column
sched_df["season"] = sched_df.get("season", season)

# Filter to selected season/week
if not sched_df.empty:
    week_sched = sched_df[sched_df["week"] == week].copy()
else:
    week_sched = pd.DataFrame()

# Guarantee feature columns exist on schedule for display
week_sched = ensure_feature_cols(week_sched, ["spread","over_under","temp_c","wind_kph","precip_prob","inj_diff","elo_diff"])

# ---------------------------
# Train model (auto train on first launch)
# ---------------------------
model, model_features = train_model_from_history(hist_df)

if model is None:
    st.error("Model training failed. Using lightweight fallback.")
else:
    st.write(f"Model trained — using features: {', '.join(model_features)}")

# ---------------------------
# Sidebar performance metrics (fill)
# ---------------------------
correct, incorrect, pct = compute_model_record_simple(hist_df, model, model_features)
# Update sidebar metrics (note Streamlit does not allow direct reassign to prior metric placeholders; re-render instead)
with st.sidebar:
    st.markdown("📈 Performance")
    col1, col2 = st.columns(2)
    col1.metric("✔️ Correct", f"{correct}")
    col2.metric("❌ Incorrect", f"{incorrect}")
    col3, col4 = st.columns(2)
    col3.metric("📉 ROI", f"—")  # ROI calc can be added if you want bet-level historical data
    col4.metric("🧾 Bets", f"—")

# ---------------------------
# Page contents
# ---------------------------
tabs = st.tabs(["Games", "Model Tracker", "Top Bets"])
tab_games, tab_tracker, tab_bets = tabs

with tab_games:
    st.header(f"Season {season} — Week {week}")
    if week_sched is None or week_sched.empty:
        st.info("No games found for this week.")
    else:
        # Pre-open game cards (user requested)
        for idx, row in week_sched.reset_index(drop=True).iterrows():
            home = row.get("home_team", "Home")
            away = row.get("away_team", "Away")
            kickoff = pretty_dt(row.get("kickoff_utc") or row.get("kickoff_ts") or row.get("kickoff"))
            status = row.get("status", "")
            spread = row.get("spread", np.nan)
            ou = row.get("over_under", np.nan)
            home_logo_path = find_logo_path(home) or find_logo_path(row.get("home_abbr",""))
            away_logo_path = find_logo_path(away) or find_logo_path(row.get("away_abbr",""))

            # card layout
            st.markdown("---")
            cols = st.columns([1,3,3,1])
            # away logo
            with cols[0]:
                if away_logo_path:
                    safe_st_image(away_logo_path, width=60)
                else:
                    st.write("")  # keep spacing
            # away text
            with cols[1]:
                st.subheader(format_team_display(away))
                st.caption("Away")
            # vs and kickoff
            with cols[2]:
                st.markdown(f"### @{format_team_display(home)}")
                st.caption(f"{kickoff} — {status}")
                st.write(f"Spread: {spread if not pd.isna(spread) else 'N/A'} | O/U: {ou if not pd.isna(ou) else 'N/A'}")
            # home logo
            with cols[3]:
                if home_logo_path:
                    safe_st_image(home_logo_path, width=60)

            # Build features for this single game (DataFrame row)
            feat_row = {}
            for f in model_features:
                feat_row[f] = row.get(f, np.nan)
            feat_df = pd.DataFrame([feat_row]).fillna(0)

            # Predict
            prob = None
            try:
                prob = model.predict_proba(feat_df[model_features].fillna(0))[:,1][0]
            except Exception:
                try:
                    # if sklearn logistic
                    prob = model.predict_proba(feat_df.fillna(0))[:,1][0]
                except Exception:
                    prob = 0.5
            prob = float(prob)
            # Market probability from spread if available (rough conversion)
            market_prob = None
            if not pd.isna(spread):
                try:
                    s = float(spread)
                    # simple conversion: home_favored -> lower home win prob
                    market_prob = 1 / (1 + 10 ** (-s / 15.0))  # rough heuristic
                except Exception:
                    market_prob = np.nan
            # blended
            if market_prob is None or pd.isna(market_prob):
                blended = prob
            else:
                blended = prob * (1 - market_weight) + market_prob * market_weight
            edge_pp = (blended - (market_prob if market_prob is not None and not pd.isna(market_prob) else prob)) * 100 if market_prob is not None else 0.0

            # Decision
            recommended = False
            if not pd.isna(edge_pp) and abs(edge_pp) >= bet_threshold:
                recommended = True

            # Display results with bars and details
            col_a, col_b, col_c = st.columns([2,6,3])
            with col_a:
                st.progress(min(max(prob, 0.0), 1.0), text=f"Home Win Prob: {prob*100:.1f}%")
                st.caption("Model")
            with col_b:
                if market_prob is not None and not pd.isna(market_prob):
                    st.caption(f"Market Prob: {market_prob*100:.1f}%")
                else:
                    st.caption("Market Prob: N/A")
                st.caption(f"Blended: {blended*100:.1f}% | Edge: {edge_pp:.2f} pp")
                if recommended:
                    st.success("Recommendation: ✅ Bet")
                else:
                    st.info("Recommendation: 🚫 No Bet")
            with col_c:
                # show if game finished and whether model was correct
                if status and "final" in status.lower():
                    hs = safe_float(row.get("home_score", np.nan))
                    ascore = safe_float(row.get("away_score", np.nan))
                    if not math.isnan(hs) and not math.isnan(ascore):
                        model_pred = "Home" if prob >= 0.5 else "Away"
                        actual_winner = "Home" if hs > ascore else ("Away" if ascore > hs else "Push")
                        correct_flag = (model_pred == actual_winner)
                        if correct_flag:
                            st.success(f"Model was correct — Final {int(hs)}-{int(ascore)}")
                        else:
                            st.error(f"Model was wrong — Final {int(hs)}-{int(ascore)}")
                    else:
                        st.write("Final — scores unavailable")
                else:
                    st.write("Not started / In progress")

with tab_tracker:
    st.header("Model Tracker")
    st.markdown("Historical model results and performance")
    # Show simple table grouping by week if possible
    if hist_df is None or hist_df.empty:
        st.info("No historical data available to show tracker.")
    else:
        # We'll compute simple weekly metrics if home_score/away_score exist
        if all(c in hist_df.columns for c in ("home_score","away_score","week")):
            df = hist_df.copy()
            df["home_win"] = pd.to_numeric(df["home_score"], errors="coerce") > pd.to_numeric(df["away_score"], errors="coerce")
            summary = df.groupby("week").apply(lambda g: pd.Series({
                "games": len(g),
                "home_win_pct": g["home_win"].mean() * 100
            })).reset_index()
            st.dataframe(summary)
        else:
            st.info("Historical file doesn't contain week/home_score/away_score for a tracker view.")

with tab_bets:
    st.header("Top Model Bets of the Week")
    # Find recommended bets for current week
    if week_sched is None or week_sched.empty:
        st.info("No games for this week.")
    else:
        recs = []
        for _, r in week_sched.iterrows():
            # build features row
            feat_row = {f: r.get(f, np.nan) for f in model_features}
            feat_df = pd.DataFrame([feat_row]).fillna(0)
            try:
                prob = float(model.predict_proba(feat_df[model_features].fillna(0))[:,1][0])
            except Exception:
                try:
                    prob = float(model.predict_proba(feat_df.fillna(0))[:,1][0])
                except Exception:
                    prob = 0.5
            market_prob = np.nan
            try:
                if not pd.isna(r.get("spread")):
                    market_prob = 1 / (1 + 10 ** (-float(r.get("spread")) / 15.0))
            except Exception:
                market_prob = np.nan
            blended = prob if np.isnan(market_prob) else prob*(1-market_weight)+market_prob*market_weight
            edge_pp = (blended - (market_prob if not np.isnan(market_prob) else prob))*100
            if not np.isnan(edge_pp) and abs(edge_pp) >= bet_threshold:
                recs.append({"away": r.get("away_team"), "home": r.get("home_team"), "edge": edge_pp, "prob": prob, "spread": r.get("spread")})
        if not recs:
            st.info("No top bets this week (edge below threshold).")
        else:
            for r in sorted(recs, key=lambda x: -abs(x["edge"]))[:10]:
                cols = st.columns([1,4,1,1])
                cols[0].write(f"{r['away']} @ {r['home']}")
                cols[1].write(f"Edge: {r['edge']:.2f} pp — Model Prob {r['prob']*100:.1f}% — Spread {r['spread']}")
                cols[2].write("")
                cols[3].write("")

# ---------------------------
# Footer / debug
# ---------------------------
st.markdown("---")
st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — Data dir: {DATA_DIR}")

if DEBUG:
    st.write("DEBUG MODE ON")
    st.write("Schedule sample:", sched_df.head().to_dict(orient="records") if not sched_df.empty else {})
    st.write("History sample:", hist_df.head().to_dict(orient="records") if not hist_df.empty else {})

# End of streamlit_app.py