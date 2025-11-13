<<<<<<< HEAD
﻿# streamlit_app.py
# DJBets NFL Predictor — updated version (logos in public/logos)
#
# Git commands to push this update (run locally where repo is initialized):
# ----------------------------------------------------------------------
# git add streamlit_app.py
# git commit -m "Update streamlit_app.py: ESPN fetching, stable model training, logos in public/logos"
# git push origin main
#
# If you need to also push all changes:
# git add .
# git commit -m "Full app update"
# git push origin main
# ----------------------------------------------------------------------

import os
import time
import json
import math
=======
﻿# app.py — DJBets NFL Predictor (Option 2: ESPN live schedule; no local schedule.csv)
# Full replacement file — paste/overwrite your existing streamlit_app.py

import os
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from io import BytesIO

>>>>>>> 5ce4016bfe9027cefdf333e500dc10a940d1a50e
import requests
import traceback
from datetime import datetime, timezone
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np
<<<<<<< HEAD

import streamlit as st

# ML
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except Exception:
    xgb = None
    XGBClassifier = None

# -------------------------
# Configuration & Constants
# -------------------------
THIS_YEAR = int(os.getenv("SEASON_YEAR", "2025"))
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LOGOS_DIR = os.path.join(os.path.dirname(__file__), "public", "logos")  # path B as requested
LOCAL_ARCHIVE = os.path.join(DATA_DIR, "nfl_archive_10Y.json")  # your uploaded 2011-2021 file
SCHEDULE_CSV = os.path.join(DATA_DIR, "schedule.csv")  # optional local schedule
ODDS_API_KEY_FILE = os.path.join(DATA_DIR, "odds_api_key.txt")  # optional local file you placed
ODDS_API_KEY = None
if os.path.exists(ODDS_API_KEY_FILE):
    try:
        ODDS_API_KEY = open(ODDS_API_KEY_FILE).read().strip()
    except Exception:
        ODDS_API_KEY = None
# Also allow Streamlit secrets:
try:
    ODDS_API_KEY = ODDS_API_KEY or st.secrets.get("ODDS_API_KEY", None)
except Exception:
    # in case st.secrets not configured
    pass

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

# stable model features order we expect for predictions
MODEL_FEATURES = ["elo_diff", "inj_diff", "temp_c", "wind_kph", "precip_prob"]

# UI defaults
DEFAULT_MARKET_WEIGHT = 0.5
DEFAULT_BET_THRESHOLD_PP = 3.0  # percent points of edge (pp)
=======
import streamlit as st

# ML
try:
    import xgboost as xgb
except Exception:
    xgb = None
from sklearn.model_selection import train_test_split

# ---------------------------
# Config
# ---------------------------

DATA_DIR = Path("data")
PUBLIC_DIR = Path("public")  # logos stored here as instructed
HIST_FILE = DATA_DIR / "nfl_archive_10Y.json"
ODDS_API_KEY_ENV = "ODDS_API_KEY"
ODDS_API_PATH = DATA_DIR / "odds_api_key.txt"

THIS_YEAR = int(os.getenv("SEASON_YEAR", datetime.now().year))
>>>>>>> 5ce4016bfe9027cefdf333e500dc10a940d1a50e
MAX_WEEKS = 18
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
ESPN_EVENTS_URL = "https://site.api.espn.com/apis/v2/sports/football/leagues/nfl/events"  # fallback

# Model features expected
MODEL_FEATURES = ["elo_diff", "inj_diff", "temp_c", "wind_kph", "precip_prob"]
# If spread/OU are available we will include them, but keep model stable to above features to avoid mismatch.

# ---------------------------
# Helpers
# ---------------------------

<<<<<<< HEAD
# Streamlit page config
st.set_page_config(page_title="DJBets NFL Predictor", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Helper functions
# -------------------------


def safe_request(url: str, params: dict = None, headers: dict = None, timeout: int = 8, retries: int = 3) -> Optional[requests.Response]:
    """Simple request wrapper with retries/backoff. Returns Response on success, else None."""
    params = params or {}
    headers = headers or {}
    backoff = 1.0
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            # don't spam UI with tracebacks here; return None after retries
            if attempt < retries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            else:
                return None
=======
def read_odds_api_key():
    key = os.environ.get(ODDS_API_KEY_ENV)
    if key:
        return key.strip()
    if ODDS_API_PATH.exists():
        return ODDS_API_PATH.read_text().strip()
>>>>>>> 5ce4016bfe9027cefdf333e500dc10a940d1a50e
    return None


<<<<<<< HEAD
@st.cache_data(ttl=60 * 60)  # cache for 1 hour
def fetch_scoreboard_week(season: int, week: int) -> pd.DataFrame:
    """Fetch scoreboard for specific week via ESPN. Returns normalized DataFrame or empty df."""
    params = {"season": season, "seasontype": 2, "week": week}
    resp = safe_request(ESPN_SCOREBOARD_URL, params=params, retries=3)
    if resp is None:
        return pd.DataFrame()
    try:
        data = resp.json()
        events = data.get("events", [])
        rows = []
        for e in events:
            # parse basic fields; handle missing gracefully
            try:
                competitions = e.get("competitions", [])
                if not competitions:
                    continue
                comp = competitions[0]
                status = comp.get("status", {}).get("type", {}).get("description", "")
                kickoff_ts = None
                if comp.get("date"):
                    kickoff_ts = comp.get("date")
                # teams
                competitors = comp.get("competitors", [])
                if len(competitors) < 2:
                    continue
                home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
                away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
                # scores
                home_score = home.get("score", None)
                away_score = away.get("score", None)
                # team names / short names
                home_team = home.get("team", {}).get("name")
                away_team = away.get("team", {}).get("name")
                home_abbr = home.get("team", {}).get("abbreviation")
                away_abbr = away.get("team", {}).get("abbreviation")
                rows.append({
                    "season": season,
                    "week": week,
                    "status": status,
                    "kickoff_ts": kickoff_ts,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_abbr": home_abbr,
                    "away_abbr": away_abbr,
                    "home_score": None if home_score == "" else (int(home_score) if home_score is not None and str(home_score).isdigit() else None),
                    "away_score": None if away_score == "" else (int(away_score) if away_score is not None and str(away_score).isdigit() else None)
                })
            except Exception:
                continue
        df = pd.DataFrame(rows)
        # normalize kickoff timestamp
        if not df.empty and "kickoff_ts" in df.columns:
            df["kickoff_ts"] = pd.to_datetime(df["kickoff_ts"], errors="coerce").dt.tz_convert(None)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60 * 60)
def fetch_scoreboard_season(season: int, max_weeks: int = MAX_WEEKS) -> pd.DataFrame:
    """Fetch scoreboard for a whole season by iterating weeks; gracefully returns whatever it can."""
    all_rows = []
    for w in range(1, max_weeks + 1):
        dfw = fetch_scoreboard_week(season, w)
        if dfw is None or dfw.empty:
            # continue to next week; ESPN may not expose certain weeks early
            continue
        all_rows.append(dfw)
    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


@st.cache_data(ttl=60 * 60)
def load_historical_archive(path: str = LOCAL_ARCHIVE) -> pd.DataFrame:
    """Load local historical JSON archive if present. If not present, return empty DataFrame."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Expect list of dicts or DataFrame serializable
        df = pd.DataFrame(data)
        return df
    except Exception:
        return pd.DataFrame()


def normalize_team_key(name: str) -> str:
    """Normalize team name to file-safe lowercase key used for logo filenames (e.g., 'New England Patriots' -> 'patriots')."""
    if not name:
        return None
    # if user supplies abbr already (e.g., 'NE'), try lowercasing
    key = str(name).strip().lower()
    # remove punctuation and common words
    for rem in ["fc", "team", "the", " ", ".", ",", "—", "-", "_"]:
        key = key.replace(rem, "")
    # heuristics: keep last word as team nickname if full name provided
    parts = str(name).strip().lower().split()
    if len(parts) >= 2:
        # use last word (e.g., "new england patriots" -> "patriots")
        key = parts[-1]
    # final cleanup
    key = "".join(ch for ch in key if ch.isalnum())
    return key


def get_logo_path(team_name_or_abbr: Optional[str]) -> Optional[str]:
    """Return a path under LOGOS_DIR if file exists, else None."""
    if not team_name_or_abbr:
        return None
    # try multiple candidate filenames
    candidates = []
    raw = str(team_name_or_abbr).strip()
    candidates.append(raw.lower() + ".png")
    candidates.append(raw.lower() + ".jpg")
    candidates.append(raw.lower() + ".svg")
    # normalized nickname
    nick = normalize_team_key(raw)
    if nick:
        candidates.append(nick + ".png")
        candidates.append(nick + ".jpg")
    # try uppercase abbr as well
    candidates.append(raw.upper() + ".png")
    for c in candidates:
        path = os.path.join(LOGOS_DIR, c)
        if os.path.exists(path):
            return path
    return None


def safe_get_odds_for_game(home_abbr: str, away_abbr: str, date_iso: str) -> dict:
    """
    Use OddsAPI to fetch current odds for a single game; return dict with spread and totals if available.
    This function uses only free endpoints (current odds). Historical odds are not requested here.
    """
    if not ODDS_API_KEY:
        return {}
    # OddsAPI expects US sports slugs; for NFL we use 'americanfootball_nfl'
    base = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "spreads,totals",
        "dateFormat": "iso"
    }
    try:
        resp = requests.get(base, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        # data is a list of events; match by date/team abbreviations if possible
        best = {}
        for ev in data:
            # "home_team" / "away_team" field names may vary; try to match using abbreviations
            ev_time = ev.get("commence_time")  # ISO
            # compare dates ignoring seconds
            try:
                if date_iso and ev_time:
                    if abs((pd.to_datetime(ev_time) - pd.to_datetime(date_iso)).total_seconds()) > 6 * 3600:
                        # not the same game time within 6 hours
                        continue
            except Exception:
                pass
            # find site with spreads/totals
            for site in ev.get("bookmakers", []):
                for m in site.get("markets", []):
                    if m.get("key") == "spreads":
                        outcomes = m.get("outcomes", [])
                        # pick the market that looks right
                        for o in outcomes:
                            name = o.get("name", "")
                            price = o.get("price")
                            point = o.get("point")
                            # identify home vs away by name matching abbr if possible
                            # best-effort
                            if home_abbr and (home_abbr.lower() in name.lower() or home_abbr.upper() in name.upper()):
                                best["spread_home"] = point
                            if away_abbr and (away_abbr.lower() in name.lower() or away_abbr.upper() in name.upper()):
                                best["spread_away"] = point
                    if m.get("key") == "totals":
                        for o in m.get("outcomes", []):
                            # usually two outcomes with 'over' and 'under' and point = total
                            if o.get("name", "").lower() in ("over", "under"):
                                best["over_under"] = o.get("point")
                if best:
                    break
            if best:
                break
        return best
    except Exception:
        return {}


# -------------------------
# Modeling helpers
# -------------------------


@st.cache_resource
def create_fallback_model(seed: int = 42) -> XGBClassifier:
    """Create a simple fallback model (very low complexity) if training fails or insufficient historical data."""
    if XGBClassifier is None:
        raise RuntimeError("xgboost is required for the model")
    model = XGBClassifier(n_estimators=10, max_depth=3, use_label_encoder=False, eval_metric="logloss", random_state=seed)
    # it will be trained later or used as-is (predict_proba may fail if not trained; we guard that)
    return model


def ensure_feature_columns(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Ensure DataFrame has required feature columns; fill missing with zeros / sensible defaults."""
    for f in features:
        if f not in df.columns:
            if f == "temp_c":
                df[f] = 0.0
            elif f == "wind_kph":
                df[f] = 0.0
            elif f == "precip_prob":
                df[f] = 0.0
            else:
                df[f] = 0.0
            st.warning(f"⚠️ Added missing feature column: {f}")
    # reorder to features order when requested outside
    return df


@st.cache_data(ttl=60 * 60)
def train_model(historical: pd.DataFrame) -> Tuple[XGBClassifier, List[str]]:
    """
    Train an XGBClassifier on historical data if possible.
    Returns (model, feature_list_used)
    Note: historical is expected to include columns: home_score, away_score, elo_home, elo_away, inj_home, inj_away, temp_c, wind_kph, precip_prob
    We'll compute label: home_win (1 if home_score > away_score)
    """
    # fallback features we aim for:
    features = MODEL_FEATURES.copy()
    # If insufficient data, return fallback model
    if XGBClassifier is None:
        st.error("XGBoost not available in environment. Model disabled.")
        return None, features
    # ensure columns
    df = historical.copy()
    if df.empty:
        st.info("⚠️ Not enough historical data — using simulated training set.")
        # create small simulated dataset
        rng = np.random.default_rng(123)
        X_sim = pd.DataFrame({
            "elo_diff": rng.normal(0, 50, 200),
            "inj_diff": rng.normal(0, 1, 200),
            "temp_c": rng.normal(12, 6, 200),
            "wind_kph": rng.normal(10, 4, 200),
            "precip_prob": rng.uniform(0, 1, 200),
        })
        y_sim = (X_sim["elo_diff"] + rng.normal(0, 60, 200) > 0).astype(int)
        model = XGBClassifier(n_estimators=80, max_depth=3, use_label_encoder=False, eval_metric="logloss", random_state=42)
        model.fit(X_sim[features], y_sim)
        return model, features

    # prepare dataset
    # compute home_win label when scores present
    if "home_score" in df.columns and "away_score" in df.columns:
        df = df.dropna(subset=["home_score", "away_score"])
        df["home_win"] = (df["home_score"].astype(float) > df["away_score"].astype(float)).astype(int)
    else:
        # cannot compute label; fallback
        st.warning("⚠️ Historical data missing 'home_score'/'away_score' — using simulated training fallback.")
        return train_model(pd.DataFrame())

    # compute ELO difference (if elo columns exist)
    if "elo_home" in df.columns and "elo_away" in df.columns:
        df["elo_diff"] = df["elo_home"].astype(float) - df["elo_away"].astype(float)
    else:
        # fallback simple average
        df["elo_diff"] = 0.0

    # inj_diff
    if "inj_home" in df.columns and "inj_away" in df.columns:
        df["inj_diff"] = df["inj_home"].astype(float) - df["inj_away"].astype(float)
    else:
        df["inj_diff"] = 0.0

    # weather defaults
    df["temp_c"] = pd.to_numeric(df.get("temp_c", 0.0), errors="coerce").fillna(0.0)
    df["wind_kph"] = pd.to_numeric(df.get("wind_kph", 0.0), errors="coerce").fillna(0.0)
    df["precip_prob"] = pd.to_numeric(df.get("precip_prob", 0.0), errors="coerce").fillna(0.0)

    ensure_feature_columns(df, features)

    X = df[features].astype(float)
    y = df["home_win"].astype(int)

    # if not enough positive/negative samples, fallback
    if len(df) < 50 or len(np.unique(y)) < 2:
        st.warning("⚠️ Not enough valid labeled historical data — using simulated training set.")
        return train_model(pd.DataFrame())

    model = XGBClassifier(n_estimators=200, max_depth=4, use_label_encoder=False, eval_metric="logloss", random_state=42)
    try:
        model.fit(X, y)
        return model, features
    except Exception as e:
        st.error("Model training failed; using fallback simulated model. Error: " + str(e))
        traceback.print_exc()
        return train_model(pd.DataFrame())


# -------------------------
# Compute helpers
# -------------------------


def compute_model_record(historical: pd.DataFrame, model: XGBClassifier, features: List[str]) -> Tuple[int, int, float]:
    """Compute model record (correct, incorrect, pct) for completed games in historical that have scores."""
    if historical is None or historical.empty or model is None:
        return 0, 0, 0.0
    hist = historical.copy()
    if "home_score" not in hist.columns or "away_score" not in hist.columns:
        return 0, 0, 0.0
    hist = hist.dropna(subset=["home_score", "away_score"])
    if hist.empty:
        return 0, 0, 0.0
    # ensure features present
    hist = ensure_feature_columns(hist, features)
    try:
        probs = model.predict_proba(hist[features])[:, 1]
    except Exception:
        return 0, 0, 0.0
    preds = (probs >= 0.5).astype(int)
    truths = (hist["home_score"].astype(float) > hist["away_score"].astype(float)).astype(int)
    correct = int((preds == truths).sum())
    incorrect = int((preds != truths).sum())
    pct = correct / (correct + incorrect) * 100 if (correct + incorrect) > 0 else 0.0
    return correct, incorrect, pct


def compute_roi(schedule_df: pd.DataFrame, market_weight: float = 0.5, bet_threshold_pp: float = 3.0) -> Tuple[float, int, float]:
    """
    Simple ROI calculator: For each recommended bet (edge_pp >= threshold), we assume 1 unit wager,
    payout 0.91 for favorite (approx -110), compute P/L.
    market_weight blends model with market probability; but this function expects schedule_df already has fields:
      - home_win_prob_model
      - market_prob (0-1)
      - edge_pp (percentage points)
      - recommended (bool)
      - home_team, away_team
      - final result fields home_score/away_score when present
    """
    if schedule_df is None or schedule_df.empty:
        return 0.0, 0, 0.0
    s = schedule_df.copy()
    bets = s[s.get("recommended", False)]
    pnl = 0.0
    count = 0
    for _, r in bets.iterrows():
        count += 1
        # determine bet side, assume betting on team with positive edge
        side = r.get("bet_on")
        amt = 1.0
        # simplified payout - assume -110 style (0.91)
        payout = 1.91
        # outcome
        won = None
        if pd.notna(r.get("home_score")) and pd.notna(r.get("away_score")):
            if side == "home":
                won = r["home_score"] > r["away_score"]
            elif side == "away":
                won = r["away_score"] > r["home_score"]
        else:
            # future game -> can't compute P/L
            won = None
        if won is True:
            pnl += (payout - 1.0) * amt
        elif won is False:
            pnl -= amt
        else:
            # no effect for upcoming games
            pnl += 0.0
    roi = (pnl / max(1, count)) * 100 if count > 0 else 0.0
    return pnl, count, roi


# -------------------------
# UI + Main App Flow
# -------------------------

st.title("🏈 DJBets — NFL Predictor")
st.caption("Local/ESPN schedule + local historical archive. Logos from public/logos/")

# Sidebar
with st.sidebar:
    st.markdown("## 🏈 DJBets NFL Predictor")
    # top: week + season
    season = st.selectbox("Season", [THIS_YEAR, THIS_YEAR - 1, THIS_YEAR - 2], index=0, key="season_select")
    # we'll populate week dropdown dynamically below after fetching schedule
    st.markdown("---")
    st.markdown("### Model Controls")
    market_weight = st.slider("Market weight (blend model vs market)", 0.0, 1.0, DEFAULT_MARKET_WEIGHT, 0.05, key="market_weight")
    bet_threshold_pp = st.slider("Bet threshold (edge, percentage points)", 0.0, 10.0, DEFAULT_BET_THRESHOLD_PP, 0.5, key="bet_threshold_pp")
    st.markdown("**Info:** Market weight blends the model probability with the market (0 = model only, 1 = market only).")
    st.markdown("Bet threshold controls how large the edge (in percent points) must be before recommending a bet.")
    st.markdown("---")
    # model tracker / ROI area will fill later; placeholder
    st.markdown("### Model Performance")
    perf_placeholder = st.empty()

# Load historical archive (local)
hist = load_historical_archive()
if hist is None or hist.empty:
    st.info("⚠️ No local historical file found or it's empty. The app will simulate training data as fallback.")
else:
    st.success(f"✅ Loaded historical data with {len(hist)} rows from local archive.")

#
# Build schedule: try local CSV first, else use ESPN
#
sched_df = pd.DataFrame()
if os.path.exists(SCHEDULE_CSV):
    try:
        sched_df = pd.read_csv(SCHEDULE_CSV)
        # normalize expected columns if present
        if "kickoff" in sched_df.columns:
            sched_df["kickoff_ts"] = pd.to_datetime(sched_df["kickoff"], errors="coerce")
        st.info(f"✅ Loaded schedule from {SCHEDULE_CSV}")
    except Exception as e:
        st.warning("Could not read local schedule.csv; will attempt ESPN. Error: " + str(e))
# If no local schedule or empty, attempt to fetch from ESPN
if sched_df.empty:
    st.info("⚙️ No local schedule file found — fetching schedule from ESPN...")
    try:
        sched_df = fetch_scoreboard_season(season)
        if sched_df is None or sched_df.empty:
            st.warning("⚠️ No games loaded from ESPN for this season.")
            sched_df = pd.DataFrame()
        else:
            st.success(f"✅ Loaded schedule from ESPN for season {season} ({len(sched_df)} games).")
    except Exception as e:
        st.error("ESPN fetch failed: " + str(e))
        sched_df = pd.DataFrame()

# Build list of weeks present for dropdown
available_weeks = sorted(list(sched_df["week"].dropna().unique())) if not sched_df.empty else list(range(1, MAX_WEEKS + 1))
if not available_weeks:
    available_weeks = list(range(1, MAX_WEEKS + 1))

# place week selector near top of sidebar (we created sidebar above, now set week)
with st.sidebar:
    week = st.selectbox("📅 Week", available_weeks, index=0, key="week_select")

# Merge in odds (current/future only) - best-effort using oddsapi; do not attempt historical odds via API
if not sched_df.empty:
    sched_df = sched_df.copy()
    # add columns for odds placeholders
    sched_df["spread"] = np.nan
    sched_df["over_under"] = np.nan
    sched_df["odds_api"] = None
    for idx, row in sched_df.iterrows():
        try:
            home_abbr = row.get("home_abbr") or row.get("home_team")
            away_abbr = row.get("away_abbr") or row.get("away_team")
            kickoff = row.get("kickoff_ts")
            kickoff_iso = pd.to_datetime(kickoff).isoformat() if pd.notna(kickoff) else None
            odds = safe_get_odds_for_game(home_abbr, away_abbr, kickoff_iso)
            if odds:
                sched_df.at[idx, "odds_api"] = odds
                if "spread_home" in odds:
                    sched_df.at[idx, "spread"] = odds.get("spread_home")
                if "over_under" in odds:
                    sched_df.at[idx, "over_under"] = odds.get("over_under")
        except Exception:
            continue

# Prepare schedule for UI: filter by selected season/week
if sched_df.empty:
    st.warning("No schedule available to display. Try switching season or ensure ESPN is reachable.")
    week_sched = pd.DataFrame()
else:
    week_sched = sched_df[sched_df["week"] == week].copy()
    if week_sched.empty:
        st.info(f"✅ Loaded schedule for {season}. Showing Week {week}.\n\n⚠️ No games found for this week.")
    else:
        st.success(f"✅ Loaded schedule for {season}. Showing Week {week} with {len(week_sched)} game(s).")

# Compute/mass-produce features for the week and run predictions
# Build a dataset row per game with required features (elo_diff, inj_diff, temp_c, wind_kph, precip_prob)
def build_features_for_week(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    w = df.copy()
    # naive ELO: use historical average or fallback
    # If historical has elo for same matchup, prefer that. Otherwise use heuristic 1500 for both.
    avg_elo = 1500.0
    # attempt to use historical if present
    if hist is not None and not hist.empty and "elo_home" in hist.columns and "elo_away" in hist.columns:
        # create mapping most recent elo by team
        try:
            elo_latest = {}
            for t in pd.unique(list(hist.get("home_team", [])) + list(hist.get("away_team", []))):
                sub = hist[(hist.get("home_team") == t) | (hist.get("away_team") == t)]
                if not sub.empty and "elo_home" in sub.columns:
                    # pick last available entry
                    val = sub.tail(1)
                    # pick elo from whichever side
                    if val["home_team"].iloc[0] == t:
                        elo_latest[t] = float(val["elo_home"].iloc[0])
                    else:
                        elo_latest[t] = float(val["elo_away"].iloc[0])
        except Exception:
            elo_latest = {}
    else:
        elo_latest = {}

    # populate features
    rows = []
    for _, r in w.iterrows():
        home = r.get("home_team") or r.get("home_abbr")
        away = r.get("away_team") or r.get("away_abbr")
        home_elo = elo_latest.get(home, avg_elo)
        away_elo = elo_latest.get(away, avg_elo)
        elo_diff = float(home_elo) - float(away_elo)
        inj_diff = 0.0  # stub: no injury provider integrated yet
        temp_c = 0.0
        wind_kph = 0.0
        precip_prob = 0.0
        rows.append({
            "season": r.get("season", season),
            "week": r.get("week", week),
            "home_team": home,
            "away_team": away,
            "home_abbr": r.get("home_abbr"),
            "away_abbr": r.get("away_abbr"),
            "kickoff_ts": r.get("kickoff_ts"),
            "elo_diff": elo_diff,
            "inj_diff": inj_diff,
            "temp_c": temp_c,
            "wind_kph": wind_kph,
            "precip_prob": precip_prob,
            "spread": r.get("spread"),
            "over_under": r.get("over_under"),
            "home_score": r.get("home_score"),
            "away_score": r.get("away_score"),
        })
    feats = pd.DataFrame(rows)
    return feats


# Build features for selected week
feats = build_features_for_week(week_sched)

# Train model on historical data (auto-train on first launch)
model = None
model_features = MODEL_FEATURES.copy()
try:
    model, model_features = train_model(hist)
except Exception as e:
    st.error("Model training failed: " + str(e))
    model = create_fallback_model()

# Ensure features present and aligned for prediction
if feats is None or feats.empty:
    st.info("No games to predict for this week.")
else:
    feats = ensure_feature_columns(feats, model_features)
    X_for_pred = feats[model_features].astype(float)

    # predict probabilities safely
    try:
        probs = model.predict_proba(X_for_pred)[:, 1]
    except Exception as e:
        st.error("⚠️ Model prediction failed: " + str(e))
        # fallback uniform probability
        probs = np.full(len(X_for_pred), 0.5)

    feats["home_win_prob_model"] = probs
    # market probability: derive from spread if present (very naive conversion)
    def spread_to_prob(spread):
        try:
            if spread is None or np.isnan(spread):
                return np.nan
            # small logistic mapping: spread in points to probability
            return 1.0 / (1.0 + math.exp(-0.15 * float(-spread)))  # negative spread means home favored by x
        except Exception:
            return np.nan

    feats["market_prob"] = feats["spread"].apply(spread_to_prob)
    # blended probability
    feats["blended_prob"] = feats["home_win_prob_model"] * (1 - market_weight) + feats["market_prob"].fillna(feats["home_win_prob_model"]) * market_weight
    # edge in percentage points
    feats["edge_pp"] = (feats["home_win_prob_model"] - feats["market_prob"]) * 100
    # recommendation rule
    def recommend(row):
        # if market_prob missing, require model confidence alone (bigger threshold)
        if pd.isna(row.get("market_prob")):
            if abs(row["home_win_prob_model"] - 0.5) * 100 >= bet_threshold_pp * 2:
                return True
            return False
        if pd.isna(row.get("edge_pp")):
            return False
        return abs(row["edge_pp"]) >= bet_threshold_pp
    feats["recommended"] = feats.apply(recommend, axis=1)
    # bet_on side: home if model > market
    def bet_side(row):
        if not row.get("recommended"):
            return None
        m = row.get("market_prob")
        mod = row.get("home_win_prob_model")
        if pd.isna(m):
            return "home" if mod > 0.5 else "away"
        return "home" if (mod - m) > 0 else "away"
    feats["bet_on"] = feats.apply(bet_side, axis=1)

    # compute predicted scores: naive decomposition from ELO diff and expected total
    # we will produce predicted total points and then split by home/away using elo_diff proportion
    feats["pred_total"] = 43.0 + (feats["precip_prob"] * -6.0)  # simple heuristic
    feats["pred_home_points"] = (feats["pred_total"] / 2) + (feats["elo_diff"] / 60.0)
    feats["pred_away_points"] = (feats["pred_total"] / 2) - (feats["elo_diff"] / 60.0)
    feats["pred_home_points"] = feats["pred_home_points"].round(1)
    feats["pred_away_points"] = feats["pred_away_points"].round(1)

# Compute model record and ROI using historical
correct, incorrect, pct = compute_model_record(hist, model, model_features)
pnl, bets_made, roi = compute_roi(hist)

with perf_placeholder:
    st.markdown(f"**Record:** {correct} correct / {incorrect} incorrect ({pct:.1f}%)")
    st.markdown(f"**ROI:** {roi:.2f}% | **PnL:** {pnl:.2f} units | **Bets tracked:** {bets_made}")

# Main display: show games in expanded cards
st.header(f"Season {season} — Week {week}")

if feats is None or feats.empty:
    st.info("No games found for this week.")
else:
    # present games: one expandable card per game, but open by default
    for i, row in feats.iterrows():
        # Title row
        home = row.get("home_team") or row.get("home_abbr") or "Home"
        away = row.get("away_team") or row.get("away_abbr") or "Away"
        kickoff = row.get("kickoff_ts")
        kickoff_str = pd.to_datetime(kickoff).strftime("%Y-%m-%d %H:%M") if pd.notna(kickoff) else "TBD"
        card_title = f"{away}  @  {home} — {kickoff_str}"
        with st.expander(card_title, expanded=True):
            cols = st.columns([1, 6, 1])
            # Left: away logo & name
            with cols[0]:
                away_logo = get_logo_path(row.get("away_abbr") or row.get("away_team"))
                if away_logo:
                    try:
                        st.image(away_logo, width=72)
                    except Exception:
                        st.write(row.get("away_team"))
                else:
                    st.write(row.get("away_team"))
            # Center: info
            with cols[1]:
                st.markdown(f"**Predicted (score)**: {row.get('pred_away_points', '—')} - {row.get('pred_home_points', '—')} (Total: {row.get('pred_total', '—')})")
                prob = row.get("home_win_prob_model", 0.5)
                try:
                    st.progress(float(max(0.0, min(1.0, prob))), text=f"Home Win Probability: {prob*100:.1f}%")
                except Exception:
                    st.write(f"Home Win Probability: {prob}")
                market_prob = row.get("market_prob")
                market_text = f"{market_prob*100:.1f}%" if pd.notna(market_prob) else "n/a"
                st.markdown(f"**Market**: {market_text} | **Spread**: {row.get('spread', 'n/a')} | **O/U**: {row.get('over_under', 'n/a')}")
                edge = row.get("edge_pp")
                edge_text = f"{edge:+.1f} pp" if pd.notna(edge) else "n/a"
                st.markdown(f"**Edge:** {edge_text}  |  **Blended:** {row.get('blended_prob', 0.0)*100:.1f}%")
                # Recommendation box
                if row.get("recommended"):
                    side = row.get("bet_on")
                    rec_text = f"✅ Recommend bet: {side.upper()} (edge {edge_text})"
                    st.success(rec_text)
                else:
                    st.info("🚫 No Bet (edge below threshold or insufficient data)")
                # show final score if present
                hs = row.get("home_score")
                as_ = row.get("away_score")
                if pd.notna(hs) and pd.notna(as_):
                    # use corrected logic for correctness
                    model_pred_home = row.get("home_win_prob_model", 0.5) >= 0.5
                    actual_home = float(hs) > float(as_)
                    was_correct = bool(model_pred_home == actual_home)
                    if was_correct:
                        st.success(f"Model was correct — final score {int(as_)} @ {int(hs)}")
                    else:
                        st.error(f"Model was incorrect — final score {int(as_)} @ {int(hs)}")
                else:
                    st.write("Game not started / final score not available yet.")
            # Right: home logo
            with cols[2]:
                home_logo = get_logo_path(row.get("home_abbr") or row.get("home_team"))
                if home_logo:
                    try:
                        st.image(home_logo, width=72)
                    except Exception:
                        st.write(row.get("home_team"))
                else:
                    st.write(row.get("home_team"))

# Optional: Top model bets summary
st.markdown("---")
st.subheader("🏆 Top Model Bets of the Week")
if feats is None or feats.empty:
    st.info("No top bets to show.")
else:
    top_bets = feats[feats["recommended"]].sort_values(by="edge_pp", ascending=False).head(10)
    if top_bets.empty:
        st.info("No recommended bets this week.")
    else:
        for _, r in top_bets.iterrows():
            away_logo = get_logo_path(r.get("away_abbr") or r.get("away_team"))
            home_logo = get_logo_path(r.get("home_abbr") or r.get("home_team"))
            cols = st.columns([1, 6, 1])
            with cols[0]:
                if away_logo:
                    try:
                        st.image(away_logo, width=48)
                    except Exception:
                        st.write(r.get("away_team"))
                else:
                    st.write(r.get("away_team"))
            with cols[1]:
                st.markdown(f"**{r.get('away_team')} @ {r.get('home_team')}** — Edge {r.get('edge_pp', 0.0):+.1f} pp — Bet: {r.get('bet_on')}")
                st.caption(f"Model prob: {r.get('home_win_prob_model')*100:.1f}%  |  Market: { (r.get('market_prob')*100):.1f}% " if pd.notna(r.get('market_prob')) else "")
            with cols[2]:
                if home_logo:
                    try:
                        st.image(home_logo, width=48)
                    except Exception:
                        st.write(r.get("home_team"))
                else:
                    st.write(r.get("home_team"))

# Footer / Notes
st.markdown("---")
st.markdown("**Notes & Next steps:**")
st.markdown("- The app uses ESPN for schedule/scores. OddsAPI (if configured) is used only for current/future odds; historical odds require a paid plan and are not requested.")
st.markdown("- Injuries/weather providers are placeholders; integrate a paid/free provider or scraping to fill `inj_diff`, `temp_c`, `wind_kph`, `precip_prob`.")
st.markdown("- If logos do not appear, ensure files are named as `<abbrev>.png` or `<nickname>.png` inside `public/logos/` (e.g., `bears.png`, `gb.png`, `patriots.png`).")
st.markdown("- Model training uses local historical archive if available; otherwise the app will simulate a training set for demo purposes.")
=======
def safe_request(url, params=None, timeout=10):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception as e:
        st.experimental_log(f"HTTP request failed: {e}")
        return None

@st.cache_data(ttl=60*60)  # cache for 1 hour
def fetch_scoreboard_week(season, week):
    """Return DataFrame of games for a given season & week from ESPN scoreboard."""
    params = {"season": season, "week": week, "seasontype": 2}
    r = safe_request(ESPN_SCOREBOARD_URL, params=params)
    if not r:
        return pd.DataFrame()
    try:
        data = r.json()
    except Exception:
        return pd.DataFrame()
    games = []
    events = data.get("events", [])
    for ev in events:
        # event contains competitions array
        comps = ev.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]
        status = comp.get("status", {}).get("type", {}).get("name")
        competitions = comp
        # extract teams
        teams = comp.get("competitors", [])
        # find home/away
        home = None
        away = None
        for t in teams:
            if t.get("homeAway") == "home":
                home = t
            else:
                away = t
        # kickoff
        kickoff = comp.get("date")
        # odds
        odds = comp.get("odds", [])
        spread = None
        total = None
        if odds:
            o = odds[0]
            details = o.get("details") or ""
            # sometimes details include "Bears -3"
            # ESPN also provides "spread" in temp fields; try to parse
            home_team = home.get("team", {}).get("abbreviation") if home else None
            # try get lines
            # structure differs across responses, so guard heavily
            lines = o.get("details")
            # better: check comp['odds']->book
            if "spread" in o:
                try:
                    spread = float(o.get("spread"))
                except Exception:
                    spread = None
            else:
                # parse details
                try:
                    # details like "CHI -3"
                    parts = details.split()
                    if len(parts) >= 2:
                        maybe = parts[-1]
                        if maybe.startswith("+") or maybe.startswith("-"):
                            spread = float(maybe)
                except Exception:
                    spread = None
            # total
            try:
                total = float(o.get("total")) if o.get("total") else None
            except Exception:
                total = None

        games.append({
            "season": season,
            "week": week,
            "game_id": ev.get("id") or comp.get("id"),
            "status": status,
            "kickoff_utc": kickoff,
            "home_team": home.get("team", {}).get("displayName") if home else None,
            "home_abbrev": home.get("team", {}).get("abbreviation") if home else None,
            "away_team": away.get("team", {}).get("displayName") if away else None,
            "away_abbrev": away.get("team", {}).get("abbreviation") if away else None,
            "home_score": int(next((sc.get("value") for sc in comp.get("score", []) if False), 0)) if False else None,
            "away_score": None,
            "spread": spread,
            "over_under": total,
            "raw": comp
        })
    return pd.DataFrame(games)

@st.cache_data(ttl=60*60)
def fetch_scoreboard_season(season):
    """Fetch all weeks for a season by querying scoreboard week-by-week."""
    all_games = []
    for w in range(1, MAX_WEEKS + 1):
        df = fetch_scoreboard_week(season, w)
        if df is None or df.empty:
            # continue — ESPN might not have future weeks yet
            continue
        all_games.append(df)
    if not all_games:
        return pd.DataFrame()
    return pd.concat(all_games, ignore_index=True)

def try_load_history():
    if HIST_FILE.exists():
        try:
            raw = json.loads(HIST_FILE.read_text())
            df = pd.json_normalize(raw)
            # Try to standardize expected columns
            # user historical may contain home_score/away_score and week/season/home/away
            # We'll attempt to map common fields
            # Normalize colnames lower
            df.columns = [c.lower() for c in df.columns]
            # rename common variants
            rename_map = {}
            for c in df.columns:
                if c in ("home_team", "hometeam", "home"):
                    rename_map[c] = "home_team"
                if c in ("away_team", "awayteam", "away"):
                    rename_map[c] = "away_team"
                if c in ("home_score", "homesc", "homescore"):
                    rename_map[c] = "home_score"
                if c in ("away_score", "awaysc", "awayscore"):
                    rename_map[c] = "away_score"
                if c == "week":
                    rename_map[c] = "week"
                if c == "season":
                    rename_map[c] = "season"
            if rename_map:
                df = df.rename(columns=rename_map)
            return df
        except Exception as e:
            st.warning("Failed to parse historical JSON: " + str(e))
            return pd.DataFrame()
    else:
        return pd.DataFrame()

# ---------------------------
# Quick Elo impl (fallback)
# ---------------------------

def compute_simple_elo(hist_df, k=20, start_elo=1500):
    """
    Compute per-team Elo across historical games.
    hist_df expected to have: season, week, home_team, away_team, home_score, away_score, date (optional)
    Returns dataframe elo_history with columns ['team','date','elo'] latest Elo per team.
    """
    teams = {}
    # Ensure safe ordering by date if available
    df = hist_df.copy()
    if "date" in df.columns:
        df = df.sort_values("date")
    for _, row in df.iterrows():
        ht = row.get("home_team")
        at = row.get("away_team")
        hs = row.get("home_score")
        as_ = row.get("away_score")
        if pd.isna(hs) or pd.isna(as_):
            continue
        teams.setdefault(ht, start_elo)
        teams.setdefault(at, start_elo)
        R_home = teams[ht]
        R_away = teams[at]
        # expected
        expected_home = 1 / (1 + 10 ** ((R_away - R_home) / 400))
        # actual
        if hs > as_:
            actual_home = 1.0
        elif hs < as_:
            actual_home = 0.0
        else:
            actual_home = 0.5
        diff = k * (actual_home - expected_home)
        teams[ht] += diff
        teams[at] -= diff
    # return dict
    return teams

# ---------------------------
# Model training & prediction
# ---------------------------

@st.cache_data(ttl=60*60*2)
def train_model(history_df):
    """
    Train an XGBoost classifier using features from history_df.
    If insufficient labeled data, return a fallback dummy model.
    """
    # If xgboost not installed, fallback to simple rule model
    class DummyModel:
        def predict_proba(self, X):
            # return 0.5 for everything
            arr = np.ones((len(X), 2)) * 0.5
            return arr

    if xgb is None:
        st.warning("xgboost not installed. Using fallback model.")
        return DummyModel(), MODEL_FEATURES

    df = history_df.copy()
    # ensure necessary columns
    # Create features: elo_diff, inj_diff, temp_c, wind_kph, precip_prob from historical if exist
    # We'll simulate features when missing.
    required = ["home_team", "away_team", "home_score", "away_score"]
    if not all(c in df.columns for c in required):
        st.warning("Not enough labeled historical games — using fallback simulated training set.")
        # create simulated dataset
        rng = np.random.default_rng(123)
        X = pd.DataFrame({
            "elo_diff": rng.normal(0, 50, 500),
            "inj_diff": rng.normal(0, 2, 500),
            "temp_c": rng.normal(15, 8, 500),
            "wind_kph": rng.normal(10, 5, 500),
            "precip_prob": rng.uniform(0, 0.5, 500),
        })
        y = (X["elo_diff"] + rng.normal(0, 30, 500)) > 0
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=100)
        model.fit(X, y)
        return model, MODEL_FEATURES

    # derive features from historical
    # compute Elo per team (simple)
    elo_map = compute_simple_elo(df)
    # build dataset rows
    rows = []
    for _, r in df.iterrows():
        ht = r["home_team"]
        at = r["away_team"]
        hs = r.get("home_score")
        as_ = r.get("away_score")
        if pd.isna(hs) or pd.isna(as_):
            continue
        elo_home = elo_map.get(ht, 1500)
        elo_away = elo_map.get(at, 1500)
        elo_diff = elo_home - elo_away
        inj_diff = 0.0
        temp_c = 12.0
        wind_kph = 8.0
        precip_prob = 0.0
        # label: home win
        label = 1 if hs > as_ else 0
        rows.append({
            "elo_diff": elo_diff,
            "inj_diff": inj_diff,
            "temp_c": temp_c,
            "wind_kph": wind_kph,
            "precip_prob": precip_prob,
            "label": label
        })
    if not rows:
        # fallback simulated
        rng = np.random.default_rng(42)
        X = pd.DataFrame({
            "elo_diff": rng.normal(0, 50, 500),
            "inj_diff": rng.normal(0, 2, 500),
            "temp_c": rng.normal(15, 8, 500),
            "wind_kph": rng.normal(10, 5, 500),
            "precip_prob": rng.uniform(0, 0.5, 500),
        })
        y = (X["elo_diff"] + rng.normal(0, 30, 500)) > 0
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=100)
        model.fit(X, y)
        return model, MODEL_FEATURES

    df_train = pd.DataFrame(rows)
    X = df_train[MODEL_FEATURES]
    y = df_train["label"].astype(int)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=100)
    try:
        model.fit(X, y)
    except Exception as e:
        st.warning("Model training error, using fallback: " + str(e))
        # fallback
        class Fallback:
            def predict_proba(self, X):
                arr = np.ones((len(X), 2)) * 0.5
                return arr
        return Fallback(), MODEL_FEATURES
    return model, MODEL_FEATURES

# ---------------------------
# Utility: logos
# ---------------------------

def get_logo_path(team_name_or_abbrev):
    """
    Try multiple file name variants in PUBLIC_DIR, return a path-like str or None.
    Accepts abbreviations like "CHI" or names like "Bears" or "Chicago Bears" or "bears".
    """
    if not team_name_or_abbrev:
        return None
    name = str(team_name_or_abbrev).strip()
    candidates = []
    # raw name lower
    base = name.lower().replace(" ", "_").replace(".", "").replace("'", "")
    candidates.append(f"{base}.png")
    candidates.append(f"{base}.jpg")
    # abbreviation uppercase
    candidates.append(f"{name.upper()}.png")
    candidates.append(f"{name.upper()}.jpg")
    # just alpha chars
    base2 = "".join([c for c in name.lower() if c.isalpha()])
    candidates.append(f"{base2}.png")
    # also try nfl_{team}.png
    candidates.append(f"nfl_{base}.png")
    # try team short names (e.g., bears)
    try:
        for c in candidates:
            p = PUBLIC_DIR / c
            if p.exists():
                return str(p)
    except Exception:
        pass
    return None

# ---------------------------
# Transform schedule -> features for model & UI
# ---------------------------

def enrich_schedule(df, elo_map=None):
    """
    Add computed fields to schedule:
      - kickoff_dt (datetime)
      - elo_home/away, elo_diff
      - inj_diff (0 placeholder)
      - temp_c, wind_kph, precip_prob placeholders
    """
    out = df.copy()
    out["kickoff_dt"] = pd.to_datetime(out["kickoff_utc"], utc=True, errors="coerce")
    # If no elo_map provided, set neutral elo
    if elo_map is None:
        elo_map = {}
    def get_elo(team):
        if team is None:
            return 1500.0
        return float(elo_map.get(team, 1500.0))
    out["elo_home"] = out["home_team"].apply(get_elo)
    out["elo_away"] = out["away_team"].apply(get_elo)
    out["elo_diff"] = out["elo_home"] - out["elo_away"]
    out["inj_diff"] = 0.0
    out["temp_c"] = 15.0
    out["wind_kph"] = 8.0
    out["precip_prob"] = 0.0
    # ensure numeric columns
    out["spread"] = pd.to_numeric(out.get("spread", np.nan), errors="coerce")
    out["over_under"] = pd.to_numeric(out.get("over_under", np.nan), errors="coerce")
    return out

# ---------------------------
# Utility: model helpers
# ---------------------------

def compute_model_record(history_df, model, features):
    """
    Compute simple model historical record (correct/incorrect) on games with known outcomes.
    Returns (correct, incorrect, pct)
    """
    if history_df is None or history_df.empty:
        return 0, 0, 0.0
    # Filter labeled games
    df = history_df.copy()
    if not all(c in df.columns for c in ("home_score", "away_score")):
        return 0, 0, 0.0
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r.get("home_score")) or pd.isna(r.get("away_score")):
            continue
        # derive features similar to train
        elo_diff = 0.0
        inj_diff = 0.0
        temp_c = 15.0
        wind_kph = 8.0
        precip_prob = 0.0
        rows.append({
            "elo_diff": elo_diff,
            "inj_diff": inj_diff,
            "temp_c": temp_c,
            "wind_kph": wind_kph,
            "precip_prob": precip_prob,
            "home_score": r.get("home_score"),
            "away_score": r.get("away_score"),
        })
    if not rows:
        return 0, 0, 0.0
    df2 = pd.DataFrame(rows)
    X = df2[features]
    try:
        probs = model.predict_proba(X)[:,1]
    except Exception:
        probs = np.full(len(X), 0.5)
    preds = probs >= 0.5
    actuals = df2["home_score"] > df2["away_score"]
    correct = int((preds == actuals).sum())
    incorrect = int((preds != actuals).sum())
    pct = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0.0
    return correct, incorrect, pct

# ---------------------------
# Odds helper (live)
# ---------------------------

def fetch_odds_for_game(home_abbrev, away_abbrev, kickoff_dt):
    """
    If ODDS_API_KEY is present and we want live odds we can call the odds API.
    This function is careful: will not query paid endpoints for history.
    For this version, we attempt a live odds lookup (v3 odds) when ODDS_API_KEY present.
    If not available, return (None, None)
    """
    if not ODDS_API_KEY:
        return None, None
    # We will use a simple free endpoint pattern — user's key may be limited so guard
    # Example OddsAPI GET: https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds?regions=us&markets=spreads,totals&dateFormat=iso&apiKey=YOUR_KEY
    try:
        url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
        params = {
            "regions": "us",
            "markets": "spreads,totals",
            "dateFormat": "iso",
            "apiKey": ODDS_API_KEY
        }
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        # Try to find matching game by teams and kickoff near time
        for g in data:
            # g has 'home_team' and 'away_team' or 'teams'
            teams = g.get("teams", [])
            if not teams:
                continue
            # normalize
            teams_low = [t.lower() for t in teams]
            if (home_abbrev and home_abbrev.lower() in teams_low) or (away_abbrev and away_abbrev.lower() in teams_low):
                # extract spreads and totals
                best_spread = None
                best_total = None
                for book in g.get("bookmakers", []):
                    for m in book.get("markets", []):
                        if m.get("key") == "spreads":
                            # outcomes with point
                            for o in m.get("outcomes", []):
                                # find home outcome
                                if o.get("name").lower() == home_abbrev.lower() or o.get("name").lower() in teams_low:
                                    # this will be the spread for that team - store if found
                                    best_spread = o.get("point")
                        if m.get("key") == "totals":
                            for o in m.get("outcomes", []):
                                best_total = o.get("total")
                return best_spread, best_total
    except Exception:
        return None, None
    return None, None

# ---------------------------
# UI layout & main
# ---------------------------

st.set_page_config(page_title="DJBets NFL Predictor", layout="wide", initial_sidebar_state="expanded")
st.title("🏈 DJBets NFL Predictor (ESPN live schedule — no local schedule.csv)")
st.markdown("Auto-trains on launch — uses local history (if present) and ESPN live schedule. Logos should be in `/public`.")

# Sidebar — week selector at top as requested
with st.sidebar:
    st.markdown("## 📅 Season & Week")
    seasons = sorted(list({THIS_YEAR, THIS_YEAR-1, THIS_YEAR-2}), reverse=True)
    season = st.selectbox("Season", seasons, index=0, key="season")
    # weeks gather from fetch; show full range
    week_sel = st.selectbox("Week", list(range(1, MAX_WEEKS+1)), index=0, key="week")
    st.markdown("---")
    st.markdown("## ⚙️ Model Controls")
    market_weight = st.slider("Market Weight (blending model vs market)", 0.0, 1.0, 0.3, 0.05, help="0 = model only, 1 = market only")
    bet_threshold = st.slider("Bet Threshold (pp)", 0.0, 20.0, 3.0, 0.5, help="Edge PP required to place a bet")
    st.markdown("---")
    st.markdown("## 📊 Model Record")
    st.write("Loading...")
    # later will populate placeholders
    st.markdown("---")
    st.markdown("## ℹ️ Notes")
    st.caption("This app fetches live schedules from ESPN. Odds are fetched from TheOddsAPI if key is provided (live only). Historical labels used for training come from data/nfl_archive_10Y.json when present.")

# load history
hist_df = try_load_history()
if hist_df.empty:
    st.warning("No historical file found in data/. App will auto-simulate training data. Place nfl_archive_10Y.json in /data to improve training.")
else:
    st.success(f"✅ Loaded historical data with {len(hist_df)} rows from {HIST_FILE}")

# compute elo map from history if possible
elo_map = {}
if not hist_df.empty:
    try:
        elo_map = compute_simple_elo(hist_df)
    except Exception as e:
        st.warning("Elo computation error: " + str(e))
        elo_map = {}

# Train model
with st.spinner("Training model..."):
    model, model_features = train_model(hist_df)
st.success("Model ready.")

# compute model record
correct, incorrect, pct = compute_model_record(hist_df, model, model_features)
# update sidebar values
with st.sidebar:
    st.metric("Correct", correct)
    st.metric("Incorrect", incorrect)
    st.metric("Accuracy", f"{pct*100:.1f}%")

# fetch schedule for selected season
with st.spinner("Fetching schedule from ESPN..."):
    sched_season = fetch_scoreboard_season(season)
if sched_season is None or sched_season.empty:
    st.error(f"⚠️ No games loaded from ESPN for season {season}.")
    sched = pd.DataFrame()
else:
    sched = sched_season.copy()

# If sched empty, try ESPN events fallback for the selected week
if sched.empty:
    # try core events fallback for the selected week only
    st.info("Attempting fallback to ESPN core events for week...")
    # attempt one-week fetch via scoreboard endpoint with week param might have failed above; we attempt events endpoint minimal
    try:
        r = safe_request(ESPN_EVENTS_URL, params={"limit": 100})
        if r:
            ev_json = r.json()
            # events shape unknown; best-effort not guaranteed.
            events = ev_json.get("events", [])
            rows = []
            for ev in events:
                comps = ev.get("competitions", [])
                if not comps: continue
                comp = comps[0]
                wk = comp.get("week")
                if wk != week_sel:
                    continue
                teams = comp.get("competitors", [])
                home = next((t for t in teams if t.get("homeAway")=="home"), None)
                away = next((t for t in teams if t.get("homeAway")=="away"), None)
                rows.append({
                    "season": season,
                    "week": wk,
                    "game_id": ev.get("id"),
                    "status": comp.get("status", {}).get("type", {}).get("name"),
                    "kickoff_utc": comp.get("date"),
                    "home_team": home.get("team", {}).get("displayName") if home else None,
                    "home_abbrev": home.get("team", {}).get("abbreviation") if home else None,
                    "away_team": away.get("team", {}).get("displayName") if away else None,
                    "away_abbrev": away.get("team", {}).get("abbreviation") if away else None,
                })
            if rows:
                sched = pd.DataFrame(rows)
    except Exception:
        pass

# filter to chosen week for UI
if not sched.empty:
    sched_enriched = enrich_schedule(sched, elo_map=elo_map)
    week_sched = sched_enriched[sched_enriched["week"] == week_sel].copy()
else:
    week_sched = pd.DataFrame()

if week_sched.empty:
    st.warning("No games found for this week.")
else:
    st.success(f"✅ Loaded schedule for {season}. Showing Week {week_sel}.")

# Prepare predictions
if not week_sched.empty:
    X = week_sched[model_features].copy()
    # ensure columns present
    for c in model_features:
        if c not in X.columns:
            X[c] = 0.0
    try:
        probs = model.predict_proba(X)[:,1]
    except Exception:
        probs = np.ones(len(X)) * 0.5
    week_sched["home_win_prob_model"] = probs
    # predicted points & score (very simple heuristic)
    week_sched["pred_total_pts"] = 43.9 + (week_sched["elo_diff"] / 100.0)  # simple scaling
    week_sched["pred_home_pts"] = (week_sched["pred_total_pts"] / 2.0) + (week_sched["elo_diff"] / 4.0)
    week_sched["pred_away_pts"] = week_sched["pred_total_pts"] - week_sched["pred_home_pts"]

# ---------------------------
# UI: Show games in cards (auto open)
# ---------------------------
st.markdown("## Games")
if week_sched.empty:
    st.info("No games to show for selected week.")
else:
    # iterate and show each game
    cols = st.columns([1, 3, 3, 3, 1])
    for idx, row in week_sched.reset_index().iterrows():
        with st.expander(f"{row['away_abbrev'] or row['away_team']} @ {row['home_abbrev'] or row['home_team']} — {row['kickoff_dt'].tz_convert('US/Eastern').strftime('%Y-%m-%d %H:%M ET') if pd.notna(row['kickoff_dt']) else 'TBD'}", expanded=True):
            # layout
            left, center, right = st.columns([1,4,1])
            # logos
            with left:
                away_logo = get_logo_path(row.get("away_abbrev") or row.get("away_team"))
                if away_logo:
                    try:
                        st.image(away_logo, width=64)
                    except Exception:
                        st.write(row.get("away_team", "Away"))
                else:
                    st.write(row.get("away_team", "Away"))
            with center:
                st.markdown(f"### {row.get('away_team','Away')}  @  {row.get('home_team','Home')}")
                # scores if available
                hs = row.get("home_score")
                as_ = row.get("away_score")
                status = row.get("status") or ""
                if pd.notna(hs) and pd.notna(as_):
                    st.write(f"Final: {row.get('home_score')} - {row.get('away_score')}")
                else:
                    st.write(f"Kickoff: {row.get('kickoff_dt').tz_convert('US/Eastern').strftime('%a %b %d %I:%M %p ET') if pd.notna(row['kickoff_dt']) else 'TBD'} — Status: {status}")
                # prediction & market
                prob = row.get("home_win_prob_model", 0.5)
                spread = row.get("spread", np.nan)
                ou = row.get("over_under", np.nan)
                st.write(f"Model Home Win Prob: **{prob*100:.1f}%**")
                st.write(f"Predicted Score: **{row.get('pred_home_pts', 0):.1f} - {row.get('pred_away_pts',0):.1f}**  (Total ≈ {row.get('pred_total_pts',0):.1f})")
                st.write(f"Vegas Spread: **{spread if not pd.isna(spread) else 'N/A'}**  |  O/U: **{ou if not pd.isna(ou) else 'N/A'}**")
                # compute edge vs market (if market exists)
                market_prob = None
                if not pd.isna(spread):
                    # convert spread -> home win prob (rough)
                    market_prob = 1 / (1 + 10 ** ((-spread) / 13.5))  # logistic-ish mapping
                blended = None
                if market_prob is not None:
                    blended = (1.0 - market_weight) * prob + market_weight * market_prob
                    edge_pp = (blended - market_prob) * 100.0
                else:
                    blended = prob
                    edge_pp = None
                if market_prob is not None:
                    st.write(f"Market Prob: **{market_prob*100:.1f}%**  |  Blended: **{blended*100:.1f}%**  |  Edge: **{edge_pp:+.1f} pp**")
                else:
                    st.write(f"Blended (model only): **{blended*100:.1f}%**")
                # Recommendation
                place_bet = False
                rec_text = "🚫 No Bet"
                if edge_pp is not None:
                    if abs(edge_pp) >= bet_threshold:
                        place_bet = True
                        if blended > market_prob:
                            rec_text = f"🟢 Bet Home (model advantage {edge_pp:.1f} pp)"
                        else:
                            rec_text = f"🔴 Bet Away (model disadvantage {edge_pp:.1f} pp)"
                else:
                    rec_text = "⚠️ Insufficient market data"
                st.markdown(f"**Recommendation:** {rec_text}")

            with right:
                home_logo = get_logo_path(row.get("home_abbrev") or row.get("home_team"))
                if home_logo:
                    try:
                        st.image(home_logo, width=64)
                    except Exception:
                        st.write(row.get("home_team", "Home"))
                else:
                    st.write(row.get("home_team", "Home"))

# ---------------------------
# Model Tracker & ROI (simple)
# ---------------------------

st.markdown("---")
st.markdown("## Model Tracker & ROI")

def compute_simple_pnl(sched_df):
    """
    Compute very simple PnL based on historical labeled games in sched_df.
    Bets placed where edge >= threshold and outcome known.
    """
    if sched_df is None or sched_df.empty:
        return 0.0, 0, 0.0
    pnl = 0.0
    bets = 0
    wins = 0
    for _, r in sched_df.iterrows():
        if pd.isna(r.get("home_score")) or pd.isna(r.get("away_score")):
            continue
        # if recommendation existed: (we simulate using model only)
        prob = r.get("home_win_prob_model", 0.5)
        spread = r.get("spread", np.nan)
        market_prob = None
        if not pd.isna(spread):
            market_prob = 1 / (1 + 10 ** ((-spread) / 13.5))
        blended = prob if market_prob is None else (1.0-market_weight) * prob + market_weight * market_prob
        edge_pp = None if market_prob is None else (blended - market_prob) * 100.0
        if edge_pp is not None and abs(edge_pp) >= bet_threshold:
            bets += 1
            # simple unit bet: if predicted winner equals actual -> win 1 unit else lose 1 unit
            predicted_home = blended >= 0.5
            actual_home = r["home_score"] > r["away_score"]
            if predicted_home == actual_home:
                pnl += 1.0
                wins += 1
            else:
                pnl -= 1.0
    roi = (pnl / bets * 100.0) if bets > 0 else 0.0
    return pnl, bets, roi

# compute from historical if possible
pnl, bets, roi = compute_simple_pnl(hist_df)
st.metric("PnL (sim)", f"{pnl:+.2f} units", f"{roi:.1f}% ROI" if bets > 0 else "No bets")
st.write(f"Total Bets simulated: {bets}")

# ---------------------------
# Footer / developer notes
# ---------------------------
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — Schedule source: ESPN. Odds source: TheOddsAPI (live) if key provided. Logos: /public (local).")

# ---------------------------
# End
# ---------------------------
>>>>>>> 5ce4016bfe9027cefdf333e500dc10a940d1a50e
