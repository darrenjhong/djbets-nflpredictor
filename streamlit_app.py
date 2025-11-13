# streamlit_app.py
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
import requests
import traceback
from datetime import datetime, timezone
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np

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
MAX_WEEKS = 18

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
    return None


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