# app.py — DJBets NFL Predictor (Option 2: ESPN live schedule; no local schedule.csv)
# Full replacement file — paste/overwrite your existing streamlit_app.py

import os
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from io import BytesIO

import requests
import pandas as pd
import numpy as np
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
MAX_WEEKS = 18
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
ESPN_EVENTS_URL = "https://site.api.espn.com/apis/v2/sports/football/leagues/nfl/events"  # fallback

# Model features expected
MODEL_FEATURES = ["elo_diff", "inj_diff", "temp_c", "wind_kph", "precip_prob"]
# If spread/OU are available we will include them, but keep model stable to above features to avoid mismatch.

# ---------------------------
# Helpers
# ---------------------------

def read_odds_api_key():
    key = os.environ.get(ODDS_API_KEY_ENV)
    if key:
        return key.strip()
    if ODDS_API_PATH.exists():
        return ODDS_API_PATH.read_text().strip()
    return None

ODDS_API_KEY = read_odds_api_key()

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
