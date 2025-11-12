# streamlit_app.py
"""
DJBets NFL Predictor - Streamlit main app (v: B - ESPN historical + local archive training)
Requirements:
 - pandas, numpy, requests, bs4, xgboost (optional), scikit-learn, matplotlib, streamlit
 - Place nfl_archive_10Y.json in ./data/
 - Place logos in ./public/ (filenames tolerant e.g. bears.png, newengland-patriots.png)
 - Optionally place OddsAPI key in ./data/odds_api_key.txt or set ODDS_API_KEY env var
"""

from datetime import datetime, timezone
import os
import io
import json
import time
import math
import re
from functools import lru_cache

import requests
import pandas as pd
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup

# ML libs
try:
    import xgboost as xgb
    has_xgb = True
except Exception:
    has_xgb = False
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Config & constants
# -------------------------------------------------------------------
st.set_page_config(page_title="DJBets NFL Predictor", layout="wide", initial_sidebar_state="expanded")
DATA_DIR = "data"
PUBLIC_DIR = "public"
LOCAL_ARCHIVE = os.path.join(DATA_DIR, "nfl_archive_10Y.json")
ODDS_KEY_FILE = os.path.join(DATA_DIR, "odds_api_key.txt")
ESPn_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
# Years to scrape from ESPN for historical training (in addition to your local archive)
ESPN_HIST_YEARS = list(range(2012, datetime.now().year + 1))
FALLBACK_SEED_ELO = 1500
K_FACTOR = 20

# Model features baseline
REQUIRED_FEATURES = ["elo_diff"]  # minimal
OPTIONAL_FEATURES = ["spread", "over_under", "temp_c", "wind_kph", "inj_diff", "precip_prob"]

# UI constants
MAX_WEEKS = 18

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def read_odds_api_key():
    # Read from env or file
    key = os.environ.get("ODDS_API_KEY")
    if key:
        return key.strip()
    if os.path.exists(ODDS_KEY_FILE):
        try:
            with open(ODDS_KEY_FILE, "r") as f:
                return f.read().strip()
        except Exception:
            return None
    return None

ODDS_API_KEY = read_odds_api_key()

def safe_request_json(url, params=None, headers=None, timeout=15):
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"Request error: {e} — {url}")
        return None

def espn_scoreboard_for_season(year):
    """
    Query ESPN scoreboard endpoint to return flattened game rows for a season.
    """
    params = {"season": year, "seasontype": 2}
    data = safe_request_json(ESPn_SCOREBOARD_URL, params=params)
    if not data:
        return pd.DataFrame()
    rows = []
    events = data.get("events", [])
    for ev in events:
        week = ev.get("week", {}).get("number") or ev.get("competitions", [{}])[0].get("week")
        comps = ev.get("competitions", [])
        for comp in comps:
            game = {}
            game["id"] = comp.get("id")
            game["season"] = year
            game["week"] = week if week is not None else comp.get("week", {}).get("number")
            game["status"] = comp.get("status", {}).get("type", {}).get("detail")
            # kickoff (convert to tz-aware)
            kickoff = comp.get("date")
            game["kickoff_ts"] = pd.to_datetime(kickoff) if kickoff else pd.NaT
            # venue
            vg = comp.get("venue", {})
            game["venue"] = vg.get("fullName")
            # competitors -> home/away
            for side in comp.get("competitors", []):
                team = side.get("team", {})
                is_home = side.get("homeAway") == "home"
                team_name = team.get("name") or team.get("displayName") or team.get("abbreviation")
                team_abbr = team.get("abbreviation") or team.get("shortDisplayName") or team.get("id")
                score = side.get("score")
                if is_home:
                    game["home_team"] = team_name
                    game["home_abbr"] = team_abbr
                    game["home_score"] = int(score) if score not in (None, "") else np.nan
                else:
                    game["away_team"] = team_name
                    game["away_abbr"] = team_abbr
                    game["away_score"] = int(score) if score not in (None, "") else np.nan
            # odds (ESPN may include 'odds' array)
            odds = comp.get("odds", [])
            # try to extract spread and total
            spread = None
            total = None
            if odds and isinstance(odds, list):
                o = odds[0]
                spread = o.get("spread")
                total = o.get("total")
            # also check 'lines' structures
            if spread is None:
                # fallback - try pickcenter or lines
                lines = comp.get("pickcenter", {}).get("lines", [])
                if lines:
                    first = lines[0]
                    spread = first.get("spread")
                    total = first.get("total") or first.get("over_under")
            game["spread"] = float(spread) if spread not in (None, "", "NA") else np.nan
            game["over_under"] = float(total) if total not in (None, "", "NA") else np.nan
            # extras: weather
            weather = comp.get("weather", {})
            game["temp_c"] = np.nan
            game["wind_kph"] = np.nan
            game["precip_prob"] = np.nan
            if weather:
                # ESPN returns temperature in F sometimes; we'll try to parse
                temp = weather.get("temp")
                if temp:
                    try:
                        t = float(temp)
                        # assume Fahrenheit -> convert to c
                        game["temp_c"] = (t - 32) * 5.0 / 9.0
                    except Exception:
                        game["temp_c"] = np.nan
                wind_kph = weather.get("wind")
                if wind_kph:
                    try:
                        # wind may be string '10 mph'
                        if isinstance(wind_kph, str):
                            m = re.search(r"(\d+(\.\d+)?)", wind_kph)
                            if m:
                                val = float(m.group(1))
                                # mph -> kph
                                game["wind_kph"] = val * 1.60934
                        else:
                            game["wind_kph"] = float(wind_kph)
                    except Exception:
                        game["wind_kph"] = np.nan
            rows.append(game)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # normalize week
    if "week" in df.columns:
        df["week"] = df["week"].astype(pd.Int64Dtype())
    return df

def load_local_archive(path=LOCAL_ARCHIVE):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            df = pd.DataFrame(raw)
            # ensure columns
            expected = ["season","week","home_team","away_team","home_score","away_score","date","spread","over_under"]
            return df
        except Exception as e:
            st.warning(f"Failed to load local archive: {e}")
            return pd.DataFrame()
    else:
        st.info("Local archive not found at data/nfl_archive_10Y.json")
        return pd.DataFrame()

def normalize_team_key(name):
    # produce a safe filename-like key for logos and matching
    if name is None: return ""
    key = str(name).lower()
    key = key.replace(" ", "-")
    key = re.sub(r"[^a-z0-9\-]", "", key)
    key = key.replace("--", "-")
    return key

@lru_cache(maxsize=512)
def get_logo_path(team_name):
    """
    Return the best matching logo path within PUBLIC_DIR for a team name.
    Tolerant lookups: exact, normalized, abbr mapping, fallback to placeholder.
    """
    if not team_name:
        return None
    # try normalized key
    key = normalize_team_key(team_name)
    candidates = [
        f"{PUBLIC_DIR}/{key}.png",
        f"{PUBLIC_DIR}/{key}.svg",
        f"{PUBLIC_DIR}/{key}.jpg",
        f"{PUBLIC_DIR}/{team_name.lower()}.png"
    ]
    # try common abbreviations mapping (ESPn -> filenames)
    # user can extend this map in repo if needed
    abbr_map = {
        "gb": "packers", "ne": "patriots", "nyj": "jets", "nyg": "giants",
        "sf": "49ers", "nwe": "patriots"
    }
    # check if team_name itself is an abbreviation (2-3 letters)
    if len(team_name.strip()) <= 4:
        k2 = abbr_map.get(team_name.lower())
        if k2:
            candidates.insert(0, f"{PUBLIC_DIR}/{k2}.png")
    # check case-insensitive files
    for c in candidates:
        if os.path.exists(c):
            return c
    # also attempt to find filename ignoring punctuation in public folder
    if os.path.isdir(PUBLIC_DIR):
        files = os.listdir(PUBLIC_DIR)
        for f in files:
            fn = os.path.splitext(f)[0].lower()
            if fn == key:
                return os.path.join(PUBLIC_DIR, f)
        # try contains
        for f in files:
            fn = os.path.splitext(f)[0].lower()
            if key in fn or fn in key:
                return os.path.join(PUBLIC_DIR, f)
    return None

def compute_simple_elo(hist_df, seed=FALLBACK_SEED_ELO, k=K_FACTOR):
    """
    Add elo_pre columns for home and away to historical games (sorted by date).
    Returns df with added columns: elo_home_pre, elo_away_pre
    Note: modifies copy and returns it.
    """
    df = hist_df.copy()
    # ensure date field exists
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT
    # sort by date (fallback to season/week)
    df = df.sort_values(by=["date", "season", "week"]).reset_index(drop=True)
    teams = {}
    def get_rating(team):
        if team not in teams:
            teams[team] = seed
        return teams[team]
    elo_home_pre = []
    elo_away_pre = []
    for i, row in df.iterrows():
        h = row.get("home_team")
        a = row.get("away_team")
        eh = get_rating(h)
        ea = get_rating(a)
        elo_home_pre.append(eh)
        elo_away_pre.append(ea)
        # update only if scores present
        hs = row.get("home_score")
        as_ = row.get("away_score")
        if pd.notna(hs) and pd.notna(as_):
            # expected
            diff = eh - ea
            expected_home = 1 / (1 + 10 ** ((ea - eh) / 400.0))
            # actual
            if hs > as_:
                actual_home = 1.0
            elif hs < as_:
                actual_home = 0.0
            else:
                actual_home = 0.5
            # margin adjustment could be added; keep simple
            change = k * (actual_home - expected_home)
            teams[h] = eh + change
            teams[a] = ea - change
    df["elo_home_pre"] = elo_home_pre
    df["elo_away_pre"] = elo_away_pre
    return df

def prepare_training_df(hist_df):
    """
    Take historical DataFrame (with scores) and produce training features and labels.
    Returns X (DataFrame) and y (Series) and list of features used.
    """
    df = hist_df.copy()
    # require home_score and away_score for labels
    df = df.dropna(subset=["home_score", "away_score"])
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=int), []
    # compute elo if not present
    if "elo_home_pre" not in df.columns or "elo_away_pre" not in df.columns:
        df = compute_simple_elo(df)
    df["elo_diff"] = df["elo_home_pre"] - df["elo_away_pre"]
    # pick features dynamically
    features = ["elo_diff"]
    # include spread/over if available in data
    for opt in ["spread", "over_under"]:
        if opt in df.columns and df[opt].notna().sum() > 0:
            features.append(opt)
    # other optional fields
    for opt in ["temp_c", "wind_kph", "inj_diff", "precip_prob"]:
        if opt in df.columns and df[opt].notna().sum() > 0:
            features.append(opt)
    # ensure features exist in dataframe
    for f in features:
        if f not in df.columns:
            df[f] = 0.0
    X = df[features].astype(float).fillna(0.0)
    y = (df["home_score"] > df["away_score"]).astype(int)
    return X, y, features

def train_model_from_df(hist_df):
    """
    Train XGBoost classifier if available else LogisticRegression.
    Returns (model, features)
    """
    X, y, features = prepare_training_df(hist_df)
    if X.empty or y.sum() == 0:
        st.warning("Not enough historical labelled games for training — using fallback model with simulated training.")
        # create fallback simulated dataset using elo differences
        # simulate logistic relationship
        sim_n = 1000
        rng = np.random.RandomState(42)
        elo_diff_sim = rng.normal(loc=0, scale=50, size=sim_n)
        probs = 1 / (1 + np.exp(-elo_diff_sim / 50.0))
        y_sim = (rng.rand(sim_n) < probs).astype(int)
        X_sim = pd.DataFrame({"elo_diff": elo_diff_sim})
        features = ["elo_diff"]
        X, y = X_sim, y_sim
    # choose model
    if has_xgb:
        try:
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=1, verbosity=0)
            model.fit(X, y)
            return model, features
        except Exception as e:
            st.warning(f"XGBoost train failed: {e}. Falling back to LogisticRegression.")
    # fallback
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model, features

def fetch_espn_current_schedule():
    """Fetch current scoreboard without specifying season to get live schedule"""
    data = safe_request_json(ESPn_SCOREBOARD_URL, params={})
    rows = []
    if not data:
        return pd.DataFrame()
    events = data.get("events", [])
    for ev in events:
        comps = ev.get("competitions", [])
        week = ev.get("week", {}).get("number")
        for comp in comps:
            row = {}
            row["season"] = ev.get("season", {}).get("year") or datetime.now().year
            row["week"] = week
            row["id"] = comp.get("id")
            row["kickoff_ts"] = pd.to_datetime(comp.get("date"))
            row["status"] = comp.get("status", {}).get("type", {}).get("name")
            for c in comp.get("competitors", []):
                if c.get("homeAway") == "home":
                    row["home_team"] = c.get("team", {}).get("name")
                    row["home_abbr"] = c.get("team", {}).get("abbreviation")
                    row["home_score"] = c.get("score")
                else:
                    row["away_team"] = c.get("team", {}).get("name")
                    row["away_abbr"] = c.get("team", {}).get("abbreviation")
                    row["away_score"] = c.get("score")
            # odds
            o = comp.get("odds")
            if o and isinstance(o, list) and len(o) > 0:
                o0 = o[0]
                row["spread"] = o0.get("spread") or np.nan
                row["over_under"] = o0.get("total") or np.nan
            rows.append(row)
    return pd.DataFrame(rows)

def get_future_odds_from_oddsapi(home_abbr, away_abbr, kickoff_ts):
    """
    Use OddsAPI to fetch odds for a given fixture (best-effort). Returns dict with spread/total or None.
    Note: Free plan likely restricts historic; we only use it for live/future games.
    """
    if not ODDS_API_KEY:
        return None
    # OddsAPI endpoint (example): https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds
    # We'll use date filter
    try:
        sport_key = "americanfootball_nfl"
        date_iso = pd.to_datetime(kickoff_ts).strftime("%Y-%m-%d")
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": "spreads,totals",
            "dateFormat": "iso",
            "oddsFormat": "american"
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        # try to find match by teams and start time (best-effort)
        for item in data:
            # bookmakers -> markets
            teams = item.get("teams", [])
            commence = item.get("commence_time")
            if not commence:
                continue
            # coarse match by date and team abbreviation
            if date_iso in commence and (home_abbr in [t.lower() for t in teams] or away_abbr in [t.lower() for t in teams]):
                # extract bookmakers[0] markets
                books = item.get("bookmakers", [])
                if not books:
                    continue
                markets = books[0].get("markets", [])
                res = {"spread": np.nan, "over_under": np.nan}
                for m in markets:
                    if m.get("key") == "spreads":
                        outcomes = m.get("outcomes", [])
                        # find home spread outcome
                        for o in outcomes:
                            if o.get("name", "").lower() == teams[0].lower():
                                res["spread"] = float(o.get("point")) if o.get("point") not in (None, "") else np.nan
                    if m.get("key") == "totals":
                        outcomes = m.get("outcomes", [])
                        if outcomes:
                            res["over_under"] = float(outcomes[0].get("point")) if outcomes[0].get("point") not in (None, "") else np.nan
                return res
    except Exception as e:
        st.info(f"OddsAPI fetch error (non-fatal): {e}")
    return None

# -------------------------------------------------------------------
# Data pipeline (load & merge)
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all_history_and_schedule():
    st.info("Loading historical data & ESPN seasons (this may take up to ~30s on first run)...")
    # Load local archive
    local_df = load_local_archive()
    # Load ESPN historical seasons (only if needed to supplement)
    espn_hist_frames = []
    for yr in ESPN_HIST_YEARS:
        try:
            df_yr = espn_scoreboard_for_season(yr)
            if not df_yr.empty:
                # normalize names & columns for arch
                # convert columns to consistent names
                if "kickoff_ts" in df_yr.columns:
                    df_yr["date"] = df_yr["kickoff_ts"]
                espn_hist_frames.append(df_yr)
        except Exception:
            continue
    if espn_hist_frames:
        espn_hist = pd.concat(espn_hist_frames, ignore_index=True, sort=False)
    else:
        espn_hist = pd.DataFrame()
    # normalize local archive (if any) columns - try to coerce column names to canonical
    # local archive may have 'date' or 'kickoff_ts' etc.
    if not local_df.empty:
        local_df = local_df.rename(columns=lambda c: c.strip())
        # ensure essential cols
        for c in ["season","week","home_team","away_team"]:
            if c not in local_df.columns:
                local_df[c] = np.nan
    # merge: prefer ESPN historical where available, then local archive append missing
    combined = pd.DataFrame()
    if not espn_hist.empty:
        combined = espn_hist.copy()
        # ensure home_score/away_score exist
        for c in ["home_score","away_score"]:
            if c not in combined.columns:
                combined[c] = np.nan
    if not local_df.empty:
        # fill missing columns in local to match combined
        append_local = local_df.copy()
        for c in ["kickoff_ts","id","home_abbr","away_abbr","spread","over_under","temp_c","wind_kph"]:
            if c not in append_local.columns:
                append_local[c] = np.nan
        # cast date
        if "date" in append_local.columns and "kickoff_ts" not in append_local.columns:
            append_local["kickoff_ts"] = pd.to_datetime(append_local["date"], errors="coerce")
        combined = pd.concat([combined, append_local], ignore_index=True, sort=False)
    # unify types
    if "week" in combined.columns:
        try:
            combined["week"] = combined["week"].astype(pd.Int64Dtype())
        except Exception:
            pass
    # compute elo timeline on combined history
    combined_for_elo = combined.copy()
    # If no scores present, Elo computation may be weak but will seed teams
    combined_for_elo["home_score"] = pd.to_numeric(combined_for_elo.get("home_score"), errors="coerce")
    combined_for_elo["away_score"] = pd.to_numeric(combined_for_elo.get("away_score"), errors="coerce")
    combined_elo = compute_simple_elo(combined_for_elo)
    st.success(f"Loaded historical data ({len(combined)} rows, Elo computed).")
    return combined_elo

# -------------------------------------------------------------------
# UI & main
# -------------------------------------------------------------------
def compute_model_record(hist_df, model, features):
    """
    Compute model record (correct/incorrect) on historical completed games.
    Returns (correct, incorrect, pct)
    """
    df = hist_df.copy()
    # require labels
    df = df.dropna(subset=["home_score", "away_score"])
    if df.empty:
        return 0, 0, 0.0
    # ensure features present
    for f in features:
        if f not in df.columns:
            df[f] = 0.0
    X = df[features].astype(float).fillna(0.0)
    try:
        probs = model.predict_proba(X)[:,1]
        preds = (probs >= 0.5).astype(int)
    except Exception:
        # fallback to predict
        preds = model.predict(X)
    truth = (df["home_score"] > df["away_score"]).astype(int).values
    correct = int((preds == truth).sum())
    incorrect = int((preds != truth).sum())
    pct = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0.0
    return correct, incorrect, pct

def compute_roi(schedule_df, model, features, threshold_pp=3.0, market_weight=0.5):
    """
    Very simple flat-bet ROI simulation:
    - For each game where edge_pp (model_pp - market_pp) > threshold_pp, bet 1 unit on recommended side
    - market_pp derived from spread -> convert spread to implied prob using normal approximation (simple)
    """
    df = schedule_df.copy()
    df["recommendation"] = "No Bet"
    df["edge_pp"] = np.nan
    df["roi_result"] = 0.0
    bets = []
    bankroll = 0.0
    for i, r in df.iterrows():
        # skip if no model prob
        mp = r.get("home_win_prob_model")
        if mp is None or pd.isna(mp):
            continue
        # derive market implied prob from spread if available
        s = r.get("spread")
        market_pp = np.nan
        if not pd.isna(s):
            try:
                # convert spread (home points) to prob via logistic approx
                # pts -> prob_home = 1/(1+exp(-spread/7))
                market_pp = 1 / (1 + math.exp(- ( -float(s) ) / 7.0 ))  # negative spread means home favored
            except Exception:
                market_pp = np.nan
        # blended = weighted model + market
        if pd.isna(market_pp):
            blended = mp
        else:
            blended = (model_prob_weight := (1.0 - market_weight)) * mp + market_weight * market_pp
        edge_pp = (mp - market_pp) * 100 if not pd.isna(market_pp) else np.nan
        df.at[i, "edge_pp"] = edge_pp
        # decide bet
        if not pd.isna(edge_pp) and abs(edge_pp) >= threshold_pp:
            # bet on whichever side has positive edge (model - market)
            bet_on_home = edge_pp > 0
            # assume a simple payout: win pays 1 unit, lose -1 unit (ignore juice)
            # apply if actual score exists
            if pd.notna(r.get("home_score")) and pd.notna(r.get("away_score")):
                actual_home_win = r["home_score"] > r["away_score"]
                win = (actual_home_win and bet_on_home) or (not actual_home_win and not bet_on_home)
                pnl = 1.0 if win else -1.0
                bankroll += pnl
                df.at[i, "roi_result"] = pnl
                df.at[i, "recommendation"] = "Bet Home" if bet_on_home else "Bet Away"
                bets.append({"idx": i, "pnl": pnl})
            else:
                df.at[i, "recommendation"] = "Projected Bet Home" if bet_on_home else "Projected Bet Away"
    total_bets = len(bets)
    total_pnl = bankroll
    roi = (total_pnl / total_bets) * 100 if total_bets > 0 else 0.0
    return total_pnl, total_bets, roi, df

# ---------------------------
# Main app layout & flow
# ---------------------------
st.title("🏈 DJBets NFL Predictor")

# Sidebar (week selector at top and controls)
with st.sidebar:
    st.markdown("## 📅 Select Season & Week")
    this_year = datetime.now().year
    season_sel = st.selectbox("Season", options=[this_year, this_year-1, this_year-2], index=0, key="season")
    # week placeholder until we load schedule
    week_placeholder = st.empty()
    # model controls
    st.markdown("---")
    st.markdown("## ⚙️ Model Controls")
    market_weight = st.slider("Market weight (blended prob)", 0.0, 1.0, 0.5, 0.05, help="Weight for market implied probability when blending with model probability")
    bet_threshold_pp = st.slider("Bet threshold (pp)", 0, 10, 3, 1, help="Minimum edge in percentage points (pp) to recommend a bet (model - market)")
    st.markdown("### 📈 Model tuning")
    st.markdown("Use sliders to tune how aggressive the recommendations are. Hover (?) icons provide help.")
    st.markdown("---")
    # display model record placeholder
    st.markdown("## 🧾 Model Performance")
    perf_placeholder = st.empty()

# Load data and train model
with st.spinner("Loading data and training model (first run may take time)..."):
    hist = load_all_history_and_schedule()
    # prepare training df limited to seasons prior (exclude current season)
    hist_train = hist[hist["season"] < season_sel] if "season" in hist.columns else hist
    # if user chose B, we train on ESPN historical + local archive - already in hist
    model, model_features = train_model_from_df(hist_train)
    # compute model record on historical completed games
    correct, incorrect, pct = compute_model_record(hist, model, model_features)
    perf_placeholder.metric("Record", f"{correct} / {correct+incorrect} ({pct*100:.1f}%)" if (correct+incorrect)>0 else "No data")

# Prepare current schedule for selected season
with st.spinner("Fetching current season schedule from ESPN..."):
    # get current season schedule via ESPN scoreboard per-season scraping function
    espn_current = espn_scoreboard_for_season(season_sel)
    if espn_current.empty:
        st.warning("⚠️ No games loaded from ESPN for this season.")
    # ensure week column
    if "week" not in espn_current.columns:
        espn_current["week"] = 1
    # ensure kickoff_ts
    if "kickoff_ts" in espn_current.columns:
        espn_current["kickoff_ts"] = pd.to_datetime(espn_current["kickoff_ts"], errors="coerce")
    else:
        espn_current["kickoff_ts"] = pd.NaT

# Build merged view (espn_current enriched by any historical / odds)
merged = espn_current.copy()
if merged.empty:
    st.warning("No schedule found for selected season/week. Try a different season.")
# ensure team fields present
for col in ["home_team","away_team","home_abbr","away_abbr","spread","over_under","home_score","away_score"]:
    if col not in merged.columns:
        merged[col] = np.nan

# compute Elo pre for these matches using latest ranks from historical pipeline
# get latest elo ratings from hist (last known per team)
try:
    latest_elo = {}
    if "elo_home_pre" in hist.columns and "home_team" in hist.columns:
        last = hist.sort_values("date").groupby("home_team").last()
        for t,row in last.iterrows():
            latest_elo[t] = row.get("elo_home_pre", FALLBACK_SEED_ELO)
    # fallback: compute fresh quick rating from history
    # We'll compute approximate current elo per team using compute_simple_elo on historical data
    curr_elo_df = compute_simple_elo(hist)
    # take last elo_home_pre by team
    for _, r in curr_elo_df.iterrows():
        t = r.get("home_team")
        if pd.notna(t):
            latest_elo[t] = r.get("elo_home_pre", latest_elo.get(t, FALLBACK_SEED_ELO))
    # populate match elo:
    merged["elo_home"] = merged["home_team"].map(lambda t: latest_elo.get(t, FALLBACK_SEED_ELO))
    merged["elo_away"] = merged["away_team"].map(lambda t: latest_elo.get(t, FALLBACK_SEED_ELO))
    merged["elo_diff"] = merged["elo_home"] - merged["elo_away"]
except Exception as e:
    st.warning(f"Elo mapping warning: {e}")
    merged["elo_home"] = FALLBACK_SEED_ELO
    merged["elo_away"] = FALLBACK_SEED_ELO
    merged["elo_diff"] = 0.0

# Fill missing numeric features
merged["temp_c"] = merged.get("temp_c", np.nan).fillna(0.0)
merged["wind_kph"] = merged.get("wind_kph", np.nan).fillna(0.0)
merged["inj_diff"] = merged.get("inj_diff", np.nan).fillna(0.0)
merged["precip_prob"] = merged.get("precip_prob", np.nan).fillna(0.0)
# keep spread & over_under as-is (may be NaN)

# compute model probabilities for matches (ensure features list present in merged)
for f in model_features:
    if f not in merged.columns:
        merged[f] = 0.0
# prepare X for prediction
X_pred = merged[model_features].astype(float).fillna(0.0) if not merged.empty else pd.DataFrame()
try:
    if not X_pred.empty:
        try:
            merged["home_win_prob_model"] = model.predict_proba(X_pred)[:,1]
        except Exception:
            # fallback to predict if predict_proba not available
            merged["home_win_prob_model"] = model.predict(X_pred)
    else:
        merged["home_win_prob_model"] = np.nan
except Exception as e:
    st.warning(f"Model prediction failed: {e}")
    merged["home_win_prob_model"] = np.nan

# Now use OddsAPI only for future live games to supplement spread if ESPN missing
if ODDS_API_KEY:
    for idx, row in merged.iterrows():
        # only attempt for future / upcoming games and missing spread
        if pd.isna(row.get("spread")) and pd.notna(row.get("kickoff_ts")) and row.get("kickoff_ts") > pd.Timestamp.utcnow():
            odds = get_future_odds_from_oddsapi(str(row.get("home_abbr") or ""), str(row.get("away_abbr") or ""), row.get("kickoff_ts"))
            if odds:
                if "spread" in odds and not pd.isna(odds["spread"]):
                    merged.at[idx, "spread"] = odds["spread"]
                if "over_under" in odds and not pd.isna(odds["over_under"]):
                    merged.at[idx, "over_under"] = odds["over_under"]

# compute ROI / recommendations based on blended probability & threshold
total_pnl, total_bets, roi_pct, merged_with_rec = compute_roi(merged, model, model_features, threshold_pp=bet_threshold_pp, market_weight=market_weight)

# Week selector replacement (now that we have data)
weeks_available = sorted(merged_with_rec["week"].dropna().unique().tolist()) if "week" in merged_with_rec.columns else [1]
if not weeks_available:
    weeks_available = [1]
# replace the placeholder with actual selectbox
with st.sidebar:
    week_sel = st.selectbox("Week", options=weeks_available, index=0, key="week")

# Filter to selected week
week_df = merged_with_rec[merged_with_rec["week"] == int(week_sel)].copy() if "week" in merged_with_rec.columns else merged_with_rec.copy()
if week_df.empty:
    st.warning("No games found for this week.")
else:
    st.success(f"Season: {season_sel} — Week: {week_sel} — {len(week_df)} games found")

# Left column: schedule cards
col1, col2 = st.columns([3,1])

with col1:
    st.header(f"Week {week_sel} Matchups")
    for idx, row in week_df.sort_values("kickoff_ts").iterrows():
        # build card: show away @ home
        away = row.get("away_team", "Away")
        home = row.get("home_team", "Home")
        # logos
        away_logo = get_logo_path(away)
        home_logo = get_logo_path(home)
        kickoff_display = ""
        if pd.notna(row.get("kickoff_ts")):
            kickoff_display = pd.to_datetime(row.get("kickoff_ts")).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        else:
            kickoff_display = "TBD"
        # status & scores
        status = row.get("status") or ""
        home_score = row.get("home_score")
        away_score = row.get("away_score")
        score_display = ""
        if pd.notna(home_score) and pd.notna(away_score):
            score_display = f"{int(away_score)} - {int(home_score)}"
        else:
            score_display = "Not started" if (status and status.lower() != "final") else "Final"
        # probabilities
        model_prob = row.get("home_win_prob_model")
        if pd.notna(model_prob):
            model_pct = f"{model_prob*100:.1f}%"
        else:
            model_pct = "N/A"
        spread = row.get("spread")
        spread_display = f"{spread}" if not pd.isna(spread) else "N/A"
        ou = row.get("over_under")
        ou_display = f"{ou}" if not pd.isna(ou) else "N/A"
        edge = row.get("edge_pp")
        edge_display = f"{edge:.1f} pp" if not pd.isna(edge) else "N/A"
        rec = row.get("recommendation", "No Bet")
        # layout card
        with st.expander(f"{away}  @  {home}   |  {kickoff_display}", expanded=True):
            cols = st.columns([1,4,1])
            # away column
            with cols[0]:
                if away_logo:
                    try:
                        st.image(away_logo, width=60)
                    except Exception:
                        st.write(away)
                else:
                    st.write(away)
            # center details
            with cols[1]:
                st.markdown(f"**{away}  @  {home}**  —  {score_display}")
                st.write(f"Kickoff: {kickoff_display}  |  Status: {status}")
                st.write(f"Model (home win): **{model_pct}**  |  Spread: **{spread_display}**  |  O/U: **{ou_display}**  |  Edge: **{edge_display}**")
                st.write(f"Recommendation: **{rec}**")
            # home column
            with cols[2]:
                if home_logo:
                    try:
                        st.image(home_logo, width=60)
                    except Exception:
                        st.write(home)
                else:
                    st.write(home)
            # deeper analysis area
            st.markdown("**Model details**")
            # show feature bar(s)
            feat_cols = st.columns(len(model_features) if model_features else 1)
            for i, f in enumerate(model_features):
                val = row.get(f, 0.0)
                feat_cols[i].metric(f, f"{val:.2f}")
            # visualization placeholder: predicted score (simple conversion)
            # predicted margin from prob: convert p to expected margin via logit scaling
            if pd.notna(model_prob):
                try:
                    margin = (math.log(model_prob / (1 - model_prob)) * 7.0)  # hub heuristic
                    predicted_home = 21 + margin/2
                    predicted_away = 21 - margin/2
                    st.write(f"Predicted score (approx): **{int(predicted_away)} - {int(predicted_home)}**  (Total ~ {predicted_home + predicted_away:.1f})")
                except Exception:
                    pass
            st.markdown("---")

with col2:
    st.header("Model Tracker & ROI")
    st.metric("Simulated ROI", f"{roi_pct:.1f}%", delta=f"{total_pnl:.2f} units")
    st.write(f"Bets (historical simulated): {total_bets}")
    st.markdown("### Model record")
    st.write(f"Correct: {correct} — Incorrect: {incorrect} — Accuracy: {pct*100:.1f}%")
    st.markdown("### Controls help")
    st.write("**Market weight** - how much weight to give market implied probability when blending with our model. Higher means we trust market more.")
    st.write("**Bet threshold** - minimum edge (in percentage points) between model and market to place a bet.")
    st.markdown("---")
    st.write("### ELO visual (team sample)")
    # quick plot: show distribution of elo_home_pre in history
    try:
        fig, ax = plt.subplots()
        if "elo_home_pre" in hist.columns:
            ax.hist(hist["elo_home_pre"].dropna(), bins=30)
            ax.set_title("Historical Elo distribution (home)")
            st.pyplot(fig)
        else:
            st.write("Elo not available in history.")
    except Exception:
        st.write("ELO plot error (non-fatal).")

st.caption(f"Data sources: ESPN (schedule & odds where available) — Local archive: {os.path.basename(LOCAL_ARCHIVE)} — Updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")

# End