# streamlit_app.py
"""
DJBets NFL Predictor - Streamlit app (full file)
Features:
 - Historical training from /data/nfl_archive_10Y.json (if present)
 - ESPN schedule scraping for current games
 - OddsAPI integration (optional) for live spreads/totals; key in env ODDS_API_KEY or /data/odds_api_key.txt
 - XGBoost classifier trained on consistent features
 - Sidebar with week selector at top, model sliders, ROI & model record
 - Logos loaded from /public/{team}.png (team names lowercase, full name e.g. bears.png)
 - Robust fallbacks for missing fields
 - All UI cards expanded by default; Away @ Home notation
"""

import os
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ML libs
try:
    import xgboost as xgb
except Exception:
    xgb = None  # handled later

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# HTTP / scraping
import requests
from bs4 import BeautifulSoup

# ---------- Configuration ----------
DATA_DIR = Path("/data")
PUBLIC_DIR = Path("public")  # relative to repo root, Streamlit serves /public automatically
HIST_JSON = DATA_DIR / "nfl_archive_10Y.json"
ODDS_KEY_FILE = DATA_DIR / "odds_api_key.txt"
ODDS_API_KEY = os.environ.get("ODDS_API_KEY") or (ODDS_KEY_FILE.read_text().strip() if ODDS_KEY_FILE.exists() else None)
THIS_YEAR = datetime.utcnow().year if datetime.utcnow().month >= 2 else datetime.utcnow().year  # safe default
SEASON_DEFAULT = 2025

# Features we will train and predict on (consistent across train/predict)
FEATURES = ["elo_diff", "inj_diff", "spread", "over_under", "temp_c"]

# Streamlit page config
st.set_page_config(page_title="DJBets NFL Predictor", layout="wide", initial_sidebar_state="expanded")

# ---------- Helpers ----------
def safe_read_json(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception as e:
        st.warning(f"Could not load {path}: {e}")
        return None

def read_oddsapi_key() -> Optional[str]:
    if ODDS_API_KEY:
        return ODDS_API_KEY.strip()
    if ODDS_KEY_FILE.exists():
        try:
            return ODDS_KEY_FILE.read_text().strip()
        except Exception:
            return None
    return None

def get_logo_path(team_name: str) -> Optional[str]:
    """Return path to logo in /public or None."""
    if not team_name:
        return None
    fname = f"{team_name.lower()}.png"
    p = PUBLIC_DIR / fname
    if p.exists():
        return str(p)
    # allow spaces -> underscores fallback
    p2 = PUBLIC_DIR / fname.replace(" ", "_")
    if p2.exists():
        return str(p2)
    return None

def ensure_numeric_column(df: pd.DataFrame, col: str, fill: float = 0.0):
    if col not in df.columns:
        df[col] = fill
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill)

def basic_elo_from_history(df_hist: pd.DataFrame) -> pd.DataFrame:
    """Compute a simple rolling Elo per team from historical record (fallback)."""
    # This returns a DataFrame {dkey -> elo_home_pre, elo_away_pre} merged by a unique game key later.
    teams = {}
    rows = []
    df_hist = df_hist.sort_values(["season", "week", "date"])
    for _, r in df_hist.iterrows():
        h = r.get("home_team")
        a = r.get("away_team")
        hs = teams.get(h, 1500)
        as_ = teams.get(a, 1500)
        # record pre-game
        rows.append({
            "date": r.get("date", pd.NaT),
            "season": int(r.get("season", SEASON_DEFAULT)),
            "week": int(r.get("week", 0) or 0),
            "home_team": h,
            "away_team": a,
            "elo_home_pre": hs,
            "elo_away_pre": as_
        })
        # update post-game
        res = None
        if not math.isnan(r.get("home_score", np.nan)) and not math.isnan(r.get("away_score", np.nan)):
            if float(r["home_score"]) > float(r["away_score"]):
                res = 1.0
            elif float(r["home_score"]) < float(r["away_score"]):
                res = 0.0
            else:
                res = 0.5
        else:
            res = 0.5  # tie / unknown
        K = 20
        expected_h = 1.0 / (1 + 10 ** ((as_ - hs) / 400))
        hs_new = hs + K * (res - expected_h)
        as_new = as_ + K * ((1 - res) - (1 - expected_h))
        teams[h] = hs_new
        teams[a] = as_new
    return pd.DataFrame(rows)

# ---------- Data fetchers ----------
@st.cache_data(ttl=3600)
def fetch_espn_schedule(season: int = SEASON_DEFAULT) -> pd.DataFrame:
    """
    Scrape ESPN schedule for given season (current/upcoming).
    NOTE: ESPN layout can change; this is a best-effort parse that returns columns:
     - season, week, home_team, away_team, kickoff_et (datetime or NaT), home_score, away_score, venue
    """
    schedules = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; DJBets/1.0)"}
    # ESPN game schedule endpoint pattern (best effort)
    # We'll attempt to use ESPN scoreboard pages week-by-week for current season
    try:
        # For current season, try scoreboard route
        url = f"https://www.espn.com/nfl/schedule/_/week/1"
        # We will instead get scoreboard for current weeks around now (espn uses date param)
        # Simpler: fetch ESPN scoreboard for today and for next 14 days to capture upcoming matchups
        for ddelta in range(-3, 21):  # a range to capture preseason/current weeks
            dt = (datetime.utcnow() + timedelta(days=ddelta)).date()
            url = f"https://www.espn.com/nfl/scoreboard/_/date/{dt.strftime('%Y%m%d')}"
            resp = requests.get(url, headers=headers, timeout=12)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            # Parse scoreboard items
            games = soup.select("section.Scoreboard") or soup.select(".scoreboard")
            # fallback if structure differs
            for g in games:
                try:
                    teams = g.select(".team-name") or g.select(".TeamName") or g.select("abbr")
                    # Find home and away - ESPN markup is inconsistent; we'll find text and scores
                    tnames = g.select(".team-name") or g.select(".team")
                    # Better approach: find team labels and score containers
                    team_elems = g.select(".ScoreCell__TeamName") or g.select(".team-name")
                    score_elems = g.select(".ScoreCell__Score") or g.select(".score")
                    # Find team names and scores
                    names = [t.get_text(strip=True) for t in team_elems][:2]
                    scores = [s.get_text(strip=True) for s in score_elems][:2]
                    if len(names) < 2:
                        # try a different selector
                        names = [n.get_text(strip=True) for n in g.find_all("abbr")][:2]
                    if len(names) < 2:
                        continue
                    away, home = names[0], names[1]
                    home_score, away_score = (np.nan, np.nan)
                    if len(scores) >= 2:
                        away_score = float(scores[0]) if scores[0].isdigit() else np.nan
                        home_score = float(scores[1]) if scores[1].isdigit() else np.nan
                    # kickoff detection
                    kickoff_text = g.select_one(".game-time") or g.select_one(".ScoreboardStatus") or None
                    kickoff_dt = pd.NaT
                    if kickoff_text:
                        kt = kickoff_text.get_text(strip=True)
                        # try parse relative expiry: "Final", "Postponed", or time like "7:15 PM ET"
                        try:
                            if "ET" in kt or ":" in kt:
                                # attach date dt
                                # Use dt from the scoreboard page
                                kickoff_dt = pd.to_datetime(f"{dt} {kt.replace('ET','').strip()}")
                        except Exception:
                            kickoff_dt = pd.NaT
                    schedules.append({
                        "season": season,
                        "week": None,
                        "home_team": home,
                        "away_team": away,
                        "kickoff_et": kickoff_dt,
                        "home_score": home_score,
                        "away_score": away_score,
                        "source_date": dt.strftime("%Y-%m-%d")
                    })
                except Exception:
                    continue
            # stop earlier if we've already found a bunch
            if len(schedules) > 100:
                break
    except Exception as e:
        st.warning(f"ESPN scrape failed: {e}")
    if not schedules:
        st.info("ESPN schedule scraping found no games; using empty schedule.")
        return pd.DataFrame(columns=["season","week","home_team","away_team","kickoff_et","home_score","away_score"])
    df = pd.DataFrame(schedules)
    # try to fill weeks by date heuristics (not ideal but helps UI)
    df["kickoff_et"] = pd.to_datetime(df["kickoff_et"], errors="coerce")
    # assign week by date offset from season start (rough heuristic: week 1 begins first Thursday/Sunday in Sept)
    # We'll leave week None for now; UI allows selecting weeks from schedule->week if available
    return df

@st.cache_data(ttl=3600)
def fetch_odds_for_matchup(home: str, away: str, kickoff_dt: Optional[pd.Timestamp]=None) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (spread_for_home, over_under)
    Uses OddsAPI (oddsapi.com) if key present, otherwise returns (nan,nan)
    Note: OddsAPI free tier limits; this call is designed for live/upcoming only.
    """
    key = read_oddsapi_key()
    if not key:
        return (np.nan, np.nan)
    try:
        params = {
            "apiKey": key,
            "sport": "americanfootball_nfl",
            "region": "us",
            "mkt": "spreads",  # returns spreads; totals via 'totals'
            # "dateFormat": "iso"
        }
        # OddsAPI provides endpoints; we'll query odds for sport and filter by teams
        resp = requests.get("https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/", params={"apiKey":key,"regions":"us","markets":"spreads,totals","dateFormat":"iso"}, timeout=12)
        if resp.status_code != 200:
            return (np.nan, np.nan)
        j = resp.json()
        # find event matching teams
        for ev in j:
            title = ev.get("teams", [])
            if not title:
                continue
            # normalize names a bit
            teams = [t.lower() for t in ev["teams"]]
            if home.lower() in teams and away.lower() in teams:
                # extract spread (bookmakers[0].markets...)
                spread = np.nan
                ou = np.nan
                for b in ev.get("bookmakers", []):
                    for m in b.get("markets", []):
                        if m.get("key") == "spreads":
                            # picks first market outcome for home (positive means favorite? careful)
                            for o in m.get("outcomes", []):
                                if o.get("name", "").lower() == home.lower():
                                    spread = float(o.get("point", 0.0)) * (1 if o.get("price", None) else 1)
                        if m.get("key") == "totals":
                            for o in m.get("outcomes", []):
                                if "over" in o.get("name", "").lower():
                                    ou = float(o.get("point", np.nan))
                return (spread, ou)
    except Exception:
        return (np.nan, np.nan)
    return (np.nan, np.nan)

# ---------- Historical + features ----------
@st.cache_data(ttl=3600)
def load_historical() -> Optional[pd.DataFrame]:
    df = safe_read_json(HIST_JSON)
    if df is None:
        st.info("No historical data file found in /data. I'll attempt to fetch alternatives or use simulated training.")
        return None
    # normalize columns expected: season, week, home_team, away_team, home_score, away_score, date, spread, over_under
    for c in ["season","week","home_team","away_team","home_score","away_score","date","spread","over_under"]:
        if c not in df.columns:
            df[c] = np.nan
    # cast types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["season"] = pd.to_numeric(df["season"], errors="coerce").fillna(SEASON_DEFAULT).astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    return df

def merge_with_elo_and_features(historical: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Given historical dataframe, return training dataset with FEATURES and label 'home_win'
    If no historical provided, returns simulated dataset.
    """
    if historical is None or historical.empty:
        # simulate small training set to avoid crashes
        st.warning("No historical games available — using simulated training data (fallback).")
        rng = np.random.RandomState(42)
        n = 600
        sim = pd.DataFrame({
            "elo_diff": rng.normal(0, 75, n),
            "inj_diff": rng.normal(0, 0.5, n),
            "spread": rng.normal(0, 5, n),
            "over_under": rng.normal(45, 3, n),
            "temp_c": rng.normal(15, 8, n),
        })
        sim["home_win"] = (sim["elo_diff"] + rng.normal(0,40,n) > 0).astype(int)
        return sim
    df = historical.copy()
    # Use simple elo calculation from history
    elo_df = basic_elo_from_history(df)
    # Merge elo_df into historical on home/away and nearby date
    # Create keys for merge: season/week/home/away
    df["dkey"] = df.apply(lambda r: f"{int(r['season'])}_{int(r['week'])}_{r['home_team']}_{r['away_team']}", axis=1)
    elo_df["dkey"] = elo_df.apply(lambda r: f"{int(r['season'])}_{int(r['week'])}_{r['home_team']}_{r['away_team']}", axis=1)
    merged = pd.merge(df, elo_df[["dkey","elo_home_pre","elo_away_pre"]], on="dkey", how="left")
    merged["elo_home_pre"].fillna(1500, inplace=True)
    merged["elo_away_pre"].fillna(1500, inplace=True)
    merged["elo_diff"] = merged["elo_home_pre"] - merged["elo_away_pre"]
    ensure_numeric_column(merged, "spread", 0.0)
    ensure_numeric_column(merged, "over_under", 45.0)
    # injuries/weather temps are rarely present — simulate / fallback to 0
    ensure_numeric_column(merged, "inj_diff", 0.0)
    ensure_numeric_column(merged, "temp_c", 12.0)
    # Label
    merged["home_win"] = ((merged["home_score"] > merged["away_score"]).astype(int)).fillna(0).astype(int)
    # Keep only features + label
    out = merged[FEATURES + ["home_win"]].copy()
    # if there are NaNs, fill with defaults
    out = out.fillna({"spread":0.0,"over_under":45.0,"inj_diff":0.0,"temp_c":12.0,"elo_diff":0.0})
    return out

# ---------- Model training/prediction ----------
@st.cache_resource(ttl=3600)
def train_model(df_train: pd.DataFrame):
    if xgb is None:
        raise RuntimeError("xgboost is required but not installed in runtime.")
    X = df_train[FEATURES].values
    y = df_train["home_win"].values
    # small safety: if not enough samples, use simulated
    if len(X) < 50:
        # simulate
        rng = np.random.RandomState(1)
        X = rng.normal(size=(300, len(FEATURES)))
        y = (rng.rand(300) > 0.5).astype(int)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0
    )
    model.fit(X, y)
    return model

def predict_for_schedule(df_sched: pd.DataFrame, model) -> pd.DataFrame:
    df = df_sched.copy()
    # Ensure feature columns exist
    for c in FEATURES:
        if c not in df.columns:
            # reasonable defaults
            if c == "elo_diff":
                df[c] = 0.0
            elif c == "inj_diff":
                df[c] = 0.0
            elif c == "temp_c":
                df[c] = 12.0
            elif c == "spread":
                df[c] = 0.0
            elif c == "over_under":
                df[c] = 45.0
            else:
                df[c] = 0.0
    # Predict probabilities
    try:
        preds = model.predict_proba(df[FEATURES].values)[:,1]
        df["home_win_prob_model"] = preds
    except Exception as e:
        st.warning(f"Model predict failed: {e}")
        df["home_win_prob_model"] = 0.5
    return df

# ---------- Utility calculations ----------
def compute_model_record(hist_df: pd.DataFrame, model) -> Tuple[int,int,float]:
    """Compute historical accuracy of model on hist_df (if possible)"""
    if hist_df is None or hist_df.empty:
        return (0,0,0.0)
    df = merge_with_elo_and_features(hist_df)
    try:
        preds = model.predict(df[FEATURES].values)
    except Exception:
        return (0,0,0.0)
    true = df["home_win"].values
    correct = int((preds == true).sum())
    total = len(true)
    incorrect = total - correct
    pct = round(100.0 * correct / total, 2) if total else 0.0
    return (correct, incorrect, pct)

def compute_roi(schedule_df: pd.DataFrame, threshold_pp: float=3.0):
    """
    Simple ROI calculator using fixed stake per bet and threshold on model-market edge in percentage points.
    - threshold_pp: minimum edge in percentage points (pp) for placing bet
    """
    bets = []
    for _, r in schedule_df.iterrows():
        # Need model prob and market prob
        p_model = r.get("home_win_prob_model", np.nan)
        market_prob = r.get("market_home_win_prob", np.nan)
        if math.isnan(p_model) or math.isnan(market_prob):
            continue
        # edge in pp
        edge_pp = (p_model - market_prob) * 100.0
        if abs(edge_pp) < threshold_pp:
            continue  # no bet
        # make bet: stake 1 unit on whichever team model favors
        bet_on_home = p_model >= 0.5
        # resolve if game complete
        if not math.isnan(r.get("home_score")) and not math.isnan(r.get("away_score")):
            home_won = float(r["home_score"]) > float(r["away_score"])
            won = (home_won and bet_on_home) or (not home_won and not bet_on_home)
            payout = 1.9 if won else -1.0
        else:
            # not resolved yet
            payout = 0.0
            won = None
        bets.append({"edge_pp":edge_pp, "bet_on_home":bet_on_home, "payout":payout, "won":won})
    if not bets:
        return 0.0, 0, 0.0
    pnl = sum(b["payout"] for b in bets)
    bets_made = len(bets)
    roi = round(100.0 * pnl / bets_made, 2) if bets_made else 0.0
    return pnl, bets_made, roi

# ---------- UI / Main ----------
st.title("🏈 DJBets NFL Predictor — Streamlit Edition")

# Sidebar layout with week dropdown at top
st.sidebar.markdown("## 📅 Week & Season")
# We'll populate weeks from schedule once loaded; temporary placeholder:
selected_season = st.sidebar.selectbox("Season", [SEASON_DEFAULT, SEASON_DEFAULT-1, SEASON_DEFAULT-2], index=0, key="season")
# placeholder for weeks; will be updated after schedule load
# We'll set a session_state default for week if missing
if "week" not in st.session_state:
    st.session_state["week"] = 1

# Model tuning controls
st.sidebar.markdown("## ⚙️ Model Controls")
market_weight = st.sidebar.slider("Market weight (blend model vs market)", 0.0, 1.0, 0.3, 0.05, help="How much weight to give the market probability vs model. 0 = model-only, 1 = market-only.")
bet_threshold_pp = st.sidebar.slider("Bet edge threshold (pp)", 0, 10, 3, 1, help="Minimum edge (percentage points) between model & market required to place a bet.")
st.sidebar.markdown("---")
# Display model record and ROI placeholders
st.sidebar.markdown("## 📈 Model Tracker")
if "model_record" not in st.session_state:
    st.session_state["model_record"] = {"correct":0,"incorrect":0,"pct":0.0}
st.sidebar.metric("Correct", st.session_state["model_record"]["correct"], delta=None)
st.sidebar.metric("Incorrect", st.session_state["model_record"]["incorrect"], delta=None)
st.sidebar.metric("Accuracy %", f"{st.session_state['model_record']['pct']}%")
st.sidebar.markdown("---")
st.sidebar.markdown("## 🗂️ Project")
st.sidebar.write("DJBets — maintain previous functionality. Logos expected at `/public/{team}.png`")

# Fetch data
with st.spinner("Loading historical data..."):
    hist = load_historical()

with st.spinner("Scraping ESPN schedule..."):
    espn_sched = fetch_espn_schedule(selected_season)

# Ensure espn_sched columns and basic cleaning
if espn_sched is None:
    espn_sched = pd.DataFrame(columns=["season","week","home_team","away_team","kickoff_et","home_score","away_score"])
espn_sched["home_team"] = espn_sched["home_team"].astype(str)
espn_sched["away_team"] = espn_sched["away_team"].astype(str)
espn_sched["kickoff_et"] = pd.to_datetime(espn_sched["kickoff_et"], errors="coerce")

# derive weeks from dates if None: simple heuristic grouping by Monday date ranges -> provide 1..18
if espn_sched["kickoff_et"].notna().any():
    # week 1 = earliest kickoff date in Sept/Oct window; fallback: group by ISO week number relative to min
    min_date = espn_sched["kickoff_et"].min()
    def guess_week(dt):
        if pd.isna(dt):
            return 1
        delta_days = (dt.date() - min_date.date()).days
        w = int(delta_days // 7) + 1
        return max(1, min(w, 18))
    espn_sched["week"] = espn_sched["kickoff_et"].apply(guess_week)
else:
    espn_sched["week"] = espn_sched.get("week", 1).fillna(1).astype(int)

# Populate week selector in sidebar (top)
weeks_available = sorted(espn_sched["week"].unique()) if not espn_sched.empty else [1]
sel_week = st.sidebar.selectbox("Week", weeks_available, index=0, key="week")
st.session_state["week"] = sel_week

# Merge in spreads & totals via OddsAPI or fallback
odds_key = read_oddsapi_key()
if odds_key:
    st.info("OddsAPI API key found — attempting to fetch market odds for upcoming games.")
else:
    st.info("No OddsAPI key found — using mocked spreads/totals where necessary.")

# For each game, fetch market odds (cached)
espn_sched_cols = espn_sched.copy()
espn_sched_cols["spread"] = np.nan
espn_sched_cols["over_under"] = np.nan
for i, r in espn_sched_cols.iterrows():
    try:
        s, ou = fetch_odds_for_matchup(r["home_team"], r["away_team"], r["kickoff_et"])
        espn_sched_cols.at[i, "spread"] = s
        espn_sched_cols.at[i, "over_under"] = ou
    except Exception:
        espn_sched_cols.at[i, "spread"] = np.nan
        espn_sched_cols.at[i, "over_under"] = np.nan

# If historical file present, merge historical spreads where possible for training
# Build training set
train_df = merge_with_elo_and_features(hist)

# Train or fallback
model = None
if xgb is None:
    st.error("xgboost is not installed in your environment — model training/prediction disabled. Install xgboost in your environment.")
else:
    with st.spinner("Training model..."):
        try:
            model = train_model(train_df)
            st.success("Model ready.")
        except Exception as e:
            st.warning(f"Model training failed — fallback to default simple predictor. Error: {e}")
            model = None

# If model present, compute model record on historical
if model is not None:
    correct, incorrect, pct = compute_model_record(hist, model)
    st.session_state["model_record"] = {"correct": correct, "incorrect": incorrect, "pct": pct}
else:
    st.session_state["model_record"] = st.session_state.get("model_record", {"correct":0,"incorrect":0,"pct":0.0})

# Predict on schedule for selected week
week_df = espn_sched_cols[espn_sched_cols["week"] == sel_week].copy()
# fill missing numeric features before predict
for c in FEATURES:
    if c not in week_df.columns:
        week_df[c] = 0.0
# If spread or over_under missing, fill with market-ish defaults (random small noise)
week_df["spread"] = pd.to_numeric(week_df["spread"], errors="coerce")
week_df["over_under"] = pd.to_numeric(week_df["over_under"], errors="coerce")
week_df["spread"].fillna(0.0, inplace=True)
week_df["over_under"].fillna(45.0, inplace=True)
# elo_diff: attempt simple calculation via historical elo_df if possible
if hist is not None and not hist.empty:
    elo_df = basic_elo_from_history(hist)
    # quick approximate join by team names: compute average elo per team from elo_df
    avg_elo = {}
    for _, r in elo_df.iterrows():
        avg_elo[r["home_team"]] = avg_elo.get(r["home_team"], []) + [r["elo_home_pre"]]
        avg_elo[r["away_team"]] = avg_elo.get(r["away_team"], []) + [r["elo_away_pre"]]
    avg_elo = {k: np.mean(v) for k,v in avg_elo.items()}
    def calc_elo_diff(row):
        h = row["home_team"]; a = row["away_team"]
        eh = avg_elo.get(h, 1500)
        ea = avg_elo.get(a, 1500)
        return eh - ea
    week_df["elo_diff"] = week_df.apply(calc_elo_diff, axis=1)
else:
    week_df["elo_diff"] = 0.0
# inj_diff and temp_c fallback
week_df["inj_diff"] = week_df.get("inj_diff", 0.0)
week_df["temp_c"] = week_df.get("temp_c", 12.0)

# Run predictions
if model is not None:
    week_df = predict_for_schedule(week_df, model)
else:
    week_df["home_win_prob_model"] = 0.5

# Market probability estimation from spread (simple transform)
def spread_to_prob(spread):
    # Convert spread to implied home win probability via logistic approx
    if np.isnan(spread):
        return np.nan
    # spread positive implies home is favored by 'spread' points
    # rough conversion: 1 point ≈ 2.5 pp (very rough). We'll use logistic
    try:
        val = 1.0 / (1 + 10 ** (-spread / 13.0))  # tuned constant
        return val
    except Exception:
        return np.nan

week_df["market_home_win_prob"] = week_df["spread"].apply(lambda s: spread_to_prob(s) if not np.isnan(s) else np.nan)
# If market prob missing, fallback to 0.5
week_df["market_home_win_prob"].fillna(0.5, inplace=True)

# Blended probability
week_df["blended_prob"] = (1 - market_weight) * week_df["home_win_prob_model"] + (market_weight) * week_df["market_home_win_prob"]

# Edge (pp)
week_df["edge_pp"] = (week_df["blended_prob"] - week_df["market_home_win_prob"]) * 100.0

# Compute recommendations
def recommend_row(r):
    edge_pp = r.get("edge_pp", 0.0)
    if np.isnan(edge_pp) or abs(edge_pp) < bet_threshold_pp:
        return "🚫 No Bet"
    return "🏁 Bet Home" if r["blended_prob"] >= 0.5 else "🏁 Bet Away"

week_df["recommendation"] = week_df.apply(recommend_row, axis=1)

# Determine result correctness for completed games
def is_completed(r):
    return (not pd.isna(r.get("home_score"))) and (not pd.isna(r.get("away_score")))

def was_model_correct(r):
    if not is_completed(r):
        return None
    model_pick_home = r["home_win_prob_model"] >= 0.5
    actual_home_won = float(r["home_score"]) > float(r["away_score"])
    return model_pick_home == actual_home_won

week_df["model_correct"] = week_df.apply(was_model_correct, axis=1)

# Compute ROI / PnL stats and display
pnl, bets_made, roi = compute_roi(week_df, threshold_pp=bet_threshold_pp)
# Show main header summary
st.markdown(f"### Season: {selected_season} — Week {sel_week}")
col1, col2, col3, col4 = st.columns([2,1,1,1])
with col1:
    st.write(f"Games: {len(week_df)}")
with col2:
    st.metric("📈 ROI", f"{roi}%", delta=None)
with col3:
    st.metric("💵 PnL (units)", f"{pnl:.2f}", delta=None)
with col4:
    st.metric("🎯 Bets made", f"{bets_made}", delta=None)

# Show games as cards (expanded by default)
for idx, row in week_df.reset_index(drop=True).iterrows():
    home = row["home_team"]
    away = row["away_team"]
    kickoff = row.get("kickoff_et")
    kickoff_str = kickoff.strftime("%Y-%m-%d %H:%M ET") if pd.notna(kickoff) else "TBD"
    home_logo = get_logo_path(home)
    away_logo = get_logo_path(away)
    # result display
    status = "Not started"
    if not pd.isna(row.get("home_score")) and not pd.isna(row.get("away_score")):
        status = f"Final — {int(row['home_score'])}–{int(row['away_score'])}"
    # predicted
    prob = row.get("home_win_prob_model", 0.5)
    market_p = row.get("market_home_win_prob", 0.5)
    blended = row.get("blended_prob", 0.5)
    edge = row.get("edge_pp", 0.0)
    rec = row.get("recommendation", "No Bet")
    model_correct = row.get("model_correct", None)
    if model_correct is True:
        correctness = "✅ Model correct"
    elif model_correct is False:
        correctness = "❌ Model incorrect"
    else:
        correctness = ""
    # Layout card
    with st.container():
        st.markdown("---")
        cols = st.columns([0.8,3.2,1.5,1.5,1.2])
        # away logo + name (left)
        with cols[0]:
            if away_logo:
                try:
                    st.image(away_logo, width=64)
                except Exception:
                    st.write("")
            else:
                st.write("")
        with cols[1]:
            st.markdown(f"**{away}**  `@`  **{home}**")
            st.caption(f"{kickoff_str} • {status}")
            if correctness:
                st.write(correctness)
        with cols[2]:
            st.metric("Model (home)", f"{prob*100:.1f}%")
            st.write(f"Market: {market_p*100:.1f}%")
        with cols[3]:
            st.metric("Blended", f"{blended*100:.1f}%")
            st.write(f"Edge: {edge:.1f} pp")
        with cols[4]:
            st.write(f"Spread: {row.get('spread', 'N/A')}")
            st.write(f"O/U: {row.get('over_under', 'N/A')}")
            st.write(f"Rec: {rec}")

# Footer: debug / advanced controls
st.markdown("---")
with st.expander("Developer / Debug"):
    st.write("Weeks available:", weeks_available)
    st.write("Sample schedule rows:")
    st.dataframe(week_df.head(20))

st.success("Rendered. If logos do not appear, confirm `/public/{teamname}.png` files and teamname matching (lowercase, no abbreviations).")

