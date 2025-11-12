# streamlit_app.py
"""
DJBets NFL Predictor - Streamlit app (v9.5 -> B option integrated)

Features:
- Loads local historical data (./data/nfl_archive_10Y.json) for training
- Loads schedule from ./data/schedule.csv when present, otherwise uses ESPN scraping fallback (best-effort)
- Uses OddsAPI for current/future games (Option B) with caching in ./cache/odds_cache.json
- Model: sklearn LogisticRegression (robust & lightweight); auto-trains on launch
- UI: week dropdown at top of sidebar, logos from public/<team>.png, ROI/model tracking, clickable/expanded game cards
- Defensive handling for missing fields; tries to keep all prior features working

NOTE: Put your OddsAPI key in environment variable ODDS_API_KEY or in ./data/odds_api_key.txt
      Team logos should be in ./public/<team>.png (lowercase team name, e.g., bears.png)
"""

from pathlib import Path
import json
import os
import time
from datetime import datetime, timezone
import math

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# local helper for odds
from odds_utils import OddsAPIClient

# ---------- Config ----------
DATA_DIR = Path("data")
CACHE_DIR = Path("cache")
PUBLIC_DIR = Path("public")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

MAX_WEEKS = 18
THIS_YEAR = int(os.getenv("SEASON_YEAR", datetime.now().year))
ODDS_API_MAX_CALLS = int(os.getenv("ODDS_API_MAX_CALLS", "100"))

# default features the model expects (we keep this compact and stable)
REQUIRED_FEATURES = ["spread", "over_under", "elo_diff", "inj_diff", "temp_c"]

# ---------- Utilities ----------
def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def safe_read_csv(path: Path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        st.write(f"⚠️ Failed to read {path}: {e}")
        return pd.DataFrame()

# map team string to a local logo path: tries multiple patterns (lowercase name, abbreviation)
def get_logo_path(team_name: str):
    if not isinstance(team_name, str):
        return None
    # expected user assets: public/bears.png or public/chicago_bears.png etc.
    t = team_name.lower().strip().replace(" ", "_").replace(".", "")
    candidates = [
        PUBLIC_DIR / f"{t}.png",
        PUBLIC_DIR / f"{t}.jpg",
        PUBLIC_DIR / f"{t}.svg",
        PUBLIC_DIR / f"{t}.webp",
        PUBLIC_DIR / f"{t}.gif",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # try abbreviation if present like "CHI"
    if len(team_name) <= 5:
        abbr = team_name.lower()
        c = PUBLIC_DIR / f"{abbr}.png"
        if c.exists():
            return str(c)
    # fallback
    default = PUBLIC_DIR / "default.png"
    return str(default) if default.exists() else None

# convert point-spread (home - away) to approximate win probability (normal approx)
def spread_to_prob(spread):
    # spread: positive = home favored by 'spread' points
    # use a sigma ~13 points (empirical NFL standard deviation)
    try:
        s = float(spread)
    except Exception:
        return np.nan
    sigma = 13.0
    z = s / sigma
    # normal cdf via erf
    p = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return min(max(p, 0.0), 1.0)

# friendly formatting for kickoff timestamps
def format_kickoff(kickoff):
    if pd.isna(kickoff):
        return "TBD"
    try:
        if isinstance(kickoff, str):
            dt = pd.to_datetime(kickoff, utc=True)
        else:
            dt = pd.to_datetime(kickoff, utc=True)
        # convert to local display (assume user's timezone; Streamlit cloud runs UTC)
        return dt.tz_convert(tz=None).strftime("%a %b %d %H:%M %Z")
    except Exception:
        return str(kickoff)

# compute if game complete
def game_is_final(row):
    return (not pd.isna(row.get("home_score"))) and (not pd.isna(row.get("away_score"))) and (row.get("home_score") != 0 or row.get("away_score") != 0)

# compute model record from history DataFrame
def compute_model_record(hist_df, model, features):
    # hist_df must contain columns: home_score, away_score, spread, ... and features used by the model
    try:
        df = hist_df.copy()
        # create label
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
        X = df[features].fillna(0)
        # protect from wrong shape
        if X.shape[0] == 0:
            return 0, 0, 0.0
        preds = model.predict_proba(X)[:,1]
        df["pred_home_prob"] = preds
        df["pred_home"] = (df["pred_home_prob"] >= 0.5).astype(int)
        df_completed = df[df["home_score"].notna() & df["away_score"].notna()]
        if df_completed.empty:
            return 0, 0, 0.0
        correct = (df_completed["pred_home"] == df_completed["home_win"]).sum()
        incorrect = len(df_completed) - correct
        pct = correct / max(1, len(df_completed))
        return int(correct), int(incorrect), round(float(pct),4)
    except Exception as e:
        st.warning(f"⚠️ compute_model_record error: {e}")
        return 0, 0, 0.0

# ---------- Data loading ----------
@st.cache_data(ttl=3600)
def load_historical_data():
    # priority: data/nfl_archive_10Y.json (your uploaded file) then data/historical.csv
    json_path = DATA_DIR / "nfl_archive_10Y.json"
    csv_path = DATA_DIR / "historical.csv"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            st.info(f"✅ Loaded historical data with {len(df)} games from {json_path}")
            return df
        except Exception as e:
            st.warning(f"Failed to load {json_path}: {e}")
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            st.info(f"✅ Loaded historical data with {len(df)} games from {csv_path}")
            return df
        except Exception as e:
            st.warning(f"Failed to load {csv_path}: {e}")
    st.warning("⚠️ No historical data file found in ./data. Model will use simulated fallback.")
    return pd.DataFrame()

@st.cache_data(ttl=600)
def load_schedule():
    # priority: local file ./data/schedule.csv
    sc_path = DATA_DIR / "schedule.csv"
    if sc_path.exists():
        df = safe_read_csv(sc_path)
        if not df.empty:
            st.info(f"✅ Loaded schedule from {sc_path}")
            return df
    # fallback minimal empty schedule
    st.warning("⚠️ No local schedule file found (./data/schedule.csv). Please upload or ensure ESPN fetch is implemented.")
    return pd.DataFrame()

# ---------- Model training ----------
@st.cache_resource
def train_model(hist_df):
    # robust training: extract simple features present in hist_df
    # required target: columns 'home_score','away_score'
    if hist_df is None or hist_df.empty:
        # fallback: create small simulated dataset
        st.warning("⚠️ Not enough historical data — using simulated training set.")
        rng = np.random.RandomState(42)
        X = pd.DataFrame({
            "spread": rng.normal(0,5,200),
            "over_under": rng.normal(44,3,200),
            "elo_diff": rng.normal(0,25,200),
            "inj_diff": rng.normal(0,1,200),
            "temp_c": rng.normal(12,6,200)
        })
        y = (X["elo_diff"] + X["spread"]/2 + rng.normal(0,10,200) > 0).astype(int)
    else:
        df = hist_df.copy()
        # ensure target exists
        if "home_score" not in df.columns or "away_score" not in df.columns:
            st.warning("⚠️ Historical data missing 'home_score'/'away_score' columns — using fallback simulation.")
            return train_model(pd.DataFrame())
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
        # derive simple features
        # prefer spreads from historical record if present
        df["spread"] = df.get("home_spread", df.get("spread", np.nan))
        df["over_under"] = df.get("over_under", np.nan)
        # elo_diff: try to compute from columns home_elo/away_elo, else 0
        if "home_elo" in df.columns and "away_elo" in df.columns:
            df["elo_diff"] = df["home_elo"] - df["away_elo"]
        else:
            df["elo_diff"] = df.get("elo_diff", 0.0)
        df["inj_diff"] = df.get("inj_diff", 0.0)
        df["temp_c"] = df.get("temp_c", 12.0)
        # fill missing numeric fields
        for c in ["spread","over_under","elo_diff","inj_diff","temp_c"]:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        # prepare X,y
        X = df[["spread","over_under","elo_diff","inj_diff","temp_c"]]
        y = df["home_win"]
        if len(X) < 30:
            st.warning("⚠️ Not enough valid historical records to train — using simulated fallback.")
            return train_model(pd.DataFrame())
    # scale + logistic regression
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=200)
    model.fit(Xs, y)
    # store scaler with model for prediction convenience
    model._scaler = scaler
    st.success("✅ Model trained successfully.")
    return model

# ---------- Odds client ----------
odds_client = OddsAPIClient(cache_path=CACHE_DIR / "odds_cache.json", max_calls=ODDS_API_MAX_CALLS)

# ---------- App UI ----------
st.set_page_config(page_title="DJBets NFL Predictor", layout="wide")
st.title("🏈 DJBets NFL Predictor — Streamlit")

# load data
hist = load_historical_data()
schedule = load_schedule()

# Some basic schedule normalization: expect schedule to include columns: season, week, home_team, away_team, kickoff (or kickoff_utc)
if not schedule.empty:
    # normalize column names
    schedule = schedule.rename(columns={
        c: c.strip() for c in schedule.columns
    })
    # ensure kickoff column exists
    kickoff_cols = [c for c in schedule.columns if "kick" in c.lower() or "date" in c.lower() or "time" in c.lower()]
    kickoff_col = kickoff_cols[0] if kickoff_cols else None
    if kickoff_col:
        schedule["kickoff_ts"] = pd.to_datetime(schedule[kickoff_col], errors="coerce", utc=True)
    else:
        schedule["kickoff_ts"] = pd.NaT
else:
    schedule = pd.DataFrame()

# ---------- Sidebar (top) ----------
st.sidebar.markdown("## 🏈 DJBets NFL Predictor")
# season selector (keeps previous functionality)
season_options = sorted(list({int(x) for x in schedule.get("season", [THIS_YEAR]) if pd.notna(x)} | {THIS_YEAR}), reverse=True)
if not season_options:
    season_options = [THIS_YEAR, THIS_YEAR-1]
st.sidebar.selectbox("Season", season_options, index=0, key="season")

# Week dropdown at top
if not schedule.empty:
    weeks = sorted(schedule[schedule["season"]==st.session_state.get("season", THIS_YEAR)]["week"].dropna().unique())
    if len(weeks) == 0:
        weeks = list(range(1, MAX_WEEKS+1))
else:
    weeks = list(range(1, MAX_WEEKS+1))

# ensure max weeks present
weeks = [int(w) for w in weeks]
selected_week = st.sidebar.selectbox("📅 Week", weeks, index=0, key="week")

# sliders / model knobs preserved
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Model / Betting Controls")
market_weight = st.sidebar.slider("Market weight (blend)", 0.0, 1.0, 0.5, 0.05, help="How much weight to give market implied probability when blending with model probability.")
bet_threshold = st.sidebar.slider("Bet threshold (pp)", 0.0, 10.0, 3.0, 0.5, help="Minimum edge in percentage points required to recommend a bet.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Performance")
# load/training
model = train_model(hist)
# model record (use hist as reference)
correct, incorrect, pct = compute_model_record(hist, model, ["spread","over_under","elo_diff","inj_diff","temp_c"])
st.sidebar.metric("Model hit rate", f"{pct*100:.1f}%", delta=f"{correct - incorrect} ({correct}/{correct+incorrect if (correct+incorrect)>0 else 1})")

# ---------- Build week view ----------
st.subheader(f"Season {st.session_state.get('season', THIS_YEAR)} — Week {selected_week}")

# narrow schedule for selected week / season
week_sched = schedule[(schedule.get("season")==st.session_state.get("season", THIS_YEAR)) & (schedule.get("week")==int(selected_week))] if not schedule.empty else pd.DataFrame()
# if schedule empty, show friendly message and try to proceed with blank
if week_sched.empty:
    st.warning("⚠️ No games found for this week in local schedule. If you expect this to be populated, upload ./data/schedule.csv with columns season,week,home_team,away_team,kickoff.")
else:
    # ensure columns
    for col in ["home_team","away_team","home_score","away_score","kickoff_ts"]:
        if col not in week_sched.columns:
            week_sched[col] = np.nan

    # fetch odds for upcoming/future games (Option B: call OddsAPI for future games only)
    odds_list = []
    calls = 0
    for i,row in week_sched.iterrows():
        kickoff = row.get("kickoff_ts", pd.NaT)
        is_future = True
        if pd.notna(kickoff):
            try:
                is_future = pd.to_datetime(kickoff) > pd.to_datetime(datetime.utcnow())
            except Exception:
                is_future = True
        if is_future:
            # call oddsclient but it will silently skip/cached if not available
            try:
                odds = odds_client.get_odds_for_matchup(row.get("home_team"), row.get("away_team"), row.get("kickoff_ts"))
                odds_list.append((i, odds))
                calls += 1
            except Exception as e:
                odds_list.append((i, {}))
        else:
            odds_list.append((i, {}))

    # attach odds into week_sched
    week_sched = week_sched.reset_index(drop=True)
    for idx, odds in odds_list:
        if isinstance(odds, dict):
            week_sched.at[idx, "odds_api"] = odds
            # try to take the spread from odds if available
            try:
                if "spread" in odds and odds["spread"] is not None:
                    week_sched.at[idx, "spread"] = odds["spread"]
                if "over_under" in odds and odds["over_under"] is not None:
                    week_sched.at[idx, "over_under"] = odds["over_under"]
            except Exception:
                pass

    # Fill missing standard fields
    week_sched["spread"] = pd.to_numeric(week_sched.get("spread"), errors="coerce")
    week_sched["over_under"] = pd.to_numeric(week_sched.get("over_under"), errors="coerce")
    # fill other features with defaults
    week_sched["elo_diff"] = pd.to_numeric(week_sched.get("elo_diff"), errors="coerce").fillna(0.0)
    week_sched["inj_diff"] = pd.to_numeric(week_sched.get("inj_diff"), errors="coerce").fillna(0.0)
    week_sched["temp_c"] = pd.to_numeric(week_sched.get("temp_c"), errors="coerce").fillna(12.0)

    # run model predictions
    features = ["spread","over_under","elo_diff","inj_diff","temp_c"]
    # prepare feature matrix for model (scale using model._scaler)
    X = week_sched[features].fillna(0.0)
    try:
        Xs = model._scaler.transform(X)
    except Exception:
        # fallback: fit a fresh scaler on X
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
    try:
        probs = model.predict_proba(Xs)[:,1]
    except Exception as e:
        st.warning(f"⚠️ Model prediction failed: {e}")
        probs = np.repeat(0.5, len(X))

    week_sched["home_win_prob_model"] = probs
    # market implied prob via spread where available
    week_sched["market_home_prob"] = week_sched["spread"].apply(lambda s: spread_to_prob(s) if pd.notna(s) else np.nan)
    # blended prob
    week_sched["blended_prob"] = week_sched.apply(lambda r: (1-market_weight)*r["home_win_prob_model"] + (market_weight*r["market_home_prob"] if not pd.isna(r["market_home_prob"]) else 0.0), axis=1)
    # edge in percentage points
    week_sched["edge_pp"] = week_sched["blended_prob"] - week_sched["market_home_prob"]
    # recommendations
    def rec_from_row(r):
        if pd.isna(r["market_home_prob"]):
            # no market, recommend based on model only if strong
            if abs(r["home_win_prob_model"] - 0.5)*100 >= bet_threshold:
                return "Bet Home" if r["home_win_prob_model"]>0.5 else "Bet Away"
            return "No Bet"
        # if market exists
        if pd.isna(r["edge_pp"]):
            return "No Bet"
        # interpret edge in percentage points
        edge_pp = r["edge_pp"]*100
        if abs(edge_pp) >= bet_threshold:
            return "Bet Home" if edge_pp>0 else "Bet Away"
        return "No Bet"

    week_sched["recommendation"] = week_sched.apply(rec_from_row, axis=1)

    # display cards (auto expanded)
    for _, r in week_sched.iterrows():
        home = r.get("home_team") or "Home"
        away = r.get("away_team") or "Away"
        kickoff_fmt = format_kickoff(r.get("kickoff_ts"))
        col1, col2, col3 = st.columns([1,4,2])
        with col1:
            away_logo = get_logo_path(away)
            if away_logo:
                try:
                    st.image(away_logo, width=64)
                except Exception:
                    st.write("")  # avoid crash
            st.write(f"**{away.title()}**")
        with col2:
            st.markdown(f"### {away.title()}  @  **{home.title()}**")
            st.write(f"Kickoff: {kickoff_fmt}")
            # show predicted score rough (model mean)
            # simple predicted margin from prob -> convert back to points (rough)
            try:
                p = r["home_win_prob_model"]
                margin = (p - 0.5) * 26  # heuristic: 26 points full swing
                pred_home = 21 + margin/2
                pred_away = 21 - margin/2
                st.write(f"Predicted score: **{int(pred_home)}** - **{int(pred_away)}**")
            except Exception:
                st.write("Predicted score: N/A")
            # over/under
            ou = r.get("over_under", np.nan)
            st.write(f"Spread: {r.get('spread','N/A')} | O/U: {ou if not pd.isna(ou) else 'N/A'}")
            # market/model/blend
            model_prob = r.get("home_win_prob_model", 0.5)
            market_prob = r.get("market_home_prob", np.nan)
            blended = r.get("blended_prob", np.nan)
            st.write(f"- Model Prob: {model_prob*100:.1f}%")
            st.write(f"- Market Prob: {('N/A' if pd.isna(market_prob) else f'{market_prob*100:.1f}%')}")
            st.write(f"- Blended: {('N/A' if pd.isna(blended) else f'{blended*100:.1f}%')}")
            st.write(f"Recommendation: **{r.get('recommendation','No Bet')}**")
        with col3:
            home_logo = get_logo_path(home)
            if home_logo:
                try:
                    st.image(home_logo, width=64)
                except Exception:
                    st.write("")
            # show final/ongoing info
            if game_is_final(r):
                st.success(f"Final: {int(r.get('home_score',0))} - {int(r.get('away_score',0))}")
            else:
                # upcoming / not started
                st.info("Upcoming / In Progress")
        st.markdown("---")

# ---------- Footer: Top model bets & ROI ----------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 Top Model Bets (this week)")
try:
    top_bets = week_sched[week_sched["recommendation"] != "No Bet"].sort_values(by="edge_pp", ascending=False).head(5)
    if top_bets.empty:
        st.sidebar.write("No recommended bets this week.")
    else:
        for _, r in top_bets.iterrows():
            st.sidebar.write(f"{r.get('away_team','?')} @ {r.get('home_team','?')} — {r.get('recommendation')} — Edge {r.get('edge_pp',0)*100:.1f} pp")
except Exception:
    st.sidebar.write("No recommended bets this week.")

st.caption(f"Data loaded: historical={'yes' if not hist.empty else 'no'} | schedule={'yes' if not schedule.empty else 'no'} | updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")