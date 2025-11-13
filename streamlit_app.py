# streamlit_app.py
"""
Main Streamlit app for DJBets NFL Predictor — version: modular v9.5 -> updated
This file is intended to be a fully-contained Streamlit entrypoint.

Expectations:
- data/ contains schedule.csv and nfl_archive_10Y.json optionally
- public/logos/ contains team logo files (png)
- your Odds API key is optional and not required for Option A (Covers scraping).
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

# local modules
from data_loader import load_or_fetch_schedule, load_historical, load_local_schedule
from covers_odds import fetch_covers_for_week
from team_logo_map import lookup_logo
from model import train_model, load_model, predict
from utils import compute_simple_elo, compute_roi

st.set_page_config(page_title="DJBets NFL Predictor", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
def safe_read_schedule(season:int):
    try:
        df = load_or_fetch_schedule(season)
        if df is None:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

def ensure_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

# ---------------------------
# Load data
# ---------------------------

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Sidebar: season + week selection
THIS_YEAR = int(os.getenv("SEASON_YEAR", datetime.now().year))
st.sidebar.markdown("## 🏈 DJBets — NFL Predictor")
# keep the dropdown style you asked for
season = st.sidebar.selectbox("Season", [THIS_YEAR, THIS_YEAR-1, THIS_YEAR-2], index=0, key="season")
# We'll populate weeks after loading schedule
# Placeholders
MAX_WEEKS = 18

# Load historical
with st.spinner("Loading historical data..."):
    hist = load_historical = None
    try:
        hist = load_historical()
        if hist is None or hist.empty:
            st.info("⚠️ No local historical file found in data/. The app will attempt to fetch scraped historical sources if available, otherwise will use simulated training fallback.")
            hist = pd.DataFrame()
        else:
            st.sidebar.write(f"Historical rows: {len(hist)}")
    except Exception:
        hist = pd.DataFrame()
        st.sidebar.write("Historical load failed.")

# Load schedule
with st.spinner("Loading schedule..."):
    sched = safe_read_schedule(season)
    if sched is None or sched.empty:
        st.warning("⚠️ No schedule loaded from local file or ESPN for this season.")
        sched = pd.DataFrame()

# Normalize schedule columns
for c in ["week","home_team","away_team","start_time","home_score","away_score","game_id"]:
    if c not in sched.columns:
        sched[c] = np.nan

# compute weeks list
weeks = sorted([int(w) for w in pd.unique(sched["week"].dropna().astype(int))]) if not sched.empty else list(range(1, MAX_WEEKS+1))
if not weeks:
    weeks = list(range(1, MAX_WEEKS+1))

week = st.sidebar.selectbox("Week", weeks, index=0, key="week")

# ---------------------------
# Model training (auto-train on first launch)
# ---------------------------
with st.spinner("Training model (or loading cached model)..."):
    model, feature_cols = load_model()
    trained_here = False
    if model is None:
        try:
            if not hist.empty:
                # compute Elo if scores available
                if "home_score" in hist.columns and "away_score" in hist.columns:
                    hist = compute_simple_elo(hist)
                model, feature_cols = train_model(hist)
            if model is None:
                # fallback: simulated dataset training for a tiny logistic regression
                sim = pd.DataFrame({
                    "elo_diff": np.random.normal(0,50,200),
                    "inj_diff": np.random.normal(0,3,200),
                    "temp_c": np.random.normal(10,5,200),
                    "spread": np.random.normal(0,4,200),
                    "over_under": np.random.normal(45,4,200),
                    "home_win": (np.random.random(200) > 0.5).astype(int)
                })
                model, feature_cols = train_model(sim)
            trained_here = True
        except Exception as e:
            st.error(f"Model training failed: {str(e)}")
            model = None
            feature_cols = ["elo_diff","inj_diff","temp_c","spread","over_under"]

# ---------------------------
# Pull odds (Covers) for week
# ---------------------------
with st.spinner("Fetching odds for week (Covers)..."):
    try:
        covers_df = fetch_covers_for_week(season, week)
    except Exception:
        covers_df = pd.DataFrame()

# Normalize schedule for display: merge covers odds where possible
display_sched = sched.copy()
display_sched["week"] = display_sched["week"].astype(float).fillna(0).astype(int)
if not covers_df.empty:
    # try to match teams by name substring (best-effort)
    def find_odds(row):
        for _, r in covers_df.iterrows():
            try:
                if str(r["home"]).lower() in str(row["home_team"]).lower() or str(row["home_team"]).lower() in str(r["home"]).lower():
                    return r.get("spread"), r.get("over_under")
            except Exception:
                continue
        return (np.nan, np.nan)
    s_spread = []
    s_ou = []
    for _, r in display_sched.iterrows():
        sp, ou = find_odds(r)
        s_spread.append(sp)
        s_ou.append(ou)
    display_sched["spread"] = s_spread
    display_sched["over_under"] = s_ou
else:
    display_sched["spread"] = np.nan
    display_sched["over_under"] = np.nan

# Ensure features present for prediction
required_features = feature_cols
for c in required_features:
    if c not in display_sched.columns:
        display_sched[c] = 0.0

# If historical Elo exists, try to bring last Elo to upcoming games by averaging team elos
if not hist.empty and "home_elo_pre" in hist.columns and "away_elo_pre" in hist.columns:
    # compute latest elo per team
    last_home = hist.groupby("home_team")["home_elo_pre"].last().to_dict()
    last_away = hist.groupby("away_team")["away_elo_pre"].last().to_dict()
    def get_elo_diff(r):
        h = r.get("home_team")
        a = r.get("away_team")
        he = last_home.get(h, 1500) if h in last_home else last_away.get(h, 1500)
        ae = last_away.get(a, 1500) if a in last_away else last_home.get(a,1500)
        return he - ae
    display_sched["elo_diff"] = display_sched.apply(get_elo_diff, axis=1)
else:
    # fallback to 0
    display_sched["elo_diff"] = display_sched.get("elo_diff",0.0).fillna(0.0)

# injuries/weather placeholders
display_sched["inj_diff"] = display_sched.get("inj_diff",0.0).fillna(0.0)
display_sched["temp_c"] = display_sched.get("temp_c",0.0).fillna(0.0)

# Filter to selected week
week_df = display_sched[display_sched["week"]==int(week)].copy() if not display_sched.empty else pd.DataFrame()
if week_df.empty:
    st.warning("No games found for this week.")
else:
    # prepare features
    X = week_df[required_features].astype(float).fillna(0.0)
    try:
        probs = predict(model, required_features, X)
        week_df["home_win_prob_model"] = probs
        week_df["predicted_winner"] = week_df.apply(lambda r: r["home_team"] if r["home_win_prob_model"]>=0.5 else r["away_team"], axis=1)
    except Exception as e:
        # feature mismatch — try to add missing columns
        for c in required_features:
            if c not in week_df.columns:
                week_df[c] = 0.0
        try:
            probs = predict(model, required_features, week_df[required_features])
            week_df["home_win_prob_model"] = probs
            week_df["predicted_winner"] = week_df.apply(lambda r: r["home_team"] if r["home_win_prob_model"]>=0.5 else r["away_team"], axis=1)
        except Exception as e2:
            week_df["home_win_prob_model"] = 0.5
            week_df["predicted_winner"] = week_df["home_team"]

# ---------------------------
# UI: Main area
# ---------------------------
st.markdown("<h1 style='text-align:left'>🏈 DJBets — NFL Predictor</h1>", unsafe_allow_html=True)
st.write(f"Season {season} — Week {week}")

# Sidebar widgets: model sliders and record info (preserve previous design)
with st.sidebar.expander("Model Controls & Record", expanded=True):
    st.markdown("### Model controls")
    market_weight = st.slider("Market weight (blend model<>market)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    bet_threshold = st.slider("Bet threshold (edge pp)", min_value=0.0, max_value=15.0, value=3.0, step=0.5)
    st.markdown("---")
    # model record
    if not hist.empty:
        try:
            hist2 = hist.copy()
            if "home_score" in hist2.columns and "away_score" in hist2.columns and "home_win_prob_model" in hist2.columns:
                hist2["pred"] = (hist2["home_win_prob_model"] >= 0.5).astype(int)
                hist2["actual"] = (hist2["home_score"] > hist2["away_score"]).astype(int)
                correct = int((hist2["pred"]==hist2["actual"]).sum())
                total = hist2.shape[0]
                st.markdown(f"**Model record:** {correct}/{total} = {correct/total:.1%}")
            else:
                st.write("Model track: not enough labeled history.")
        except Exception:
            st.write("Model track: N/A")
    else:
        st.write("Model track: no historical file found.")

# Main content: game cards
if not week_df.empty:
    for idx, row in week_df.iterrows():
        home = row.get("home_team","HOME")
        away = row.get("away_team","AWAY")
        home_logo = lookup_logo(home)
        away_logo = lookup_logo(away)

        col1, col2, col3 = st.columns([1,4,2])
        with col1:
            # away left, home right style with '@'
            try:
                if os.path.exists(away_logo):
                    st.image(away_logo, width=64)
                else:
                    st.write(away)
            except Exception:
                st.write(away)
        with col2:
            st.markdown(f"### {away}  @  {home}")
            # kickoff formatting
            kickoff = row.get("start_time")
            if pd.notna(kickoff):
                try:
                    kd = pd.to_datetime(kickoff)
                    st.write(f"Kickoff: {kd.strftime('%a %b %d %Y %H:%M %Z') if hasattr(kd,'tzinfo') else kd.strftime('%a %b %d %Y %H:%M')}")
                except Exception:
                    st.write(f"Kickoff: {kickoff}")
            # model probability
            p = row.get("home_win_prob_model", 0.5)
            if pd.isna(p):
                p = 0.5
            st.progress(float(p), text=f"Home Win Probability: {float(p)*100:.1f}%")
            # show predicted winner
            st.write(f"**Predicted winner:** {row.get('predicted_winner','TBD')}")
        with col3:
            # odds
            sp = row.get("spread", np.nan)
            ou = row.get("over_under", np.nan)
            st.write(f"Spread (vegas): {sp if not pd.isna(sp) else 'N/A'}")
            st.write(f"O/U: {ou if not pd.isna(ou) else 'N/A'}")
            # recommendation simple edge calc: model_prob - market_prob (market prob from spread)
            rec = "🚫 No Bet"
            if not pd.isna(sp):
                try:
                    # convert spread to market probability (approx)
                    # simple logistic approximation: p = 1 / (1 + 10^(spread/12))
                    market_prob_home = 1.0 / (1.0 + 10 ** (float(sp)/12.0))
                    blended = (1-market_weight)*float(p) + market_weight*market_prob_home
                    edge_pp = (blended - market_prob_home) * 100
                    if abs(edge_pp) >= bet_threshold:
                        rec = "🛫 Bet Home" if blended>market_prob_home else "🛫 Bet Away"
                    st.write(f"Edge: {edge_pp:+.1f} pp  | Market Prob: {market_prob_home*100:.1f}% | Blended: {blended*100:.1f}%")
                except Exception:
                    st.write("Edge: N/A")
            else:
                st.write("Edge: N/A")
            st.write(rec)
        st.markdown("---")

# Model bets & ROI
try:
    pnl, bets_made, roi = compute_roi(hist)
    st.sidebar.markdown("### Performance")
    st.sidebar.write(f"📈 PNL: {pnl:.2f}")
    st.sidebar.write(f"🎯 Bets made: {bets_made}")
    st.sidebar.write(f"💵 ROI per bet: {roi:.3f}")
except Exception:
    st.sidebar.write("Performance: N/A")

st.caption(f"Updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")