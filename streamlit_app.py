# streamlit_app.py
import os
import io
import time
from datetime import datetime
import traceback

import pandas as pd
import numpy as np
import streamlit as st

from data_loader import (
    load_historical,
    load_local_schedule,
    fetch_espn_week,
    fetch_espn_season,
    load_or_fetch_schedule,
)
from covers_odds import fetch_covers_for_week
from team_logo_map import canonical_team_name
from model import train_or_load_model, predict_row, has_trained_model
from utils import (
    get_logo_path,
    compute_simple_elo,
    safe_request_json,
    compute_roi,
)

# --- Page config ---
st.set_page_config(page_title="DJBets — NFL Predictor", layout="wide")

# Constants
CURRENT_SEASON = datetime.now().year
MAX_WEEKS = 18

DATA_DIR = "data"
LOGOS_DIR = "public/logos"

# Ensure data dir exists
os.makedirs(DATA_DIR, exist_ok=True)


# ----------------------
# Load data (cached)
# ----------------------
@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_historical_cached(path=os.path.join(DATA_DIR, "nfl_archive_10Y.json")):
    return load_historical(path)


@st.cache_data(ttl=60 * 10, show_spinner=False)
def load_schedule_cached(season=CURRENT_SEASON):
    return load_or_fetch_schedule(season)


# ----------------------
# UI - Sidebar
# ----------------------
with st.sidebar:
    st.markdown("## 🏈 DJBets NFL Predictor")
    # week selector at top (dropdown)
    schedule_df = load_schedule_cached(CURRENT_SEASON)
    if not schedule_df.empty and "week" in schedule_df.columns:
        available_weeks = sorted(int(w) for w in schedule_df["week"].dropna().unique().tolist())
        week = st.selectbox("📅 Week", options=available_weeks, index=0)
    else:
        week = st.selectbox("📅 Week", options=list(range(1, MAX_WEEKS + 1)), index=0)

    st.markdown("### ⚙️ Model Controls")
    market_weight = st.slider("Market weight (blend model <> market)", 0.0, 1.0, 0.0, 0.05)
    bet_threshold = st.slider("Bet threshold (edge pts)", 0.0, 20.0, 8.0, 0.5)

    st.markdown("### 📊 Model Record")
    hist = load_historical_cached()
    model_trained = has_trained_model()
    if model_trained:
        st.success("Trained model available")
    else:
        st.info("No trained model available — Elo fallback active.")

    st.markdown("---")
    st.caption("Drop files into /data to override sources (schedule.csv, nfl_archive_10Y.json)")

# ----------------------
# Main
# ----------------------
st.header(f"DJBets — NFL Predictor — Season {CURRENT_SEASON} — Week {week}")

# load history & schedule
hist_df = load_historical_cached()
schedule_df = load_schedule_cached(CURRENT_SEASON)

# prepare week schedule: prefer ESPN schedule rows for the chosen week
def prepare_week_schedule(schedule_df, wk):
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame()

    df = schedule_df.copy()
    # Normalize columns
    for c in ["home_team", "away_team", "week", "season"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    # ensure numeric week
    if "week" in df.columns:
        try:
            df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int)
        except Exception:
            pass

    week_df = df[df["week"] == int(wk)].copy()
    if week_df.empty:
        return pd.DataFrame()

    # canonicalize team names for logo lookup and matching
    for col in ["home_team", "away_team"]:
        if col in week_df.columns:
            week_df[col] = week_df[col].astype(str).apply(lambda s: canonical_team_name(s.lower()))
    return week_df

week_sched = prepare_week_schedule(schedule_df, week)

# If no games for the week from ESPN/local, attempt to fetch via ESPN API for that week (one-shot)
if week_sched.empty:
    st.warning("No schedule entries for this week found in schedule sources. Attempting ESPN fetch for week...")
    try:
        espn_w = fetch_espn_week(CURRENT_SEASON, int(week))
        if not espn_w.empty:
            # canonicalize
            espn_w["home_team"] = espn_w["home_team"].apply(lambda s: canonical_team_name(str(s).lower()))
            espn_w["away_team"] = espn_w["away_team"].apply(lambda s: canonical_team_name(str(s).lower()))
            week_sched = espn_w
            st.success(f"Loaded {len(week_sched)} games from ESPN for week {week}")
        else:
            st.warning("ESPN returned no games for that week.")
    except Exception as e:
        st.error("ESPN fetch failed (network or API).")
        st.write(e)

# If still empty, try Covers to build simple matchups
if week_sched.empty:
    st.info("Trying Covers matchup scraping to build schedule...")
    try:
        covers_df = fetch_covers_for_week(CURRENT_SEASON, int(week))
        if not covers_df.empty:
            covers_df["home_team"] = covers_df["home"].apply(lambda s: canonical_team_name(str(s).lower()))
            covers_df["away_team"] = covers_df["away"].apply(lambda s: canonical_team_name(str(s).lower()))
            # create minimal week_sched
            week_sched = pd.DataFrame({
                "home_team": covers_df["home_team"],
                "away_team": covers_df["away_team"],
                "season": CURRENT_SEASON,
                "week": int(week),
                "spread": covers_df.get("spread"),
                "over_under": covers_df.get("over_under"),
            })
            st.success(f"Constructed schedule with {len(week_sched)} games from Covers")
        else:
            st.warning("Covers did not return matchups either.")
    except Exception as e:
        st.error("Covers scraping failed.")
        st.write(e)

# final guard
if week_sched.empty:
    st.error("No games found for this week. Ensure schedule.csv or data is present, or ESPN/Covers are reachable.")
    st.stop()

# ------------------------------------------------------------
# Enrich week dataframe: logos, spreads/OU from Covers if missing,
# compute simple ELO if history present, then predict
# ------------------------------------------------------------
# add logo paths
def ensure_logo_cols(df):
    df = df.copy()
    df["home_logo"] = df["home_team"].apply(lambda s: get_logo_path(s))
    df["away_logo"] = df["away_team"].apply(lambda s: get_logo_path(s))
    return df

week_sched = ensure_logo_cols(week_sched)

# fill spreads/over_under if missing via Covers (already attempted above)
if "spread" not in week_sched.columns or week_sched["spread"].isna().all():
    try:
        cov = fetch_covers_for_week(CURRENT_SEASON, int(week))
        if not cov.empty:
            # map cov rows into week_sched by team pairs
            for i, r in cov.iterrows():
                h = canonical_team_name(str(r["home"]).lower())
                a = canonical_team_name(str(r["away"]).lower())
                mask = (week_sched["home_team"] == h) & (week_sched["away_team"] == a)
                if mask.any():
                    idx = week_sched[mask].index[0]
                    week_sched.at[idx, "spread"] = r.get("spread")
                    week_sched.at[idx, "over_under"] = r.get("over_under")
    except Exception:
        pass

# compute Elo columns using history (simple team Elo from historical results)
if hist_df is not None and not hist_df.empty:
    try:
        elo_map = compute_simple_elo(hist_df)
        # attach elo_home / elo_away
        week_sched["elo_home"] = week_sched["home_team"].map(lambda t: elo_map.get(t, 1500))
        week_sched["elo_away"] = week_sched["away_team"].map(lambda t: elo_map.get(t, 1500))
        week_sched["elo_diff"] = week_sched["elo_home"] - week_sched["elo_away"]
    except Exception:
        week_sched["elo_home"] = 1500
        week_sched["elo_away"] = 1500
        week_sched["elo_diff"] = 0
else:
    week_sched["elo_home"] = 1500
    week_sched["elo_away"] = 1500
    week_sched["elo_diff"] = 0

# ensure numeric columns
for c in ["spread", "over_under"]:
    if c in week_sched.columns:
        week_sched[c] = pd.to_numeric(week_sched[c], errors="coerce")
    else:
        week_sched[c] = np.nan

# ----------------------
# Train or load model
# ----------------------
with st.spinner("Training/loading model..."):
    model, features = train_or_load_model(hist_df)

# ----------------------
# Predict per game
# ----------------------
pred_rows = []
for idx, row in week_sched.reset_index().iterrows():
    # prepare feature vector; use available features and fallbacks
    Xrow = {}
    # features we expect: elo_diff, spread, over_under
    Xrow["elo_diff"] = float(row.get("elo_diff", 0))
    Xrow["spread"] = float(row.get("spread")) if pd.notna(row.get("spread")) else np.nan
    Xrow["over_under"] = float(row.get("over_under")) if pd.notna(row.get("over_under")) else np.nan

    # predict
    try:
        prob, pred_home_pts, pred_away_pts = predict_row(model, Xrow)
    except Exception:
        prob = None
        pred_home_pts = None
        pred_away_pts = None

    # compute market prob (from spread -> implied prob) if spread available
    market_prob = None
    if pd.notna(row.get("spread")):
        # naive conversion: smaller spread -> closer to 50%; map spread to prob via logistic
        s = float(row["spread"])
        market_prob = 1 / (1 + np.exp(-(-s) / 3.0))  # heuristic

    # blended probability
    if prob is None:
        blended = market_prob
    elif market_prob is None:
        blended = prob
    else:
        blended = (1 - market_weight) * prob + market_weight * market_prob

    # edge: model - market (in percentage points)
    edge_pp = None
    if prob is not None and market_prob is not None:
        edge_pp = (prob - market_prob) * 100

    # recommendation
    rec = "🚫 No Bet"
    if edge_pp is not None and abs(edge_pp) >= bet_threshold:
        if edge_pp > 0:
            rec = "🛫 Bet Home (spread)"
        else:
            rec = "🛫 Bet Away (spread)"

    pred_rows.append(
        {
            "idx": idx,
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_logo": row["home_logo"],
            "away_logo": row["away_logo"],
            "prob_home": prob,
            "market_prob": market_prob,
            "blended": blended,
            "edge_pp": edge_pp,
            "recommendation": rec,
            "pred_home_pts": pred_home_pts,
            "pred_away_pts": pred_away_pts,
            "spread": row.get("spread"),
            "over_under": row.get("over_under"),
        }
    )

pred_df = pd.DataFrame(pred_rows)

# ----------------------
# Sidebar: show simple ROI/model record (light)
# ----------------------
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🎯 Model Snapshot")
    # compute quick model record/ROI if history exists and model trained
    try:
        if hist_df is not None and not hist_df.empty and model is not None:
            pnl, bets_made, roi = compute_roi(hist_df, model)
            st.metric("ROI", f"{roi:.2f}%")
            st.metric("Bets Made", f"{int(bets_made)}")
        else:
            st.write("Historical record unavailable")
    except Exception:
        st.write("Model performance unavailable")

# ----------------------
# Main UI: list games
# ----------------------
cols = st.columns(1)
for i, r in pred_df.iterrows():
    card = st.container()
    with card:
        st.markdown("---")
        c1, c2, c3 = st.columns([1, 5, 2])
        with c1:
            # away logo left, home logo right
            try:
                if r["away_logo"]:
                    st.image(r["away_logo"], width=56)
            except Exception:
                pass
        with c2:
            # matchup line: "away @ home" (home right)
            st.markdown(f"**{r['away_team'].replace('_',' ').title()}  @  {r['home_team'].replace('_',' ').title()}**")
            # probability
            prob_txt = "N/A"
            if r["prob_home"] is not None:
                prob_txt = f"{r['prob_home']*100:.1f}%"
            st.write(f"Home Win Probability: **{prob_txt}**")

            # predicted score if available
            if r["pred_home_pts"] is not None and r["pred_away_pts"] is not None:
                st.write(f"Predicted score: **{r['home_team'].split('_')[-1].title()} {r['pred_home_pts']:.1f} - {r['away_team'].split('_')[-1].title()} {r['pred_away_pts']:.1f}**")

            # spread / ou
            s_txt = "N/A" if pd.isna(r["spread"]) else f"{r['spread']:+.1f}"
            ou_txt = "N/A" if pd.isna(r["over_under"]) else f"{r['over_under']:.1f}"
            st.write(f"Spread (vegas): **{s_txt}** | O/U: **{ou_txt}**")

            # edge / recommendation
            edge_txt = "N/A" if r["edge_pp"] is None else f"{r['edge_pp']:+.1f} pp"
            st.write(f"Edge vs market: **{edge_txt}**")
            st.write(f"Recommendation: **{r['recommendation']}**")

        with c3:
            try:
                if r["home_logo"]:
                    st.image(r["home_logo"], width=64)
            except Exception:
                pass

# footer
st.caption(f"Data sources: ESPN (schedule) + Covers (odds). Local historical archive used for training. Last update {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
