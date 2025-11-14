# streamlit_app.py
"""
Main Streamlit UI for DJBets NFL Predictor
- Week selector at top of sidebar
- Minimal sidebar layout with icons
- Games auto-expanded; logos centered; '@' for home indicator
- Uses local schedule/historical first; falls back to ESPN/Covers scraping
"""

import os
import sys
import traceback
from datetime import datetime
import math

import pandas as pd
import numpy as np
import streamlit as st

# local modules
from data_loader import load_or_fetch_schedule, load_historical, prepare_week_schedule
from covers_odds import fetch_covers_for_week
from team_logo_map import canonical_team_name, get_logo_path
from model import build_model_from_history, predict_game_row
from utils import safe_request_json

# page config
st.set_page_config(page_title="DJBets NFL Predictor", layout="wide")

CURRENT_SEASON = datetime.now().year
MAX_WEEKS = 18

# -------------------------
# Sidebar (minimal with icons)
# -------------------------
with st.sidebar:
    st.markdown("🏈 **DJBets NFL Predictor**", unsafe_allow_html=True)
    st.write("")
    # Week picker at top (single selection)
    schedule_df = load_or_fetch_schedule(CURRENT_SEASON)
    if schedule_df is not None and not schedule_df.empty and "week" in schedule_df.columns:
        available_weeks = sorted([int(x) for x in schedule_df["week"].dropna().unique().tolist()])
        default_week = available_weeks[0] if available_weeks else 1
    else:
        available_weeks = list(range(1, MAX_WEEKS + 1))
        default_week = 1

    week = st.selectbox("📅 Week", options=available_weeks, index=max(0, available_weeks.index(default_week)) if default_week in available_weeks else 0)

    st.markdown("---")
    st.markdown("⚙️ **Model Controls**")
    market_weight = st.slider("Market weight (blend model <> market)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    bet_threshold = st.slider("Bet threshold (edge pts)", min_value=0.0, max_value=20.0, value=3.0, step=0.5)

    st.markdown("---")
    st.markdown("📊 **Model Record**")
    # placeholder trackers - filled later
    model_record_col = st.empty()

    st.markdown("---")

# -------------------------
# Load historical & model
# -------------------------
hist = load_historical()  # local archive if present
model_info = build_model_from_history(hist)  # returns trained object or fallback data structure

# Show model status in sidebar
if model_info.get("trained", False):
    model_record_col.markdown(f"✅ Trained model available — {model_info.get('notes','')}")
else:
    model_record_col.markdown(f"⚠️ No trained model available — Elo fallback active.")

# -------------------------
# Prepare week schedule (merge spreads from Covers/ESPN/local)
# -------------------------
# Try to fetch covers odds for the week (best-effort)
covers_df = fetch_covers_for_week(CURRENT_SEASON, int(week)) if "fetch_covers_for_week" in globals() else pd.DataFrame()
week_sched = prepare_week_schedule(CURRENT_SEASON, int(week), schedule_df=schedule_df, covers_df=covers_df)

st.markdown(f"# DJBets — NFL Predictor — Season {CURRENT_SEASON} — Week {week}")

if week_sched.empty:
    st.warning("No schedule entries for this week found in schedule sources (local/ESPN/Covers).")
    st.info("Place `data/schedule.csv` or ensure ESPN/Covers are reachable.")
    st.stop()

# -------------------------
# Render each game card (auto-expanded)
# -------------------------
for idx, r in week_sched.iterrows():
    home_name = r.get("home_team", "Unknown")
    away_name = r.get("away_team", "Unknown")
    status = r.get("status", "upcoming")
    vegas_spread = r.get("spread", np.nan)
    vegas_ou = r.get("over_under", np.nan)

    # Predict with model (uses model_info and simple fallback)
    pred = predict_game_row(r, model_info)
    model_prob_home = pred.get("home_prob", None)
    model_spread = pred.get("model_spread", None)
    pred_home_score = pred.get("pred_home_score", None)
    pred_away_score = pred.get("pred_away_score", None)
    pred_total = None
    if pred_home_score is not None and pred_away_score is not None:
        pred_total = round(pred_home_score + pred_away_score, 1)

    # Recommendation logic (blend with market weight). If market (vegas_spread) missing -> rely on model only
    rec = "No Bet"
    edge_pp = None
    if model_spread is not None:
        if not math.isnan(vegas_spread):
            # edge = model_spread - vegas_spread (model spread positive => home favored)
            edge_pp = model_spread - vegas_spread
            blended_edge = (1 - market_weight) * edge_pp
            if abs(blended_edge) >= bet_threshold:
                rec = "Bet Home" if blended_edge > 0 else "Bet Away"
        else:
            # no market; use model threshold on absolute model spread
            if abs(model_spread) >= bet_threshold:
                rec = "Bet Home" if model_spread > 0 else "Bet Away"

    # Layout: columns for away logo/name / center '@' / home logo/name
    with st.expander(f"{away_name} @ {home_name} | {status}", expanded=True):
        c1, c2, c3 = st.columns([1, 0.2, 1])
        # away
        with c1:
            away_logo_path = get_logo_path(canonical_team_name(away_name))
            if away_logo_path and os.path.exists(away_logo_path):
                st.image(away_logo_path, width=110)
            else:
                st.write("")  # placeholder
            st.markdown(f"**{away_name}**")
        # center
        with c2:
            st.markdown("<h3 style='text-align:center;'>@</h3>", unsafe_allow_html=True)
        # home
        with c3:
            home_logo_path = get_logo_path(canonical_team_name(home_name))
            if home_logo_path and os.path.exists(home_logo_path):
                st.image(home_logo_path, width=110)
            else:
                st.write("")
            st.markdown(f"**{home_name}**")

        # Info lines
        st.markdown("")
        spread_str = f"{vegas_spread:+.1f}" if (vegas_spread is not None and not (isinstance(vegas_spread, float) and math.isnan(vegas_spread))) else "N/A"
        ou_str = f"{vegas_ou:.1f}" if (vegas_ou is not None and not (isinstance(vegas_ou, float) and math.isnan(vegas_ou))) else "N/A"

        st.markdown(f"**Vegas Spread:** {spread_str}  |  **O/U:** {ou_str}")
        if model_prob_home is not None:
            st.markdown(f"**Model Home Win Probability:** {model_prob_home*100:.1f}%")
        if model_spread is not None:
            st.markdown(f"**Model Spread Prediction:** {model_spread:+.1f}")
        if pred_total is not None:
            st.markdown(f"**Predicted Total Points:** {pred_total:.1f}")
            st.markdown(f"**Predicted Score:** {home_name} {pred_home_score:.1f} — {pred_away_score:.1f} {away_name}")

        edge_txt = f"{edge_pp:+.1f} pp" if edge_pp is not None else "N/A"
        st.markdown(f"**Edge vs Market:** {edge_txt}")
        st.markdown(f"**Recommendation:** {rec}")

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — Schedule source: local/ESPN/Covers (fallbacks)")