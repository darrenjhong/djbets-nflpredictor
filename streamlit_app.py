# streamlit_app.py
"""
DJBets NFL Predictor - Streamlit app (updated)
Replace existing file with this one. Expects:
 - public/logos/<canonical_team_name_lower_underscored>.png (or .jpg)
 - data/nfl_archive_10Y.json (historical)
 - optional data/schedule.csv fallback
"""
import streamlit as st
import pandas as pd
import numpy as np
import traceback
from datetime import datetime
import logging

# local modules
from data_loader import load_or_fetch_schedule, load_historical, load_local_schedule, merge_schedule_and_history
from covers_odds import fetch_covers_for_week
from team_logo_map import canonical_from_string, lookup_logo
from utils import get_logo_path, compute_simple_elo, compute_roi, normalize_team_name

st.set_page_config(page_title="DJBets NFL Predictor", layout="wide")
logger = logging.getLogger("djbets.app")
logger.setLevel(logging.INFO)

THIS_YEAR = datetime.now().year

# ------------------
# Load data
# ------------------
@st.cache_data(ttl=600)
def load_data():
    hist = load_historical("data/nfl_archive_10Y.json")
    sched = load_or_fetch_schedule(season=THIS_YEAR, weeks=18)
    return hist, sched

hist, sched = load_data()
if sched is None:
    sched = pd.DataFrame()

# canonicalize schedule
sched = merge_schedule_and_history(sched, hist)

# compute weeks available
WEEKS = sorted(list(set(int(x) for x in sched["week"].dropna().astype(int))) ) if not sched.empty else list(range(1,19))

# -----------------------------
# Sidebar — Minimal UI
# -----------------------------
with st.sidebar:
    st.markdown("## 🏈 DJBets NFL")

    # --- WEEK SELECTOR ---
    current_week = st.number_input("Week", min_value=1, max_value=18, value=week, step=1)

    # --- MODEL SLIDERS (minimal style) ---
    st.markdown("### ⚙️ Model Settings")

    market_weight = st.slider(
        "📉 Market Weight",
        min_value=0.0, max_value=1.0, value=default_market_weight, step=0.05,
        help="Blend between the model and market lines."
    )

    bet_threshold = st.slider(
        "🎯 Bet Threshold (points)",
        min_value=0.0, max_value=15.0, value=default_threshold, step=0.5,
        help="Minimum required edge to trigger a bet."
    )

    # --- MODEL RECORD (simple, clean) ---
    if model_record is not None:
        wins, losses = model_record
        winrate = wins / max(1, wins + losses) * 100
        st.markdown(
            f"**📊 Model Record:** {wins}-{losses} ({winrate:.1f}%)"
        )
    else:
        st.markdown("📊 Model Record: *Not available*")

    st.markdown("---")

# ------------------
# Main view
# ------------------
st.title("🏈 DJBets — NFL Predictor")
st.caption(f"Season: {THIS_YEAR} — Week {week}")

# If schedule empty -> show help
if sched.empty:
    st.warning("No schedule loaded. Please add data/schedule.csv or check ESPN connectivity.")
    st.stop()

# filter for week
week_df = sched[sched["week"].astype(int) == int(week)].copy()
if week_df.empty:
    st.info("No games found for this week.")
    st.stop()

# fetch covers odds for week (best-effort)
covers_df = fetch_covers_for_week(THIS_YEAR, week)
if not covers_df.empty:
    # canonical names already in covers parser
    # create quick lookup by home-away canonical pair
    covers_map = {}
    for _, r in covers_df.iterrows():
        key = (r["home"], r["away"])
        covers_map[key] = {"spread": r.get("spread"), "over_under": r.get("over_under")}
else:
    covers_map = {}

# prepare display columns
cols = st.columns(1)
for idx, r in week_df.reset_index(drop=True).iterrows():
    home = r.get("home_team")
    away = r.get("away_team")
    status = r.get("status", "SCHEDULED")
    home_score = r.get("home_score")
    away_score = r.get("away_score")

    # logos
    home_logo = get_logo_path(home)
    away_logo = get_logo_path(away)

    # covers lookup
    cov = covers_map.get((home, away)) or covers_map.get((away, home)) or {}
    spread = cov.get("spread")
    ou = cov.get("over_under")

    # assemble card
    with st.expander(f"{away} @ {home} — {status}", expanded=True):
        rowc = st.columns([1,6,1])
        # away block (left)
        with rowc[0]:
            if away_logo:
                try:
                    st.image(away_logo, width=60)
                except Exception:
                    st.write(away)
            else:
                st.write(away)
        # center block: teams + model outputs
        with rowc[1]:
            st.markdown(f"**{away}**  @  **{home}**")
            # show scores if present
            if pd.notna(home_score) and pd.notna(away_score):
                st.write(f"Final: {away} {away_score} — {home} {home_score}")
            else:
                st.write("Kickoff: TBD / upcoming")

            # simple model features (we will simulate a fallback model probability)
            # In production, replace with model.predict_proba over feature vector.
            # For now: simple baseline from ELO difference if we can compute elos from history
            model_prob = 0.5
            try:
                # compute ELOs for teams based on history
                if not hist.empty:
                    # use small helper to compute elos df for historical games
                    elos = compute_simple_elo(hist)
                    # get latest elos for each team (approx)
                    h_elo = elos[elos["home"]==home]["elo_home"].dropna().iloc[-1] if any(elos["home"]==home) else 1500
                    a_elo = elos[elos["away"]==away]["elo_away"].dropna().iloc[-1] if any(elos["away"]==away) else 1500
                    elo_diff = h_elo - a_elo
                    # convert to prob
                    model_prob = 1.0 / (1 + 10 ** ((-elo_diff)/400.0))
                else:
                    model_prob = 0.5
            except Exception:
                model_prob = 0.5

            st.progress(min(max(model_prob, 0.0), 1.0), text=f"Model home win probability: {model_prob*100:.1f}%")

            # market probabilities via cover spread (convert spread to market prob crude)
            market_prob = None
            if spread is not None:
                # If spread is positive -> home favored by spread (positive means ???)
                # We'll assume spread is home - away (common). Convert to prob roughly:
                market_prob = 1.0 / (1 + 10 ** ((-spread)/12))  # heuristic
            blended_prob = None
            if market_prob is not None:
                blended_prob = market_weight * market_prob + (1 - market_weight) * model_prob

            # display market/spread/edge
            st.write(f"Spread (vegas): {spread if spread is not None else 'N/A'}")
            st.write(f"O/U: {ou if ou is not None else 'N/A'}")
            if market_prob is None:
                st.write(f"Market Probability: N/A")
            else:
                st.write(f"Market Probability: {market_prob*100:.1f}%")
            if blended_prob is None:
                st.write(f"Blended Probability: N/A")
                st.write("Edge: N/A")
                st.write("Recommendation: 🚫 No Bet")
            else:
                edge_pp = (blended_prob - market_prob) * 100
                st.write(f"Blended Probability: {blended_prob*100:.1f}%")
                st.write(f"Edge: {edge_pp:+.2f} pp")
                # recommend if abs(edge_pp) >= threshold
                if abs(edge_pp) >= bet_threshold_pp:
                    rec = "Bet Home" if blended_prob > market_prob else "Bet Away"
                    st.write(f"Recommendation: ✅ {rec} (edge {edge_pp:+.2f} pp)")
                else:
                    st.write("Recommendation: 🚫 No Bet (insufficient edge)")

        # right block
        with rowc[2]:
            if home_logo:
                try:
                    st.image(home_logo, width=60)
                except Exception:
                    st.write(home)
            else:
                st.write(home)

st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")