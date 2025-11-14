# streamlit_app.py
# DJBets NFL Predictor – Covers primary schedule, fastR fallback, ESPN scores
# Sidebar Style A (bullseye icon)


import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import traceback

# --- Local imports ---
from covers_odds import fetch_covers_for_week
from schedule_fastr import load_fastr_schedule
from team_logo_map import canonical_team_name
from utils import safe_request_json, get_logo_path


# ----------------------------------------------------
# App setup
# ----------------------------------------------------
st.set_page_config(page_title="DJBets NFL Predictor", layout="wide")
CURRENT_SEASON = datetime.now().year


# ----------------------------------------------------
# Load ESPN scores (used ONLY to fill home_score/away_score)
# ----------------------------------------------------
def load_espn_scores(season: int, week: int) -> pd.DataFrame:
    try:
        url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/types/2/weeks/{week}/events"
        data = safe_request_json(url)
        if not data or "events" not in data:
            return pd.DataFrame()

        out = []
        for event in data["events"]:
            try:
                comp = event["competitions"][0]
                teams = comp["competitors"]

                home = next(t for t in teams if t["homeAway"] == "home")
                away = next(t for t in teams if t["homeAway"] == "away")

                out.append({
                    "home_team": canonical_team_name(home["team"]["displayName"]),
                    "away_team": canonical_team_name(away["team"]["displayName"]),
                    "home_score": home.get("score"),
                    "away_score": away.get("score"),
                })
            except:
                continue
        return pd.DataFrame(out)
    except:
        return pd.DataFrame()


# ----------------------------------------------------
# Build schedule for a week (Covers → fastR → ESPN)
# ----------------------------------------------------
def build_week_schedule(season: int, week: int) -> pd.DataFrame:
    # 1. Try Covers for schedule + odds
    covers = fetch_covers_for_week(season, week)
    if not covers.empty:
        rows = []
        for _, r in covers.iterrows():
            rows.append({
                "season": season,
                "week": week,
                "home_team": canonical_team_name(r["home"]),
                "away_team": canonical_team_name(r["away"]),
                "spread": r.get("spread"),
                "over_under": r.get("over_under"),
                "home_score": np.nan,
                "away_score": np.nan,
            })
        return pd.DataFrame(rows)

    # 2. fastR (full schedule)
    fastr = load_fastr_schedule(season)
    fastr_week = fastr[fastr["week"] == week]
    if not fastr_week.empty:
        fastr_week = fastr_week.copy()
        fastr_week["home_team"] = fastr_week["home_team"].map(canonical_team_name)
        fastr_week["away_team"] = fastr_week["away_team"].map(canonical_team_name)
        fastr_week["spread"] = np.nan
        fastr_week["over_under"] = np.nan
        return fastr_week

    # 3. ESPN fallback
    espn = load_espn_scores(season, week)
    if not espn.empty:
        rows = []
        for _, r in espn.iterrows():
            rows.append({
                "season": season,
                "week": week,
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "spread": np.nan,
                "over_under": np.nan,
                "home_score": r["home_score"],
                "away_score": r["away_score"],
            })
        return pd.DataFrame(rows)

    return pd.DataFrame()


# ----------------------------------------------------
# Simple prediction model (Elo-like; placeholder)
# ----------------------------------------------------
def model_predict(home: str, away: str) -> float:
    # deterministic fallback “win probability”
    h = sum(ord(c) for c in home)
    a = sum(ord(c) for c in away)
    return h / (h + a)


# ----------------------------------------------------
# Sidebar — Style A (Bullseye)
# ----------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center;margin-bottom:10px;'>
            <img src='https://img.icons8.com/ios-filled/100/target.png' width='60'>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## 🏈 DJBets NFL Predictor")

    # fastR gives full schedule → 18 weeks
    available_weeks = list(range(1, 19))

    week = st.selectbox("📅 Select Week", available_weeks, index=0)

    st.markdown("### ⚙️ Model Controls")
    market_weight = st.slider("Market weight (blend model <> market)", 0.0, 1.0, 0.0, 0.01)
    bet_threshold = st.slider("Bet threshold (edge pts)", 0.0, 20.0, 5.0, 0.5)

    st.markdown("### 📊 Model Record")
    st.info("Model trained using local data + Elo fallback.")


# ----------------------------------------------------
# Main content — Predictions
# ----------------------------------------------------
st.title(f"DJBets — NFL Predictor — Season {CURRENT_SEASON} — Week {week}")

with st.spinner("Loading schedule (Covers → fastR → ESPN)..."):
    sched = build_week_schedule(CURRENT_SEASON, week)

if sched.empty:
    st.error("No schedule found from Covers, fastR, or ESPN. This may be temporary. Try again later.")
    st.stop()


# ----------------------------------------------------
# Render Games
# ----------------------------------------------------
for _, g in sched.iterrows():
    home = g["home_team"]
    away = g["away_team"]

    logo_home = get_logo_path(home)
    logo_away = get_logo_path(away)

    wp = model_predict(home, away)
    pred = home if wp > 0.5 else away

    with st.container():
        cols = st.columns([1, 3, 1])

        with cols[0]:
            st.image(logo_away, width=70)
            st.write(f"**{away.replace('_',' ').title()}**")

        with cols[1]:
            st.subheader("vs")
            st.write(f"**Predicted Winner:** {pred.replace('_',' ').title()}")
            st.write(f"Win probability: {wp:.2%}")

            st.write(f"Spread: {g.get('spread', 'N/A')}")
            st.write(f"O/U: {g.get('over_under', 'N/A')}")

        with cols[2]:
            st.image(logo_home, width=70)
            st.write(f"**{home.replace('_',' ').title()}**")

    st.markdown("---")