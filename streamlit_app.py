# streamlit_app.py — DJBets (Covers → fastR backup → ESPN scores)
# --------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import traceback

# Local modules
from covers_odds import fetch_covers_for_week
from schedule_fastr import load_fastr_schedule
from team_logo_map import canonical_team_name
from utils import safe_request_json

CURRENT_SEASON = datetime.now().year

st.set_page_config(page_title="DJBets Predictor", layout="wide")
st.title("🏈 DJBets — NFL Predictor")

# -------------------------------------------------------------------
# Sidebar (UI Style A — bullseye top, clean simple controls)
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; margin-bottom:10px;'>
        <img src='https://img.icons8.com/ios-filled/100/target.png' width='55'>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🏈 NFL Predictions")

    # Week selector — final list will be overwritten after schedule loads
    current_week = st.selectbox("📅 Select Week", list(range(1,19)), index=0)

    st.markdown("### ⚙️ Model Controls")
    market_weight = st.slider("Market weight (market)", 0.0, 1.0, 0.0, 0.05)
    bet_threshold = st.slider("Bet threshold", 0.0, 20.0, 5.0, 0.5)

    st.markdown("### 📊 Model Record")
    st.info("Model trained using local data + Elo fallback.")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def try_espen_scores(season, week):
    """ESPN scores only — not schedule."""
    try:
        url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/types/2/weeks/{week}/events"
        data = safe_request_json(url)
        if not data or "events" not in data:
            return pd.DataFrame()

        rows = []
        for event in data["events"]:
            try:
                comp = event.get("competitions", [{}])[0]
                teams = comp.get("competitors", [])
                if len(teams) != 2:
                    continue

                home = [t for t in teams if t.get("homeAway")=="home"][0]
                away = [t for t in teams if t.get("homeAway")=="away"][0]

                rows.append({
                    "home_team": canonical_team_name(home["team"]["displayName"]),
                    "away_team": canonical_team_name(away["team"]["displayName"]),
                    "home_score": home.get("score", np.nan),
                    "away_score": away.get("score", np.nan)
                })
            except:
                continue

        return pd.DataFrame(rows)
    except:
        return pd.DataFrame()


def finalize_schedule(df):
    """Normalize names & ensure required fields."""
    if df.empty:
        return df

    df = df.copy()
    df["home_team"] = df["home_team"].apply(lambda x: canonical_team_name(str(x)))
    df["away_team"] = df["away_team"].apply(lambda x: canonical_team_name(str(x)))

    # Fill missing numeric fields
    for c in ["spread", "over_under", "home_score", "away_score"]:
        if c not in df:
            df[c] = np.nan

    return df


# -------------------------------------------------------------------
# Build the Week Schedule (Covers → fastR → ESPN-scores)
# -------------------------------------------------------------------
def load_week_data(season, week):
    # 1 — Try Covers (primary)
    covers = pd.DataFrame()
    try:
        covers = fetch_covers_for_week(season, week)
    except Exception:
        covers = pd.DataFrame()

    if not covers.empty:
        covers["season"] = season
        covers["week"] = week
        return finalize_schedule(covers)

    # 2 — fastR full-season schedule fallback
    fastR = load_fastr_schedule(int(season))
    if not fastR.empty:
        wk = fastR[fastR["week"] == week]
        if not wk.empty:
            wk = wk.copy()
            wk["spread"] = np.nan
            wk["over_under"] = np.nan
            return finalize_schedule(wk)

    # 3 — ESPN scores fallback (not full schedule)
    espn_scores = try_espen_scores(season, week)
    if not espn_scores.empty:
        tmp = espn_scores.copy()
        tmp["season"] = season
        tmp["week"] = week
        tmp["spread"] = np.nan
        tmp["over_under"] = np.nan
        return finalize_schedule(tmp)

    return pd.DataFrame()


# -------------------------------------------------------------------
# MAIN CONTENT
# -------------------------------------------------------------------
st.subheader(f"DJBets — Season {CURRENT_SEASON} — Week {current_week}")

with st.spinner("Loading schedule..."):
    schedule_df = load_week_data(CURRENT_SEASON, int(current_week))

# If nothing worked
if schedule_df.empty:
    st.error("No schedule found from Covers, fastR, or ESPN. This may be temporary (Cloudflare). Try again later.")
    st.stop()

# Show games
st.success(f"Loaded {len(schedule_df)} games for Week {current_week}")

for _, g in schedule_df.iterrows():
    home = g["home_team"]
    away = g["away_team"]

    st.markdown(f"""
    ### **{home.replace('_',' ').title()} vs {away.replace('_',' ').title()}**
    - Spread: {g['spread'] if not pd.isna(g['spread']) else 'N/A'}
    - Total: {g['over_under'] if not pd.isna(g['over_under']) else 'N/A'}
    - Scores: {g['home_score']} — {g['away_score']}
    """)