import streamlit as st
st.set_page_config(page_title="DJBets NFL Predictor", layout="wide")

import pandas as pd
import numpy as np
import requests
import os

# Local imports
from schedule_loader import load_fastr_schedule
from team_logo_map import canonical_team_name
from utils import safe_request_json, get_logo_path


CURRENT_SEASON = 2025


# -------------------------------------------------
# Covers: primary schedule + odds source
# -------------------------------------------------
def fetch_covers_matchups(season, week):
    """
    Scrapes Covers NFL matchups (very reliable).
    Returns DataFrame with home/away & spreads.
    """
    try:
        url = f"https://www.covers.com/sports/nfl/matchups?selectedDate={season}-W{week:02d}"
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        html = r.text.lower()

        games = []
        blocks = html.split("cmg_matchup_game_box")[1:]

        for block in blocks:
            try:
                # Extract teams
                home = block.split("home-team-name")[1].split(">")[1].split("<")[0]
                away = block.split("away-team-name")[1].split(">")[1].split("<")[0]

                # Attempt spread extraction
                if "spread-display" in block:
                    spread = block.split("spread-display")[1].split(">")[1].split("<")[0]
                else:
                    spread = np.nan

                games.append({
                    "season": season,
                    "week": week,
                    "home_team": canonical_team_name(home),
                    "away_team": canonical_team_name(away),
                    "spread": spread,
                    "over_under": np.nan,
                    "source": "covers"
                })

            except Exception:
                continue

        return pd.DataFrame(games)

    except Exception as e:
        print("[fetch_covers_matchups] ERROR:", e)
        return pd.DataFrame()


# -------------------------------------------------
# ESPN fallback
# -------------------------------------------------
def fetch_espn_scoreboard(season, week):
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?week={week}&year={season}"
        data = safe_request_json(url)
        if not data:
            return pd.DataFrame()

        games = []
        for event in data.get("events", []):
            comp = event.get("competitions", [{}])[0]

            teams = comp.get("competitors", [])
            if len(teams) != 2:
                continue

            home = [t for t in teams if t.get("homeAway") == "home"]
            away = [t for t in teams if t.get("homeAway") == "away"]
            if not home or not away:
                continue

            home = home[0]
            away = away[0]

            games.append({
                "season": season,
                "week": week,
                "home_team": canonical_team_name(home["team"]["shortDisplayName"]),
                "away_team": canonical_team_name(away["team"]["shortDisplayName"]),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "status": comp.get("status", {}).get("type", {}).get("name", "unknown"),
                "source": "espn"
            })

        return pd.DataFrame(games)
    except Exception:
        return pd.DataFrame()


# -------------------------------------------------
# Build week schedule (Covers → ESPN → fastR)
# -------------------------------------------------
def build_week_schedule(season, week):
    # 1) Try Covers
    covers = fetch_covers_matchups(season, week)
    if not covers.empty:
        return covers

    # 2) ESPN fallback
    espn = fetch_espn_scoreboard(season, week)
    if not espn.empty:
        return espn

    # 3) fastR backup
    fastr = load_fastr_schedule(season)

    # Safety: if fastR did not load
    if fastr.empty or "week" not in fastr.columns:
        return pd.DataFrame()

    fastr_week = fastr[fastr["week"] == week]
    return fastr_week.reset_index(drop=True)


# -------------------------------------------------
# Sidebar UI (Style A)
# -------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; margin-bottom:15px;'>
        <img src='https://img.icons8.com/ios-filled/100/target.png' width='55'>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🏈 DJBets NFL Predictor")

    # Week selector: always allow 1–18 even if schedule empty
    week_list = list(range(1, 19))
    current_week = st.selectbox("📅 Select Week", week_list, index=0)

    st.markdown("### ⚙️ Model Controls")
    market_weight = st.slider("Market weight (blend model <> market)", 0.0, 1.0, 0.00, 0.01)
    bet_threshold = st.slider("Bet threshold (edge pts)", 0.0, 20.0, 5.0, 0.5)

    st.markdown("### 📊 Model Record")
    st.info("Model trained using local data + Elo fallback.")


# -------------------------------------------------
# Main UI
# -------------------------------------------------
st.subheader(f"DJBets — Season {CURRENT_SEASON} — Week {current_week}")

with st.spinner("Loading NFL schedule..."):
    sched = build_week_schedule(CURRENT_SEASON, current_week)

if sched.empty:
    st.error("No schedule found from Covers, fastR, or ESPN. This may be temporary. Try again later.")
    st.stop()

st.success(f"Loaded {len(sched)} games for Week {current_week}")

# -------------------------------------------------
# Display schedule + predictions placeholder
# -------------------------------------------------
for idx, game in sched.iterrows():
    home = game.get("home_team", "")
    away = game.get("away_team", "")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.image(get_logo_path(home), width=60)
        st.write(home.replace("_", " ").title())

    with col2:
        st.markdown("### VS")

    with col3:
        st.image(get_logo_path(away), width=60)
        st.write(away.replace("_", " ").title())

    st.divider()