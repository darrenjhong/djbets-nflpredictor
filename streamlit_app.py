# DJBets NFL Predictor v11.0
# Adds: market baseline, probability blending, ROI tracking, model vs market edges

import os
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import hashlib

from market_baseline import spread_to_home_prob, blend_probs
from trainer import train_walkforward

# --------------------------------------------------------------
# ⚙️ Configuration
# --------------------------------------------------------------
st.set_page_config(page_title="DJBets NFL Predictor", page_icon="🏈", layout="wide")

DATA_DIR = "data"
MODEL_FILE = os.path.join(DATA_DIR, "model.json")
SCHEDULE_FILE = os.path.join(DATA_DIR, "schedule.csv")
os.makedirs(DATA_DIR, exist_ok=True)
MAX_WEEKS = 18
MODEL_FEATURES = ["elo_diff", "temp_c", "wind_kph", "precip_prob"]

TEAMS = [
    "BUF", "MIA", "NE", "NYJ", "BAL", "CIN", "CLE", "PIT",
    "HOU", "IND", "JAX", "TEN", "DEN", "KC", "LV", "LAC",
    "DAL", "NYG", "PHI", "WAS", "CHI", "DET", "GB", "MIN",
    "ATL", "CAR", "NO", "TB", "ARI", "LAR", "SF", "SEA"
]

# --------------------------------------------------------------
# 🧠 Model
# --------------------------------------------------------------
@st.cache_resource
def load_or_train_model():
    """Load or train a simple baseline XGBoost model."""
    if os.path.exists(MODEL_FILE):
        model = xgb.XGBClassifier()
        model.load_model(MODEL_FILE)
        return model

    np.random.seed(42)
    df = pd.DataFrame({
        "elo_diff": np.random.normal(0, 100, 600),
        "temp_c": np.random.uniform(-5, 25, 600),
        "wind_kph": np.random.uniform(0, 25, 600),
        "precip_prob": np.random.uniform(0, 1, 600),
    })
    logits = 0.015*df["elo_diff"] - 0.04*(df["precip_prob"] - 0.4) - 0.02*(df["wind_kph"] - 10) + 0.01*(df["temp_c"] - 10)
    p = 1 / (1 + np.exp(-logits))
    y = (np.random.uniform(0, 1, 600) < p).astype(int)

    model = xgb.XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.08)
    model.fit(df[MODEL_FEATURES].values, y.values)
    model.save_model(MODEL_FILE)
    return model

# --------------------------------------------------------------
# 🏈 ESPN Data Scraper
# --------------------------------------------------------------
@st.cache_data(ttl=604800)
def fetch_schedule(season: int):
    """Scrape schedule and scores directly from ESPN."""
    games = []
    for week in range(1, MAX_WEEKS + 1):
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?year={season}&seasontype=2&week={week}"
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception:
            continue

        for ev in data.get("events", []):
            comp = (ev.get("competitions") or [{}])[0]
            if not comp.get("competitors"):
                continue

            home, away, home_logo, away_logo = "TBD", "TBD", "", ""
            home_score, away_score = np.nan, np.nan
            state = ev.get("status", {}).get("type", {}).get("state", "pre")
            short_detail = ev.get("status", {}).get("type", {}).get("shortDetail", "")

            for team in comp["competitors"]:
                t = team.get("team", {})
                abbr = t.get("abbreviation", "")
                logo = t.get("logo") or (t.get("logos", [{}])[0].get("href", ""))
                score = team.get("score")
                if team.get("homeAway") == "home":
                    home, home_logo, home_score = abbr, logo, score
                else:
                    away, away_logo, away_score = abbr, logo, score

            odds = (comp.get("odds") or [{}])[0]
            spread = odds.get("details", "N/A")
            over_under = odds.get("overUnder", np.nan)
            kickoff = comp.get("date", None)

            games.append({
                "season": season,
                "week": week,
                "home_team": home,
                "away_team": away,
                "home_logo": home_logo,
                "away_logo": away_logo,
                "kickoff_et": kickoff,
                "spread": spread,
                "over_under": over_under,
                "home_score": pd.to_numeric(home_score, errors="coerce"),
                "away_score": pd.to_numeric(away_score, errors="coerce"),
                "state": state,
                "status_text": short_detail
            })

    df = pd.DataFrame(games)
    df.to_csv(SCHEDULE_FILE, index=False)
    return df

# --------------------------------------------------------------
# 🔧 Helper Functions
# --------------------------------------------------------------
def parse_spread(value):
    if not isinstance(value, str) or value in ["N/A", "", None]:
        return np.nan
    try:
        num = ''.join(ch for ch in value if ch in "+-.0123456789")
        if num == "" or num == ".":
            return np.nan
        return float(num)
    except Exception:
        return np.nan

def simulate_features(df, week=1):
    np.random.seed(week * 123)
    df["elo_diff"] = np.random.normal(0, 100, len(df))
    df["temp_c"] = np.random.uniform(-5, 25, len(df))
    df["wind_kph"] = np.random.uniform(0, 25, len(df))
    df["precip_prob"] = np.random.uniform(0, 1, len(df))
    df["over_under"] = df["over_under"].fillna(
        np.clip(45 + (df["elo_diff"]/100)*2 + np.random.normal(0, 2, len(df)), 37, 56)
    ).round(1)
    return df

def compute_roi(df):
    stake = 1.0
    pnl, bets = 0.0, 0
    for _, r in df.iterrows():
        if np.isnan(r.get("edge_pp")) or abs(r["edge_pp"]) < 3:
            continue
        pick_home = r["edge_pp"] > 0
        if np.isnan(r["home_score"]) or np.isnan(r["away_score"]):
            continue
        home_win = r["home_score"] > r["away_score"]
        win = (pick_home == home_win)
        pnl += (0.91 if win else -1.0)  # -110 line
        bets += 1
    return pnl, bets, (pnl / max(bets, 1))

# --------------------------------------------------------------
# 🎛️ Sidebar Controls
# --------------------------------------------------------------
st.sidebar.header("🏈 DJBets NFL Predictor")
season = st.sidebar.selectbox("Season", [2026, 2025, 2024], index=1)
week = st.sidebar.selectbox("Week", list(range(1, MAX_WEEKS+1)), index=0)

ALPHA = st.sidebar.slider("Market weight (α)", 0.0, 1.0, 0.6, 0.05)
edge_thresh = st.sidebar.slider("Bet threshold (pp)", 0.0, 10.0, 3.0, 0.5)

if st.sidebar.button("♻️ Refresh Data"):
    fetch_schedule.clear()
    st.rerun()

# --------------------------------------------------------------
# 📊 Load Data
# --------------------------------------------------------------
model = load_or_train_model()
sched = fetch_schedule(season)
sched["kickoff_et"] = pd.to_datetime(sched["kickoff_et"], errors="coerce")

week_df = sched.query("week == @week").copy()
if week_df.empty:
    st.warning("No games found for this week.")
    st.stop()

# Generate features
week_df = simulate_features(week_df, week)
X = week_df[MODEL_FEATURES].astype(float)
week_df["home_win_prob_model"] = model.predict_proba(X)[:, 1]
week_df["market_prob_home"] = week_df["spread"].apply(spread_to_home_prob)
week_df["blended_prob_home"] = [
    blend_probs(m, mk, ALPHA) for m, mk in zip(week_df["home_win_prob_model"], week_df["market_prob_home"])
]
week_df["edge_pp"] = (week_df["blended_prob_home"] - week_df["market_prob_home"]) * 100

# ROI tracking
pnl, bets, roi = compute_roi(sched)

# --------------------------------------------------------------
# 🧾 Sidebar Metrics
# --------------------------------------------------------------
st.sidebar.markdown("### 📈 Performance")
st.sidebar.markdown(f"💵 ROI: **{roi*100:.1f}%** ({bets} bets)")
st.sidebar.caption("Based on closed games and simulated -110 lines")

st.sidebar.markdown("---")
st.sidebar.markdown("🟩 = Home favored\n🟥 = Away favored\n🟨 = Neutral")
st.sidebar.caption("Bar = predicted home win probability (blended)")

# --------------------------------------------------------------
# 🎯 Display Games
# --------------------------------------------------------------
st.title(f"🏈 DJBets NFL Predictor — Week {week} ({season})")

for _, row in week_df.iterrows():
    prob = row["blended_prob_home"]
    color = "🟩 Home Favored" if prob > 0.55 else ("🟥 Away Favored" if prob < 0.45 else "🟨 Even")
    edge = row["edge_pp"]
    edge_txt = f"Edge: {edge:+.2f} pp" if not np.isnan(edge) else "No market edge"

    state = row.get("state", "pre")
    status = {"pre": "⏳ Upcoming", "in": "🟢 Live", "post": "🔵 Final"}.get(state, "⚪ Unknown")

    st.markdown(f"### {row['away_team']} @ {row['home_team']} ({status})")
    st.markdown(f"**{color}** | {edge_txt}")
    kickoff = row["kickoff_et"].strftime("%a %b %d, %I:%M %p") if pd.notna(row["kickoff_et"]) else "TBD"
    st.caption(f"Kickoff: {kickoff}")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.image(row["away_logo"], width=60)
        st.markdown(f"**{row['away_team']}**")

    with col3:
        st.image(row["home_logo"], width=60)
        st.markdown(f"**{row['home_team']}**")

    with col2:
        st.progress(prob, text=f"Home Win Probability: {prob*100:.1f}%")
        st.markdown(f"Spread: {row['spread']} | O/U: {row['over_under']}")
        if state == "post" and not np.isnan(row["home_score"]) and not np.isnan(row["away_score"]):
            home_win = row["home_score"] > row["away_score"]
            pred_win = prob >= 0.5
            result = "✅ Correct" if home_win == pred_win else "❌ Missed"
            st.markdown(f"**Final:** {row['away_score']} - {row['home_score']} ({result})")
        elif state == "in":
            st.markdown(f"**Live:** {row['status_text']}")
        else:
            st.markdown("⏳ Not started yet")

        with st.expander("📊 Betting Breakdown"):
            st.markdown(f"**Model Probability:** {row['home_win_prob_model']*100:.1f}%")
            st.markdown(f"**Market Probability:** {row['market_prob_home']*100:.1f}%")
            st.markdown(f"**Blended Probability:** {row['blended_prob_home']*100:.1f}%")
            st.markdown(f"**Edge:** {edge:+.2f} pp")
            st.markdown(f"**Recommendation:** " +
                        ("🏠 Bet Home" if edge > edge_thresh else "🛫 Bet Away" if edge < -edge_thresh else "🚫 No Bet"))

st.markdown("---")
st.caption("🏈 DJBets NFL Predictor v11.0 — blended model + market edge tracking")
