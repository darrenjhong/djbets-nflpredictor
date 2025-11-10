# streamlit_app.py — DJBets NFL Predictor v9.7 (Real ESPN weeks + spreads + safe retrain)

import os
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
import streamlit as st
from datetime import datetime

# --------------------------------------------------------------
# 🧱 Setup
st.set_page_config(page_title="DJBets NFL Predictor", page_icon="🏈", layout="wide")
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
MODEL_FILE = os.path.join(DATA_DIR, "model.json")
MAX_WEEKS = 18
MODEL_FEATURES = ["elo_diff", "inj_diff", "temp_c", "wind_kph", "precip_prob"]

# Load Odds API Key if available
try:
    ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    ODDS_API_KEY = os.getenv("ODDS_API_KEY")

# --------------------------------------------------------------
# 🧮 Feature simulation
def simulate_features(df, week):
    np.random.seed(week)
    df["elo_home"] = np.random.normal(1550, 100, len(df))
    df["elo_away"] = np.random.normal(1500, 100, len(df))
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    df["inj_diff"] = np.random.normal(0, 5, len(df))
    df["temp_c"] = np.random.normal(10, 8, len(df))
    df["wind_kph"] = np.random.uniform(0, 30, len(df))
    df["precip_prob"] = np.random.uniform(0, 1, len(df))
    return df

# --------------------------------------------------------------
# 🧠 Model handling
def train_fresh_model():
    np.random.seed(42)
    df = pd.DataFrame({
        "elo_diff": np.random.normal(0, 100, 1000),
        "inj_diff": np.random.normal(0, 10, 1000),
        "temp_c": np.random.uniform(-5, 25, 1000),
        "wind_kph": np.random.uniform(0, 25, 1000),
        "precip_prob": np.random.uniform(0, 1, 1000),
    })
    logits = 0.02 * df["elo_diff"] + 0.01 * df["inj_diff"] - 0.03 * df["precip_prob"]
    p = 1 / (1 + np.exp(-logits))
    y = (np.random.uniform(0, 1, 1000) < p).astype(int)
    model = xgb.XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.07)
    model.fit(df[MODEL_FEATURES], y)
    model.save_model(MODEL_FILE)
    return model

@st.cache_resource
def load_or_train_model():
    if not os.path.exists(MODEL_FILE):
        return train_fresh_model()
    try:
        model = xgb.XGBClassifier()
        model.load_model(MODEL_FILE)
        expected = len(model.get_booster().feature_names or [])
        if expected != len(MODEL_FEATURES):
            st.warning(f"Retraining model: expected {expected}, got {len(MODEL_FEATURES)}")
            return train_fresh_model()
        return model
    except Exception:
        st.warning("Model load failed — retraining.")
        return train_fresh_model()

# --------------------------------------------------------------
# 🗓️ ESPN Schedule Loader
@st.cache_data(ttl=3600, show_spinner="Fetching NFL schedule...")
def fetch_schedule_espn(season: int, week: int):
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?year={season}&week={week}"
    r = requests.get(url)
    if r.status_code != 200:
        st.error("Failed to fetch schedule from ESPN.")
        return pd.DataFrame()

    data = r.json()
    games = []
    for ev in data.get("events", []):
        comp = ev.get("competitions", [{}])[0]
        home = next((c for c in comp["competitors"] if c["homeAway"] == "home"), None)
        away = next((c for c in comp["competitors"] if c["homeAway"] == "away"), None)
        if not home or not away:
            continue
        games.append({
            "season": season,
            "week": week,
            "home_team": home["team"]["displayName"],
            "away_team": away["team"]["displayName"],
            "home_score": int(home.get("score", 0)),
            "away_score": int(away.get("score", 0)),
            "kickoff_et": comp.get("date"),
            "state": comp.get("status", {}).get("type", {}).get("name", "pre").lower(),
        })
    df = pd.DataFrame(games)
    df["kickoff_et"] = pd.to_datetime(df["kickoff_et"], errors="coerce")
    return df

# --------------------------------------------------------------
# 🏈 Merge with Odds API (if available)
@st.cache_data(ttl=86400, show_spinner="Merging spreads and totals...")
def merge_odds(df, season, week):
    if not ODDS_API_KEY:
        df["spread"] = np.nan
        df["over_under"] = np.nan
        return df

    try:
        url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?regions=us&markets=spreads,totals&dateFormat=iso&oddsFormat=american&apiKey={ODDS_API_KEY}"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            st.warning("Odds API unavailable — skipping spreads.")
            df["spread"] = np.nan
            df["over_under"] = np.nan
            return df
        data = r.json()
        odds_map = {}
        for ev in data:
            odds_map[ev["home_team"]] = {
                "spread": ev.get("bookmakers", [{}])[0].get("markets", [{}])[0].get("outcomes", [{}])[0].get("point", np.nan),
                "over_under": ev.get("bookmakers", [{}])[0].get("markets", [{}])[1].get("outcomes", [{}])[0].get("point", np.nan),
            }
        df["spread"] = df["home_team"].map(lambda t: odds_map.get(t, {}).get("spread", np.nan))
        df["over_under"] = df["home_team"].map(lambda t: odds_map.get(t, {}).get("over_under", np.nan))
        return df
    except Exception as e:
        st.warning(f"Odds merge failed: {e}")
        df["spread"] = np.nan
        df["over_under"] = np.nan
        return df

# --------------------------------------------------------------
# 📈 ROI tracking
def compute_roi(df):
    pnl, bets = 0.0, 0
    for _, r in df.iterrows():
        if r.get("state") != "post":
            continue
        edge = r.get("edge_pp")
        if edge is None or np.isnan(edge) or abs(edge) < 3:
            continue
        bets += 1
        home_win = r["home_score"] > r["away_score"]
        bet_home = edge > 0
        pnl += 1 if (home_win == bet_home) else -1
    roi = (pnl / bets * 100) if bets else 0
    return round(pnl, 2), bets, round(roi, 1)

# --------------------------------------------------------------
# 🎛️ Sidebar
st.sidebar.header("🏈 DJBets NFL Predictor")
season = st.sidebar.selectbox("Season", [2026, 2025, 2024], index=1)
week = st.sidebar.selectbox("Week", list(range(1, MAX_WEEKS + 1)), index=0)
alpha = st.sidebar.slider("Market Weight (α)", 0.0, 1.0, 0.6, 0.05)
edge_thresh = st.sidebar.slider("Bet Threshold (pp)", 0.0, 10.0, 3.0, 0.5)

# --------------------------------------------------------------
# 📊 Load Model + Schedule
model = load_or_train_model()
sched = fetch_schedule_espn(season, week)
sched = merge_odds(sched, season, week)
sched = simulate_features(sched, week)

if sched.empty:
    st.warning("No games found for this week.")
    st.stop()

# --------------------------------------------------------------
# 🧠 Predictions
X = sched[MODEL_FEATURES].fillna(0).astype(float)
sched["home_win_prob_model"] = model.predict_proba(X.values)[:, 1]
sched["market_prob_home"] = 1 / (1 + np.exp(-0.2 * sched["spread"].fillna(0)))
sched["blended_prob_home"] = (1 - alpha) * sched["home_win_prob_model"] + alpha * sched["market_prob_home"]
sched["edge_pp"] = (sched["blended_prob_home"] - sched["market_prob_home"]) * 100

# --------------------------------------------------------------
# 📈 Sidebar performance
pnl, bets, roi = compute_roi(sched)
st.sidebar.markdown("### 📈 Model Performance")
st.sidebar.metric("ROI", f"{roi:.1f}%", f"{pnl:+.2f} units")
st.sidebar.metric("Bets Made", str(bets))

# --------------------------------------------------------------
# 🖥️ Main display
st.title(f"🏈 DJBets NFL Predictor — {season} Week {week}")

for _, row in sched.iterrows():
    kickoff = row["kickoff_et"].strftime("%a %b %d %I:%M %p") if pd.notna(row["kickoff_et"]) else "TBD"
    prob = row["blended_prob_home"]
    color = "🟩 Home Favoured" if prob > 0.55 else ("🟥 Away Favoured" if prob < 0.45 else "🟨 Even")
    rec = ("🏠 Bet Home" if row["edge_pp"] > edge_thresh else
           "🛫 Bet Away" if row["edge_pp"] < -edge_thresh else
           "🚫 No Bet")

    st.markdown(f"### {row['away_team']} @ {row['home_team']}")
    st.caption(f"Kickoff: {kickoff}")
    st.markdown(f"{color} | Edge: {row['edge_pp']:+.2f} pp | Spread: {row['spread']} | O/U: {row['over_under']}")

    st.progress(min(max(prob, 0), 1), text=f"Home Win Probability: {prob*100:.1f}%")
    st.markdown(f"**Model Probability:** {row['home_win_prob_model']*100:.1f}%")
    st.markdown(f"**Market Probability:** {row['market_prob_home']*100:.1f}%")
    st.markdown(f"**Blended Probability:** {row['blended_prob_home']*100:.1f}%")
    st.markdown(f"**Recommendation:** {rec}")

    if row["state"] == "post":
        st.markdown(f"**Final Score:** {row['away_score']} - {row['home_score']}")
    else:
        st.markdown("⏳ Game not started")

st.caption("v9.7 — ESPN-integrated schedule with Odds API merge & retrain-safe model")
