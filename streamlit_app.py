# streamlit_app.py — DJBets NFL Predictor v9.6-safe-auto-retrain
# Automatically retrains XGBoost if feature shape mismatch occurs.

import os
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
import streamlit as st
from datetime import datetime
from market_baseline import spread_to_home_prob, blend_probs

# --------------------------------------------------------------
# ⚙️ Setup
st.set_page_config(page_title="DJBets NFL Predictor", page_icon="🏈", layout="wide")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
MODEL_FILE = os.path.join(DATA_DIR, "model.json")
MAX_WEEKS = 18
MODEL_FEATURES = ["elo_diff", "inj_diff", "temp_c", "wind_kph", "precip_prob"]

# Load API key
try:
    ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    ODDS_API_KEY = os.getenv("ODDS_API_KEY")

# --------------------------------------------------------------
# 🧮 Simulated features
def simulate_features(df: pd.DataFrame, week: int):
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
# 🧠 Model loading/training with schema check
def train_fresh_model():
    np.random.seed(42)
    df = pd.DataFrame({
        "elo_diff": np.random.normal(0, 100, 1000),
        "inj_diff": np.random.normal(0, 10, 1000),
        "temp_c": np.random.uniform(-5, 25, 1000),
        "wind_kph": np.random.uniform(0, 25, 1000),
        "precip_prob": np.random.uniform(0, 1, 1000),
    })
    logits = 0.02 * df["elo_diff"] + 0.01 * df["inj_diff"] - 0.04 * df["precip_prob"]
    p = 1 / (1 + np.exp(-logits))
    y = (np.random.uniform(0, 1, 1000) < p).astype(int)
    model = xgb.XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.07)
    model.fit(df[MODEL_FEATURES].values, y.values)
    model.save_model(MODEL_FILE)
    return model

@st.cache_resource
def load_or_train_model():
    if not os.path.exists(MODEL_FILE):
        return train_fresh_model()
    try:
        model = xgb.XGBClassifier()
        model.load_model(MODEL_FILE)
        booster = model.get_booster()
        expected_features = len(booster.feature_names) if booster.feature_names else None
        if expected_features and expected_features != len(MODEL_FEATURES):
            st.warning(f"🔄 Retraining model: expected {expected_features} features, now {len(MODEL_FEATURES)}")
            return train_fresh_model()
        return model
    except Exception as e:
        st.error(f"⚠️ Model load failed, retraining: {e}")
        return train_fresh_model()

# --------------------------------------------------------------
# 📊 Odds + Schedule Loader with Cache + CSV Backup
@st.cache_data(ttl=86400, show_spinner="Fetching schedule (cached 24h)...")
def fetch_schedule_and_odds(season: int, week: int):
    csv_path = f"{DATA_DIR}/odds_{season}_week{week}.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    st.session_state.setdefault("api_calls", 0)
    if st.session_state["api_calls"] >= 450:
        st.warning("⚠️ API quota nearly reached — loading from local cache only.")
        return pd.DataFrame()

    url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?regions=us&markets=spreads,totals&dateFormat=iso&oddsFormat=american&season={season}&week={week}&apiKey={ODDS_API_KEY}"

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        st.session_state["api_calls"] += 1
        data = r.json()
    except Exception as e:
        st.error(f"❌ Odds API failed: {e}")
        return pd.DataFrame()

    games = []
    for ev in data:
        home = ev.get("home_team")
        away = ev.get("away_team")
        kickoff = ev.get("commence_time")
        spread = np.nan
        over_under = np.nan
        for book in ev.get("bookmakers", []):
            for m in book.get("markets", []):
                if m["key"] == "spreads":
                    for o in m.get("outcomes", []):
                        if o["name"] == home:
                            spread = o.get("point", np.nan)
                if m["key"] == "totals":
                    over_under = m["outcomes"][0].get("point", np.nan)
        games.append({
            "season": season,
            "week": week,
            "home_team": home,
            "away_team": away,
            "kickoff_et": kickoff,
            "spread": spread,
            "over_under": over_under,
            "home_score": np.nan,
            "away_score": np.nan,
            "state": "pre"
        })

    df = pd.DataFrame(games)
    df.to_csv(csv_path, index=False)
    return df

# --------------------------------------------------------------
# 💵 ROI Computation
def compute_roi(df: pd.DataFrame):
    pnl, bets = 0.0, 0
    for _, r in df.iterrows():
        if r.get("state") != "post":
            continue
        edge = r.get("edge_pp")
        if edge is None or np.isnan(edge) or abs(edge) < 3:
            continue
        home_won = r.get("home_score", 0) > r.get("away_score", 0)
        bet_home = edge > 0
        bets += 1
        pnl += 1 if ((bet_home and home_won) or (not bet_home and not home_won)) else -1
    roi = (pnl / bets * 100) if bets else 0
    return round(pnl, 2), bets, round(roi, 1)

# --------------------------------------------------------------
# 🎛️ Sidebar
st.sidebar.header("🏈 DJBets NFL Predictor")
season = st.sidebar.selectbox("Season", [2026, 2025, 2024], index=1)
week = st.sidebar.selectbox("Week", list(range(1, MAX_WEEKS + 1)), index=0)
ALPHA = st.sidebar.slider("Market Weight (α)", 0.0, 1.0, 0.6, 0.05,
                          help="Higher = trust market odds more than model.")
edge_thresh = st.sidebar.slider("Bet Threshold (pp)", 0.0, 10.0, 3.0, 0.5,
                                help="Minimum edge before betting.")
if st.sidebar.button("♻️ Force Refresh"):
    fetch_schedule_and_odds.clear()
    st.experimental_rerun()

# --------------------------------------------------------------
# 🧠 Load Model + Data
model = load_or_train_model()
sched = fetch_schedule_and_odds(season, week)

if sched.empty:
    st.warning("No games found for this week.")
    st.stop()

sched["kickoff_et"] = pd.to_datetime(sched["kickoff_et"], errors="coerce")
sched = simulate_features(sched, week)

# --------------------------------------------------------------
# ✅ Predict safely
for f in MODEL_FEATURES:
    if f not in sched.columns:
        sched[f] = 0.0
X = sched[MODEL_FEATURES].fillna(0).astype(float)

try:
    sched["home_win_prob_model"] = model.predict_proba(X.values)[:, 1]
except ValueError as e:
    if "feature shape" in str(e).lower():
        st.warning("🔄 Retraining model due to feature shape mismatch.")
        model = train_fresh_model()
        sched["home_win_prob_model"] = model.predict_proba(X.values)[:, 1]
    else:
        raise e

sched["market_prob_home"] = sched["spread"].apply(spread_to_home_prob)
sched["blended_prob_home"] = [
    blend_probs(m, mk, ALPHA)
    for m, mk in zip(sched["home_win_prob_model"], sched["market_prob_home"])
]
sched["edge_pp"] = (sched["blended_prob_home"] - sched["market_prob_home"]) * 100

# --------------------------------------------------------------
# 📈 Stats
pnl, bets, roi = compute_roi(sched)
st.sidebar.markdown("### 📈 Performance")
st.sidebar.metric("ROI", f"{roi:.1f}%", f"{pnl:+.2f} units")
st.sidebar.metric("Bets", str(bets))
st.sidebar.caption("🟩 Home Favoured 🟥 Away Favoured 🟨 Even")

# --------------------------------------------------------------
# 🖥️ Main UI
st.title(f"🏈 DJBets NFL Predictor — Week {week} ({season})")

for _, row in sched.iterrows():
    prob = float(min(max(row["blended_prob_home"] if not np.isnan(row["blended_prob_home"]) else 0.5, 0), 1))
    color = "🟩 Home Favoured" if prob > 0.55 else ("🟥 Away Favoured" if prob < 0.45 else "🟨 Even")
    edge_txt = f"Edge: {row['edge_pp']:+.2f} pp" if not np.isnan(row["edge_pp"]) else "No edge"

    st.markdown(f"### {row['away_team']} @ {row['home_team']}")
    st.caption(f"Kickoff: {row['kickoff_et']:%a %b %d, %I:%M %p}" if pd.notna(row["kickoff_et"]) else "TBD")
    st.markdown(f"{color} | {edge_txt}")

    st.progress(prob, text=f"Home Win Probability: {prob*100:.1f}%")
    st.markdown(f"Spread: {row['spread']} | O/U: {row['over_under']}")
    st.markdown(f"**Model Probability:** {row['home_win_prob_model']*100:.1f}%")
    st.markdown(f"**Market Probability:** {row['market_prob_home']*100:.1f}%")
    st.markdown(f"**Blended Probability:** {prob*100:.1f}%")

    rec = ("🏠 Bet Home" if row["edge_pp"] > edge_thresh else
           "🛫 Bet Away" if row["edge_pp"] < -edge_thresh else
           "🚫 No Bet")
    st.markdown(f"**Recommendation:** {rec}")

    if row["state"] == "post":
        st.markdown(f"**Final Score:** {row['away_score']} - {row['home_score']}")
    else:
        st.markdown("⏳ Game not started")

st.markdown("---")
st.caption("🏈 DJBets NFL Predictor — v9.6 auto-retrain (safe & quota-aware)")
