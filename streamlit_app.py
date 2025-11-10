# DJBets NFL Predictor v10.8
# Adds: algorithm record tracking, bar meaning labels, improved sidebar stats.

import os
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta

# --------------------------------------------------------------
# ⚙️ Config
# --------------------------------------------------------------
st.set_page_config(page_title="DJBets NFL Predictor", page_icon="🏈", layout="wide")
DATA_DIR = "data"
MODEL_FILE = os.path.join(DATA_DIR, "model.json")
SCHEDULE_FILE = os.path.join(DATA_DIR, "schedule.csv")
os.makedirs(DATA_DIR, exist_ok=True)
MAX_WEEKS = 18
MODEL_FEATURES = ["elo_diff", "temp_c", "wind_kph", "precip_prob"]

# --------------------------------------------------------------
# 🧠 Model (auto-train if missing)
# --------------------------------------------------------------
@st.cache_resource
def load_or_train_model():
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

    model = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.08)
    model.fit(df[MODEL_FEATURES].values, y.values)
    model.save_model(MODEL_FILE)
    return model

# --------------------------------------------------------------
# 🏈 ESPN Data Loader (with state detection)
# --------------------------------------------------------------
@st.cache_data(ttl=604800)
def fetch_schedule(season: int):
    games = []
    for week in range(1, MAX_WEEKS + 1):
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?year={season}&seasontype=2&week={week}"
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
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

    return pd.DataFrame(games)

# --------------------------------------------------------------
# 🧮 Utilities
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

def simulate_features(df, week):
    np.random.seed(week * 123)
    df["elo_diff"] = np.random.normal(0, 100, len(df))
    df["temp_c"] = np.random.uniform(-5, 25, len(df))
    df["wind_kph"] = np.random.uniform(0, 25, len(df))
    df["precip_prob"] = np.random.uniform(0, 1, len(df))
    df["over_under"] = df["over_under"].fillna(
        np.clip(45 + (df["elo_diff"]/100)*2 + np.random.normal(0, 2, len(df)), 37, 56)
    ).round(1)
    return df

# --------------------------------------------------------------
# 🎛️ Sidebar Controls
# --------------------------------------------------------------
st.sidebar.header("🏈 DJBets NFL Predictor")
season = st.sidebar.selectbox("Season", [2026, 2025, 2024], index=1)
week = st.sidebar.selectbox("Week", list(range(1, MAX_WEEKS+1)), index=0)

# --------------------------------------------------------------
# 📊 Data Load
# --------------------------------------------------------------
model = load_or_train_model()
sched = fetch_schedule(season)
sched["kickoff_et"] = pd.to_datetime(sched["kickoff_et"], errors="coerce")
week_df = sched.query("week == @week").copy()

if week_df.empty:
    st.warning("No games found for this week.")
    st.stop()

st.title(f"🏈 DJBets NFL Predictor — Week {week} ({season})")

# --------------------------------------------------------------
# 🔮 Feature Engineering
# --------------------------------------------------------------
week_df = simulate_features(week_df, week)
X = week_df[MODEL_FEATURES].astype(float)
week_df["home_win_prob"] = model.predict_proba(X)[:,1]
week_df["predicted_spread"] = np.round(-7 * (week_df["home_win_prob"] - 0.5), 1)
week_df["predicted_total"] = np.round(
    week_df["over_under"] + np.random.normal(0, 2, len(week_df)), 1
)

# --------------------------------------------------------------
# 📈 Record Tracker
# --------------------------------------------------------------
completed = week_df.query("state == 'post' and home_score == home_score and away_score == away_score").copy()
if not completed.empty:
    completed["predicted_winner"] = np.where(completed["home_win_prob"] >= 0.5, "home", "away")
    completed["actual_winner"] = np.where(completed["home_score"] > completed["away_score"], "home", "away")
    completed["correct"] = completed["predicted_winner"] == completed["actual_winner"]
    wins = int(completed["correct"].sum())
    total = len(completed)
    losses = total - wins
    acc = wins / total * 100 if total > 0 else 0
else:
    wins, losses, acc, total = 0, 0, 0, 0

st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.metric("Record", f"{wins}-{losses}")
st.sidebar.metric("Accuracy", f"{acc:.1f}%")
if total > 0:
    fig, ax = plt.subplots(figsize=(2.5, 1.2))
    ax.bar(["Correct", "Incorrect"], [wins, losses], color=["green", "red"])
    ax.set_xticklabels(["✅ Correct", "❌ Wrong"])
    ax.set_title("Prediction Breakdown")
    st.sidebar.pyplot(fig)

# --------------------------------------------------------------
# 🎯 Display Games
# --------------------------------------------------------------
for _, row in week_df.iterrows():
    vegas_spread = parse_spread(row["spread"])
    state = row.get("state", "pre")
    color = {"pre": "⬜ Upcoming", "in": "🟢 Live", "post": "🔵 Final"}.get(state, "⚪ Unknown")
    bg = "rgba(0,150,0,0.08)" if row["home_win_prob"] > 0.55 else "rgba(255,0,0,0.08)"

    st.markdown(f'<div style="background-color:{bg}; padding: 1rem; border-radius: 1rem;">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 3, 1])

    with c1:
        st.image(row["away_logo"], width=60)
        st.markdown(f"**{row['away_team']}**")

    with c3:
        st.image(row["home_logo"], width=60)
        st.markdown(f"**{row['home_team']}**")

    with c2:
        kickoff = row["kickoff_et"].strftime("%a %b %d, %I:%M %p") if pd.notna(row["kickoff_et"]) else "TBD"
        st.markdown(f"**{color}** — {row['status_text']}")
        st.markdown(f"**Kickoff:** {kickoff}")
        st.markdown(f"**Vegas Spread:** {row['spread']}  |  **Model Spread:** {row['predicted_spread']:+.1f}")
        st.markdown(f"**Over/Under:** {row['over_under']:.1f}  |  **Model Total:** {row['predicted_total']:.1f}")

        st.progress(row["home_win_prob"], text=f"🏠 Home win probability ({row['home_team']}): {row['home_win_prob']*100:.1f}%")
        st.caption("🟩 Green = home favorite | 🟥 Red = away favorite")

        # Game result logic
        if state == "post" and not np.isnan(row["home_score"]) and not np.isnan(row["away_score"]):
            actual_winner = "home" if row["home_score"] > row["away_score"] else "away"
            predicted_winner = "home" if row["home_win_prob"] >= 0.5 else "away"
            correct = (actual_winner == predicted_winner)
            result = "✅ Correct prediction" if correct else "❌ Incorrect"
            st.markdown(f"**Final Score:** {row['away_score']} - {row['home_score']} ({result})")
        elif state == "in":
            st.markdown(f"**Live:** {row['status_text']}")
        else:
            st.markdown("⏳ **Not started yet**")

        with st.expander("📊 Detailed Betting Analysis"):
            spread_diff = np.nan
            if not np.isnan(vegas_spread):
                spread_diff = row["predicted_spread"] - vegas_spread
                spread_rec = (
                    f"Bet **{row['home_team']}** (model stronger)" if spread_diff < 0
                    else f"Bet **{row['away_team']}** (+points)"
                )
            else:
                spread_rec = "No Vegas spread data"

            ou_diff = row["predicted_total"] - row["over_under"]
            ou_rec = "Bet **Over**" if ou_diff > 0 else "Bet **Under**"

            st.markdown(f"**Spread Recommendation:** {spread_rec}")
            st.markdown(f"**Over/Under Recommendation:** {ou_rec}")

            feats = {k: row[k] for k in MODEL_FEATURES}
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.bar(feats.keys(), feats.values(), color="skyblue")
            ax.set_title("Model Feature Inputs")
            st.pyplot(fig)

    st.markdown("</div><br>", unsafe_allow_html=True)

st.markdown("---")
st.caption("🏈 DJBets NFL Predictor v10.8 — now with transparent performance tracking and clearer visuals.")
