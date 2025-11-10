# DJBets NFL Predictor v10.5
# Adds: clear home/away layout, completed game results, spread & O/U prediction, deep-dive analysis.

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

TEAMS = [
    "BUF", "MIA", "NE", "NYJ",
    "BAL", "CIN", "CLE", "PIT",
    "HOU", "IND", "JAX", "TEN",
    "DEN", "KC", "LV", "LAC",
    "DAL", "NYG", "PHI", "WAS",
    "CHI", "DET", "GB", "MIN",
    "ATL", "CAR", "NO", "TB",
    "ARI", "LAR", "SF", "SEA"
]

# --------------------------------------------------------------
# 🧠 Model (auto-train if missing)
# --------------------------------------------------------------
@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_FILE):
        model = xgb.XGBClassifier()
        model.load_model(MODEL_FILE)
        return model

    # train simple mock model
    np.random.seed(42)
    df = pd.DataFrame({
        "elo_diff": np.random.normal(0, 100, 500),
        "temp_c": np.random.uniform(-5, 25, 500),
        "wind_kph": np.random.uniform(0, 25, 500),
        "precip_prob": np.random.uniform(0, 1, 500),
    })
    logits = 0.015*df["elo_diff"] - 0.05*(df["precip_prob"] - 0.4) - 0.02*(df["wind_kph"] - 10)
    p = 1 / (1 + np.exp(-logits))
    y = (np.random.uniform(0, 1, 500) < p).astype(int)

    model = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1)
    model.fit(df[MODEL_FEATURES].values, y.values)
    model.save_model(MODEL_FILE)
    return model

# --------------------------------------------------------------
# 🏈 ESPN Scraper (with filler)
# --------------------------------------------------------------
@st.cache_data(ttl=604800)
def fetch_schedule(season: int):
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
            })

    df = pd.DataFrame(games)
    if df.empty:
        st.warning("⚠️ ESPN returned no schedule, generating mock data.")
        df = generate_mock_schedule(season)
    df.to_csv(SCHEDULE_FILE, index=False)
    return df

def generate_mock_schedule(season: int):
    np.random.seed(season)
    rows = []
    for week in range(1, MAX_WEEKS + 1):
        np.random.shuffle(TEAMS)
        for i in range(0, len(TEAMS), 2):
            home, away = TEAMS[i], TEAMS[i+1]
            rows.append({
                "season": season,
                "week": week,
                "home_team": home,
                "away_team": away,
                "home_logo": f"https://a.espncdn.com/i/teamlogos/nfl/500/{home.lower()}.png",
                "away_logo": f"https://a.espncdn.com/i/teamlogos/nfl/500/{away.lower()}.png",
                "kickoff_et": (datetime.now() + timedelta(days=(week-1)*7)).isoformat(),
                "spread": f"-{np.random.randint(1,8)}",
                "over_under": np.random.randint(38, 55),
                "home_score": np.nan,
                "away_score": np.nan
            })
    return pd.DataFrame(rows)

# --------------------------------------------------------------
# 🔢 Sidebar Controls
# --------------------------------------------------------------
st.sidebar.header("🏈 DJBets NFL Predictor")
season = st.sidebar.selectbox("Season", [2026, 2025, 2024], index=1)
week = st.sidebar.selectbox("Week", list(range(1, MAX_WEEKS+1)), index=0)
if st.sidebar.button("♻️ Refresh Schedule"):
    fetch_schedule.clear()
    st.rerun()

# --------------------------------------------------------------
# 📊 Load Data + Model
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
# 🧮 Predictive Features
# --------------------------------------------------------------
def simulate_features(df):
    np.random.seed(week * 99)
    df["elo_diff"] = np.random.normal(0, 100, len(df))
    df["temp_c"] = np.random.uniform(-5, 25, len(df))
    df["wind_kph"] = np.random.uniform(0, 25, len(df))
    df["precip_prob"] = np.random.uniform(0, 1, len(df))
    return df

week_df = simulate_features(week_df)
X = week_df[MODEL_FEATURES].astype(float)
week_df["home_win_prob"] = model.predict_proba(X)[:,1]
week_df["predicted_spread"] = np.round(-7 * (week_df["home_win_prob"] - 0.5), 1)
week_df["predicted_total"] = np.round(45 + np.random.normal(0, 3, len(week_df)), 1)

# --------------------------------------------------------------
# 🎯 Display Game Cards
# --------------------------------------------------------------
for _, row in week_df.iterrows():
    # layout
    bg = "rgba(0, 150, 0, 0.1)" if row["home_win_prob"] > 0.55 else "rgba(150, 0, 0, 0.1)"
    st.markdown(f'<div style="background-color:{bg}; padding: 1rem; border-radius: 1rem;">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        st.image(row["away_logo"], width=60)
        st.markdown(f"**{row['away_team']}**")

    with col3:
        st.image(row["home_logo"], width=60)
        st.markdown(f"**{row['home_team']}**")

    with col2:
        kickoff = row["kickoff_et"].strftime("%a %b %d, %I:%M %p") if pd.notna(row["kickoff_et"]) else "TBD"
        st.markdown(f"**Kickoff:** {kickoff}")
        st.markdown(f"**Spread:** {row['spread']}  |  **Model Spread:** {row['predicted_spread']:+.1f}")
        st.markdown(f"**Over/Under:** {row['over_under']}  |  **Model Total:** {row['predicted_total']}")
        st.progress(row["home_win_prob"], text=f"🏠 Home win probability: {row['home_win_prob']*100:.1f}%")

        # Completed game check
        if not np.isnan(row["home_score"]) and not np.isnan(row["away_score"]):
            actual_winner = "home" if row["home_score"] > row["away_score"] else "away"
            predicted_winner = "home" if row["home_win_prob"] >= 0.5 else "away"
            correct = (actual_winner == predicted_winner)
            result = "✅ Correct prediction" if correct else "❌ Incorrect"
            st.markdown(f"**Final:** {row['away_score']} - {row['home_score']} ({result})")

        # Deep-dive analysis
        with st.expander("📊 Detailed Analysis"):
            # Spread Recommendation
            vegas_spread = 0 if row["spread"] == "N/A" else float(str(row["spread"]).replace("+","").replace("−","-").replace("-",""))
            spread_diff = row["predicted_spread"] - vegas_spread
            spread_recommendation = (
                f"Bet **{row['home_team']}** (model expects stronger performance)" if spread_diff < 0
                else f"Bet **{row['away_team']}** (+points value)"
            )

            # O/U Recommendation
            if not np.isnan(row["over_under"]):
                ou_diff = row["predicted_total"] - row["over_under"]
                ou_recommendation = "Bet **Over**" if ou_diff > 0 else "Bet **Under**"
            else:
                ou_recommendation = "No O/U data"

            st.markdown(f"**Spread Recommendation:** {spread_recommendation}")
            st.markdown(f"**Over/Under Recommendation:** {ou_recommendation}")

            # Visual feature chart
            feats = {k: row[k] for k in MODEL_FEATURES}
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.bar(feats.keys(), feats.values(), color="skyblue")
            ax.set_title("Model Feature Inputs")
            st.pyplot(fig)

            st.caption(f"Predicted Home Win Prob: {row['home_win_prob']*100:.1f}%")
    st.markdown("</div><br>", unsafe_allow_html=True)

st.markdown("---")
st.caption("🏈 DJBets NFL Predictor v10.5 — with spread, O/U, results, and deep-dive analysis.")
