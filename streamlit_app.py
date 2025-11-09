# ==============================================================
# 🌟 DJBets NFL Predictor v9.7-S - Interactive + Logos Edition
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
import matplotlib.pyplot as plt
import os

# --------------------------------------------------------------
# ⚙️ Initialization
# --------------------------------------------------------------

DEFAULT_SEASON = 2025
DEFAULT_WEEK = 1
MAX_WEEKS = 18

session_defaults = {
    "season": DEFAULT_SEASON,
    "week": DEFAULT_WEEK,
    "model_trained": False,
    "schedule_loaded": False,
    "active_schedule_file": None,
    "active_historical_file": None,
    "selected_game": None,
    "refresh_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

for k, v in session_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --------------------------------------------------------------
# 🎛️ Sidebar Controls
# --------------------------------------------------------------
st.sidebar.markdown("## 🏈 DJBets NFL Predictor")
st.sidebar.selectbox("Season", [2025, 2024, 2023], index=0, key="season")
st.sidebar.number_input("Week", 1, MAX_WEEKS, step=1, key="week")

if st.sidebar.button("🔄 Reset Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

# --------------------------------------------------------------
# 📂 Data Loaders
# --------------------------------------------------------------
@st.cache_data
def load_schedule(season):
    path = f"data/schedule_{season}.csv"
    if not os.path.exists(path):
        teams = [
            "Chiefs", "Eagles", "Bills", "49ers", "Ravens",
            "Cowboys", "Lions", "Dolphins", "Jets", "Packers",
            "Bengals", "Texans", "Seahawks", "Chargers", "Vikings", "Bears"
        ]
        data = []
        for w in range(1, MAX_WEEKS + 1):
            np.random.shuffle(teams)
            for i in range(0, len(teams), 2):
                data.append({
                    "season": season,
                    "week": w,
                    "home_team": teams[i],
                    "away_team": teams[i+1],
                    "kickoff_et": datetime(2025, 9, 5, 13, 0) + pd.to_timedelta((w-1)*7, "d"),
                    "spread": round(np.random.uniform(-7, 7), 1),
                    "elo_home": np.random.randint(1450, 1700),
                    "elo_away": np.random.randint(1450, 1700),
                    "temp_c": np.random.uniform(-5, 30),
                    "wind_kph": np.random.uniform(0, 40),
                    "precip_prob": np.random.uniform(0, 1),
                })
        df = pd.DataFrame(data)
        os.makedirs("data", exist_ok=True)
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)
    df["kickoff_et"] = pd.to_datetime(df["kickoff_et"], errors="coerce")
    return df

@st.cache_data
def load_historical():
    path = "data/historical.csv"
    if not os.path.exists(path):
        np.random.seed(42)
        df = pd.DataFrame({
            "elo_diff": np.random.randn(200),
            "inj_diff": np.random.randn(200),
            "temp_c": np.random.uniform(-5, 30, 200),
            "wind_kph": np.random.uniform(0, 40, 200),
            "precip_prob": np.random.uniform(0, 1, 200),
            "home_win": np.random.choice([0, 1], 200)
        })
        os.makedirs("data", exist_ok=True)
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)
    return df

# --------------------------------------------------------------
# 🤖 Model
# --------------------------------------------------------------
@st.cache_resource
def train_or_load_model():
    model_path = "data/xgb_model.json"
    if os.path.exists(model_path):
        model = xgb.XGBClassifier()
        try:
            model.load_model(model_path)
            return model
        except:
            pass
    df = load_historical()
    X = df[["elo_diff", "inj_diff", "temp_c", "wind_kph", "precip_prob"]]
    y = df["home_win"]
    model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=4)
    model.fit(X, y)
    model.save_model(model_path)
    return model

model = train_or_load_model()
schedule = load_schedule(st.session_state["season"])
st.session_state["active_schedule_file"] = f"schedule_{st.session_state['season']}.csv"
st.session_state["active_historical_file"] = "historical.csv"

week_df = schedule[schedule["week"] == st.session_state["week"]]
if week_df.empty:
    st.warning("No games found for this week.")
    st.stop()

# --------------------------------------------------------------
# 🧮 Predictions
# --------------------------------------------------------------
feats = pd.DataFrame({
    "elo_diff": week_df["elo_home"] - week_df["elo_away"],
    "inj_diff": np.random.randn(len(week_df)),
    "temp_c": week_df["temp_c"],
    "wind_kph": week_df["wind_kph"],
    "precip_prob": week_df["precip_prob"],
})
probs = model.predict_proba(feats)[:, 1]
week_df["home_win_prob"] = (probs * 100).round(1)

# --------------------------------------------------------------
# 🎨 Layout
# --------------------------------------------------------------
st.title("🏈 DJBets NFL Predictor")
st.markdown(f"### Season {st.session_state['season']} — Week {st.session_state['week']}")
st.caption(
    f"Data: {st.session_state['active_schedule_file']} | "
    f"Updated {st.session_state['refresh_time']}"
)

# --------------------------------------------------------------
# 🧩 Game Cards
# --------------------------------------------------------------
st.markdown("### 🕹️ Click on a matchup to view analysis")

for _, game in week_df.iterrows():
    home = game["home_team"]
    away = game["away_team"]
    key = f"{home}_vs_{away}_W{int(game['week'])}"

    with st.container(border=True):
        cols = st.columns([3, 1, 3])
        with cols[0]:
            st.image(f"public/logos/{away.lower()}.png", width=70)
            st.subheader(away)
            st.text("Away")
        with cols[1]:
            st.markdown("#### 🕒")
            st.write(game["kickoff_et"].strftime("%a, %b %d %I:%M %p"))
            st.write(f"Spread: {game['spread']:+}")
            if st.button("View Details", key=key):
                st.session_state["selected_game"] = key
                st.session_state["selected_data"] = game
                st.experimental_rerun()
        with cols[2]:
            st.image(f"public/logos/{home.lower()}.png", width=70)
            st.subheader(home)
            st.progress(float(game["home_win_prob"]) / 100)
            st.text(f"Win %: {game['home_win_prob']}")

# --------------------------------------------------------------
# 📊 Details Popup
# --------------------------------------------------------------
if st.session_state.get("selected_game"):
    game = st.session_state["selected_data"]
    st.markdown("---")
    st.markdown(f"## 🧠 Matchup Analysis: {game['away_team']} @ {game['home_team']}")
    st.write(f"**Kickoff:** {game['kickoff_et'].strftime('%A, %b %d %I:%M %p')}")
    st.write(f"**ELO Home:** {game['elo_home']} | **ELO Away:** {game['elo_away']}")
    st.write(f"**Weather:** {game['temp_c']:.1f}°C, {game['wind_kph']:.1f} km/h wind, {game['precip_prob']*100:.0f}% rain chance")
    st.write(f"**Vegas Spread:** {game['spread']:+}")
    st.write(f"**Model Home Win Probability:** {game['home_win_prob']}%")

    # Visual
    st.markdown("### 📈 Win Probability Factors")
    fig, ax = plt.subplots()
    features = ["ELO Diff", "Injury Diff", "Temp °C", "Wind (kph)", "Precip Prob"]
    values = [
        game["elo_home"] - game["elo_away"],
        np.random.randn(),
        game["temp_c"],
        game["wind_kph"],
        game["precip_prob"] * 100,
    ]
    ax.barh(features, values, color=["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"])
    st.pyplot(fig)

    if st.button("⬅️ Back to Week View"):
        st.session_state["selected_game"] = None
        st.experimental_rerun()

# --------------------------------------------------------------
# 🧾 Footer
# --------------------------------------------------------------
st.markdown("---")
st.caption(f"🏈 DJBets NFL Predictor v9.7-S • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
