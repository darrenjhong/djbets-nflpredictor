import os
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
from datetime import datetime, timedelta

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="DJBets NFL Predictor v9.5-S",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark modern theme
st.markdown("""
    <style>
        body {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .stApp {
            background-color: #0E1117;
        }
        .block-container {
            padding-top: 2rem;
        }
        h1, h2, h3 {
            color: #00BFFF !important;
        }
        .game-card {
            background-color: #1E222A;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0px 0px 8px rgba(0, 0, 0, 0.5);
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# MOCK DATA GENERATION
# -------------------------------------------------------------
TEAMS = [
    "Kansas City Chiefs", "Buffalo Bills", "Philadelphia Eagles", "Dallas Cowboys",
    "San Francisco 49ers", "Miami Dolphins", "Baltimore Ravens", "Detroit Lions"
]

LOGOS = {
    "Kansas City Chiefs": "https://a.espncdn.com/i/teamlogos/nfl/500/kc.png",
    "Buffalo Bills": "https://a.espncdn.com/i/teamlogos/nfl/500/buf.png",
    "Philadelphia Eagles": "https://a.espncdn.com/i/teamlogos/nfl/500/phi.png",
    "Dallas Cowboys": "https://a.espncdn.com/i/teamlogos/nfl/500/dal.png",
    "San Francisco 49ers": "https://a.espncdn.com/i/teamlogos/nfl/500/sf.png",
    "Miami Dolphins": "https://a.espncdn.com/i/teamlogos/nfl/500/mia.png",
    "Baltimore Ravens": "https://a.espncdn.com/i/teamlogos/nfl/500/bal.png",
    "Detroit Lions": "https://a.espncdn.com/i/teamlogos/nfl/500/det.png",
}

def generate_mock_games(week: int):
    np.random.seed(week)
    games = []
    for i in range(4):
        home, away = np.random.choice(TEAMS, 2, replace=False)
        kickoff = datetime.now() + timedelta(days=np.random.randint(1, 7))
        spread = round(np.random.uniform(-7, 7), 1)
        games.append({
            "week": week,
            "home_team": home,
            "away_team": away,
            "elo_diff": np.random.randn(),
            "inj_diff": np.random.randn(),
            "temp_c": np.random.uniform(-5, 25),
            "wind_kph": np.random.uniform(0, 40),
            "precip_prob": np.random.uniform(0, 1),
            "spread": spread,
            "kickoff": kickoff
        })
    return pd.DataFrame(games)

# -------------------------------------------------------------
# MODEL LOADING / TRAINING
# -------------------------------------------------------------
model_path = "model.xgb"
model = xgb.XGBClassifier()

if not os.path.exists(model_path):
    with st.spinner("Training model for the first time..."):
        df = generate_mock_games(1)
        X = df[["elo_diff", "inj_diff", "temp_c", "wind_kph", "precip_prob"]]
        y = np.random.randint(0, 2, len(df))
        model.fit(X, y)
        model.save_model(model_path)
        time.sleep(1)
        st.success("✅ Model trained and saved!")
else:
    model.load_model(model_path)

# -------------------------------------------------------------
# UI: MAIN APP
# -------------------------------------------------------------
st.title("🏈 DJBets NFL Predictor v9.5-S")
st.caption("Smart predictions powered by XGBoost — with mock ESPN-style data.")

# Week selector
week = st.selectbox("Select Week", [1, 2, 3, 4, 5, 6, 7, 8])

games_df = generate_mock_games(week)
X_pred = games_df[["elo_diff", "inj_diff", "temp_c", "wind_kph", "precip_prob"]]
probs = model.predict_proba(X_pred)[:, 1]

games_df["home_win_prob"] = probs
games_df["predicted_winner"] = np.where(probs >= 0.5, games_df["home_team"], games_df["away_team"])

# -------------------------------------------------------------
# UI: DISPLAY GAMES
# -------------------------------------------------------------
st.subheader(f"📅 Week {week} Matchups & Predictions")

for _, g in games_df.iterrows():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"""
        <div class='game-card'>
            <h3>{g['away_team']} @ {g['home_team']}</h3>
            <p><b>Kickoff:</b> {g['kickoff'].strftime('%A, %b %d %I:%M %p')}</p>
            <p><b>Spread:</b> {g['spread']} pts</p>
            <p><b>Predicted Winner:</b> {g['predicted_winner']}</p>
            <p><b>Home Win Probability:</b> {round(g['home_win_prob']*100,1)}%</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.image(LOGOS.get(g["home_team"], ""), width=80)
        st.image(LOGOS.get(g["away_team"], ""), width=80)

# -------------------------------------------------------------
# OPTIONAL: Game Detail Viewer
# -------------------------------------------------------------
st.markdown("---")
selected_team = st.selectbox("🔍 View Analysis for Team", TEAMS)
st.write(f"### {selected_team} — Analytical Summary")

st.write("""
- **Current Form:** Improving (last 3 games trending up)
- **Defensive Style:** Mix of Cover-2 zone and man coverage
- **Key Factors:**
  - Injuries minimal
  - Weather impact negligible
  - QB performance stable
- **Betting Insights:** Model sees slight value on teams with home ELO edge > +0.5σ
""")

st.info("📊 Tip: Once you integrate real ESPN APIs, this analysis section will auto-populate with real stats, spreads, and performance data.")
