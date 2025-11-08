import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

st.set_page_config(page_title="DJBets NFL Predictor", page_icon="🏈", layout="wide")

st.markdown(
    """
    <h1 style='text-align:center; color:#00BFFF;'>🏈 DJBets NFL Predictor</h1>
    <p style='text-align:center; color:gray;'>Automated Win Probabilities • Elo + Injuries + Weather + Spread</p>
    """,
    unsafe_allow_html=True
)


# Load model
model_path = Path("models/xgb_model.json")
model = xgb.XGBClassifier()
model.load_model(model_path)

# Load mock data
schedule = pd.read_csv("data/schedule_2025.csv")
logos_path = Path("public/logos")

st.title("🏈 DJBets NFL Predictor")
week = st.selectbox("Select Week", sorted(schedule["week"].unique()))
games = schedule[schedule["week"] == week]

cols = st.columns(2)
for _, game in games.iterrows():
    home_logo = str(logos_path / f"{game['home_team']}.svg")
    away_logo = str(logos_path / f"{game['away_team']}.svg")
    with cols[_ % 2]:
        st.markdown("---")
        c1, c2 = st.columns([1, 3])
        with c1:
            st.image(home_logo, width=70)
            st.image(away_logo, width=70)
        with c2:
            st.markdown(f"### {game['away_team']} @ {game['home_team']}")
            st.caption(f"**Weather:** {game['weather']} | **Spread:** {game['spread']} | **Home Win %:** {game['home_win_prob']*100:.1f}%")
            with st.expander("📊 Game Analysis"):
                st.write(f"**Elo Diff:** {game['elo_diff']}")
                st.write(f"**Injury Diff:** {game['inj_diff']}")
                st.write(f"**Predicted Winner:** {'🏠 Home' if game['home_win_prob'] > 0.5 else '🛫 Away'}")

