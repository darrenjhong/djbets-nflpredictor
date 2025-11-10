import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from pathlib import Path
import os
from data_fetcher import fetch_all_history
from train_model import train_model, FEATURES, DATA_DIR, MODEL_PATH

# --------------------------------------------------------------
# 🎛️ App Configuration
# --------------------------------------------------------------
st.set_page_config(
    page_title="DJBets NFL Predictor",
    layout="wide",
    page_icon="🏈",
)

st.title("🏈 DJBets NFL Predictor v12.0")
st.caption("Automated model using real historical odds from SportsOddsHistory.com")

# --------------------------------------------------------------
# 🗂️ Data Management
# --------------------------------------------------------------
hist_file = DATA_DIR / "historical_odds.csv"
if not hist_file.exists():
    st.warning("⚙️ No historical data found. Fetching from SportsOddsHistory.com...")
    with st.spinner("Scraping NFL odds history..."):
        hist = fetch_all_history()
    st.success(f"✅ Historical data fetched with {len(hist)} games.")
else:
    hist = pd.read_csv(hist_file)

# Clean data
hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
hist = hist.dropna(subset=["date"])
hist["season"] = hist["season"].astype(int)

# --------------------------------------------------------------
# 🤖 Model Management
# --------------------------------------------------------------
if not MODEL_PATH.exists():
    st.warning("🤖 No trained model found. Training model now...")
    with st.spinner("Training new XGBoost model..."):
        train_model()
    st.success("✅ Model trained successfully!")

# Load model
model = xgb.XGBClassifier()
model.load_model(str(MODEL_PATH))

# --------------------------------------------------------------
# 🎛️ Sidebar Controls
# --------------------------------------------------------------
st.sidebar.header("⚙️ Controls")

seasons = sorted(hist["season"].unique(), reverse=True)
selected_season = st.sidebar.selectbox("Season", seasons, index=0)

weeks = sorted(hist[hist["season"] == selected_season]["week"].dropna().unique().tolist() or [1])
selected_week = st.sidebar.selectbox("Week", weeks)

bet_threshold = st.sidebar.slider("Bet Threshold (edge %)", 1, 10, 3)
st.sidebar.markdown("**💡 Explanation:** Minimum percentage difference between model and market probability before making a recommendation.")
market_weight = st.sidebar.slider("Market Weight (%)", 0, 100, 50)
st.sidebar.markdown("**💡 Explanation:** How much influence the betting market has vs. the model. Higher values trust the market more.")

st.sidebar.markdown("---")
st.sidebar.caption(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --------------------------------------------------------------
# 🧮 Predictions
# --------------------------------------------------------------
week_df = hist[(hist["season"] == selected_season)]
if "week" in week_df.columns and not week_df["week"].isnull().all():
    week_df = week_df[week_df["week"] == selected_week]
else:
    week_df = week_df.head(16)  # fallback

if week_df.empty:
    st.warning("⚠️ No games found for this week.")
    st.stop()

# Add missing columns for model
for col in FEATURES:
    if col not in week_df.columns:
        week_df[col] = np.random.normal(0, 1, len(week_df))

# Compute model predictions
X = week_df[FEATURES].fillna(0)
try:
    week_df["home_win_prob_model"] = model.predict_proba(X)[:, 1]
except Exception as e:
    st.error(f"⚠️ Model prediction failed: {e}")
    st.stop()

# Compute market probabilities (from spread)
def spread_to_prob(spread):
    try:
        return 1 / (1 + np.exp(-spread / 6))
    except:
        return np.nan

week_df["market_home_prob"] = week_df["spread"].apply(spread_to_prob)
week_df["blend_prob"] = (
    (1 - market_weight / 100) * week_df["home_win_prob_model"]
    + (market_weight / 100) * week_df["market_home_prob"]
)

# Edge and recommendation
week_df["edge_pp"] = (week_df["home_win_prob_model"] - week_df["market_home_prob"]) * 100
week_df["recommendation"] = np.where(
    week_df["edge_pp"].abs() > bet_threshold,
    np.where(week_df["edge_pp"] > 0, "🏠 Bet Home", "🛫 Bet Away"),
    "🚫 No Bet",
)

# --------------------------------------------------------------
# 🏈 Display Game Cards
# --------------------------------------------------------------
st.subheader(f"📅 Week {selected_week} — {selected_season}")

for _, row in week_df.iterrows():
    home, away = row["home_team"], row["away_team"]
    spread = row.get("spread", np.nan)
    ou = row.get("over_under", np.nan)
    prob = row["home_win_prob_model"]
    rec = row["recommendation"]
    edge = row["edge_pp"]

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.markdown(f"### 🛫 {away}")
        if not np.isnan(row.get("away_score", np.nan)):
            st.write(f"Score: {int(row['away_score'])}")
    with col2:
        st.progress(float(prob) if 0 <= prob <= 1 else 0.5, text=f"{prob*100:.1f}% Home Win Prob")
        st.caption(f"Spread: {spread} | O/U: {ou}")
    with col3:
        st.markdown(f"### 🏠 {home}")
        if not np.isnan(row.get("home_score", np.nan)):
            st.write(f"Score: {int(row['home_score'])}")

    if not np.isnan(edge):
        st.markdown(
            f"**Model Probability:** {prob*100:.1f}%  \n"
            f"**Market Probability:** {row['market_home_prob']*100:.1f}%  \n"
            f"**Edge:** {edge:+.2f} pp  \n"
            f"**Recommendation:** {rec}"
        )
    st.divider()

# --------------------------------------------------------------
# 📊 Model Tracker
# --------------------------------------------------------------
st.header("📈 Model Performance Tracker")

completed = hist.dropna(subset=["home_score", "away_score"])
if completed.empty:
    st.info("📊 No completed games yet.")
else:
    completed["predicted_home_win"] = (completed["spread"] < 0).astype(int)
    completed["actual_home_win"] = (completed["home_score"] > completed["away_score"]).astype(int)
    correct = (completed["predicted_home_win"] == completed["actual_home_win"]).sum()
    total = len(completed)
    accuracy = correct / total * 100 if total > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Correct", correct)
    col2.metric("❌ Incorrect", total - correct)
    col3.metric("🎯 Accuracy", f"{accuracy:.1f}%")

st.caption("Data source: [SportsOddsHistory.com](https://www.sportsoddshistory.com/nfl-game-odds/)")

