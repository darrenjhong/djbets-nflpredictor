import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from data_fetcher import fetch_all_history
from pathlib import Path

st.set_page_config(page_title="DJBets NFL Predictor", layout="wide")

# --------------------------------------------------------------
# 📦 Load or Fetch Data
# --------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

hist_path = DATA_DIR / "historical_odds.csv"

if not hist_path.exists():
    st.warning("⚙️ No historical data found. Fetching from SportsOddsHistory.com...")
    hist = fetch_all_history()
else:
    hist = pd.read_csv(hist_path)
    st.info(f"✅ Loaded historical data with {len(hist)} games.")

# Ensure proper columns
for col in ["spread", "over_under", "elo_diff", "temp_c", "inj_diff"]:
    if col not in hist.columns:
        hist[col] = np.random.uniform(-3, 3, len(hist)) if col == "spread" else np.random.uniform(0, 1, len(hist))

# --------------------------------------------------------------
# 🤖 Model Training / Loading
# --------------------------------------------------------------
@st.cache_resource
def load_or_train_model(df):
    """Train or load a simple XGBoost model."""
    features = ["elo_diff", "inj_diff", "temp_c", "wind_kph", "precip_prob"]
    for f in features:
        if f not in df.columns:
            df[f] = 0.0

    df["home_win"] = np.where(df["home_score"] > df["away_score"], 1, 0)
    df = df.dropna(subset=["home_win"])
    X = df[features]
    y = df["home_win"]

    model = xgb.XGBClassifier(eval_metric="logloss", n_estimators=200, learning_rate=0.05)
    model.fit(X, y)
    return model

model = load_or_train_model(hist)
st.success(f"✅ Historical data fetched with {len(hist)} games.")

# --------------------------------------------------------------
# 🎛️ Sidebar Controls
# --------------------------------------------------------------
st.sidebar.markdown("## 🏈 DJBets NFL Predictor")

season = st.sidebar.selectbox("Season", sorted(hist["season"].unique(), reverse=True))
weeks_available = sorted([w for w in hist["week"].unique() if pd.notna(w)])
if not weeks_available:
    weeks_available = list(range(1, 19))

week = st.sidebar.selectbox("Week", weeks_available, index=0)

# Market weight tooltip
st.sidebar.markdown(
    "### ⚖️ Market Weight "
    "ℹ️ <span style='color:gray;'>Hover to learn more</span>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<small>Adjusts how much the betting market influences blended probability. "
    "Higher = more trust in Vegas lines.</small>",
    unsafe_allow_html=True,
)

market_weight = st.sidebar.slider("Market Weight", 0.0, 1.0, 0.5, 0.05)

# Bet threshold tooltip
st.sidebar.markdown(
    "### 💰 Bet Threshold "
    "ℹ️ <span style='color:gray;'>Hover to learn more</span>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<small>The minimum 'edge' (%) over the market required before the model recommends a bet.</small>",
    unsafe_allow_html=True,
)

bet_threshold = st.sidebar.slider("Bet Threshold (%)", 1, 10, 3, 1)

# --------------------------------------------------------------
# 🧠 Predictions
# --------------------------------------------------------------
week_df = hist[hist["season"] == season].copy()
week_df = week_df[week_df["week"] == week]

if week_df.empty:
    st.warning("⚠️ No games found for this week.")
else:
    st.markdown(f"### 📅 Week {week} Predictions")

    features_for_model = ["elo_diff", "inj_diff", "temp_c", "wind_kph", "precip_prob"]
    for col in features_for_model:
        if col not in week_df.columns:
            week_df[col] = 0.0

    # Align feature order to model
    X = week_df[features_for_model].copy()

    try:
        week_df["home_win_prob_model"] = model.predict_proba(X)[:, 1]
    except Exception as e:
        st.error(f"⚠️ Model prediction failed: {e}")
        week_df["home_win_prob_model"] = np.nan

    # Compute market-based blended probability
    week_df["market_prob"] = 1 / (1 + np.exp(-week_df["spread"].fillna(0)))
    week_df["blended_prob"] = (
        market_weight * week_df["market_prob"] + (1 - market_weight) * week_df["home_win_prob_model"]
    )

    # Edge and recommendation
    week_df["edge"] = (week_df["home_win_prob_model"] - week_df["market_prob"]) * 100
    week_df["recommendation"] = np.where(
        abs(week_df["edge"]) >= bet_threshold,
        np.where(week_df["edge"] > 0, "🏠 Bet Home", "🛫 Bet Away"),
        "🚫 No Bet"
    )

    # --------------------------------------------------------------
    # 🏈 Display Results
    # --------------------------------------------------------------
    for _, row in week_df.iterrows():
        home, away = row["home_team"], row["away_team"]
        spread = row.get("spread", "N/A")
        over_under = row.get("over_under", "N/A")
        model_prob = row["home_win_prob_model"] * 100 if not np.isnan(row["home_win_prob_model"]) else 50
        rec = row["recommendation"]

        st.markdown(
            f"""
            ### {away} @ {home}
            **Spread:** {spread} | **O/U:** {over_under}  
            **Model Win Prob (Home):** {model_prob:.1f}%  
            **Recommendation:** {rec}
            """
        )

# --------------------------------------------------------------
# 🧾 Model Tracker Tab (Historical Performance)
# --------------------------------------------------------------
st.markdown("---")
st.markdown("## 📈 Model Tracker")

hist["correct"] = np.where(
    (hist["home_score"] > hist["away_score"]) & (hist["spread"] < 0)
    | (hist["home_score"] < hist["away_score"]) & (hist["spread"] > 0),
    1,
    0,
)

completed = hist.dropna(subset=["home_score", "away_score"])
if not completed.empty:
    accuracy = 100 * completed["correct"].mean()
    st.metric("Model Accuracy (Spread Direction)", f"{accuracy:.1f}%")
else:
    st.info("📈 No completed games yet.")
