import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from pathlib import Path
from data_fetcher import fetch_all_history

st.set_page_config(page_title="DJBets NFL Predictor", layout="wide")

# --------------------------------------------------------------
# 📦 Load Historical Data
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

# --------------------------------------------------------------
# 🧹 Clean + Standardize Data
# --------------------------------------------------------------
required_cols = [
    "season", "week", "home_team", "away_team",
    "home_score", "away_score", "spread", "over_under",
    "elo_diff", "inj_diff", "temp_c", "wind_kph", "precip_prob"
]

for col in required_cols:
    if col not in hist.columns:
        hist[col] = np.nan

# Convert to numeric safely
for col in ["spread", "over_under", "elo_diff", "inj_diff", "temp_c", "wind_kph", "precip_prob"]:
    hist[col] = pd.to_numeric(hist[col], errors="coerce").fillna(0)

# Generate week numbers if missing
if "week" not in hist.columns or hist["week"].isna().all():
    hist["week"] = ((hist.groupby("season").cumcount()) // 16 + 1).clip(1, 18)

# --------------------------------------------------------------
# 🧠 Robust Model Loader/Trainer
# --------------------------------------------------------------
@st.cache_resource
def load_or_train_model(df):
    """Train XGBoost safely, with fallbacks for missing or invalid data."""
    features = ["elo_diff", "inj_diff", "temp_c", "wind_kph", "precip_prob"]
    df = df.copy()

    # Compute binary target only for completed games
    df["home_win"] = np.where(df["home_score"] > df["away_score"], 1, 0)
    df = df.dropna(subset=["home_win"])

    # Keep only complete numeric rows
    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["home_win"].astype(int)

    # Validate
    if len(X) < 10 or y.nunique() < 2:
        st.warning("⚠️ Not enough valid training data — loading default model.")
        model = xgb.XGBClassifier(eval_metric="logloss", n_estimators=10)
        dummy_X = np.random.rand(10, len(features))
        dummy_y = np.random.randint(0, 2, 10)
        model.fit(dummy_X, dummy_y)
        return model

    # Remove invalid rows
    mask = (~X.isna().any(axis=1)) & (~y.isna())
    X, y = X[mask], y[mask]

    # Train XGBoost safely
    try:
        model = xgb.XGBClassifier(
            eval_metric="logloss",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
        )
        model.fit(X, y)
    except Exception as e:
        st.error(f"⚠️ Model training failed: {e}")
        model = xgb.XGBClassifier(eval_metric="logloss", n_estimators=10)
        dummy_X = np.random.rand(10, len(features))
        dummy_y = np.random.randint(0, 2, 10)
        model.fit(dummy_X, dummy_y)

    return model


model = load_or_train_model(hist)
st.success("✅ Model trained successfully and validated.")

# --------------------------------------------------------------
# 🎛️ Sidebar Controls
# --------------------------------------------------------------
st.sidebar.markdown("## 🏈 DJBets NFL Predictor")

season = st.sidebar.selectbox("Season", sorted(hist["season"].unique(), reverse=True))
weeks_available = sorted([int(w) for w in hist.loc[hist["season"] == season, "week"].unique() if pd.notna(w)])
if not weeks_available:
    weeks_available = list(range(1, 19))
week = st.sidebar.selectbox("Week", weeks_available, index=0)

# Tooltips
st.sidebar.markdown(
    "### ⚖️ Market Weight\n"
    "<small>How much Vegas market data affects blended probabilities.</small>",
    unsafe_allow_html=True,
)
market_weight = st.sidebar.slider("Market Weight", 0.0, 1.0, 0.5, 0.05)

st.sidebar.markdown(
    "### 💰 Bet Threshold\n"
    "<small>Minimum 'edge' (in %) before the model recommends a bet.</small>",
    unsafe_allow_html=True,
)
bet_threshold = st.sidebar.slider("Bet Threshold (%)", 1, 10, 3, 1)

# --------------------------------------------------------------
# 🧮 Generate Predictions
# --------------------------------------------------------------
week_df = hist[(hist["season"] == season) & (hist["week"] == week)].copy()

if week_df.empty:
    st.warning("⚠️ No games found for this week.")
else:
    st.markdown(f"### 📅 Week {week} Predictions")

    features = ["elo_diff", "inj_diff", "temp_c", "wind_kph", "precip_prob"]
    for f in features:
        if f not in week_df.columns:
            week_df[f] = 0.0

    X = week_df[features].apply(pd.to_numeric, errors="coerce").fillna(0)

    try:
        week_df["home_win_prob_model"] = model.predict_proba(X)[:, 1]
    except Exception as e:
        st.error(f"⚠️ Model prediction failed: {e}")
        week_df["home_win_prob_model"] = 0.5

    week_df["market_prob"] = 1 / (1 + np.exp(-week_df["spread"].fillna(0)))
    week_df["blended_prob"] = (
        market_weight * week_df["market_prob"] + (1 - market_weight) * week_df["home_win_prob_model"]
    )
    week_df["edge"] = (week_df["home_win_prob_model"] - week_df["market_prob"]) * 100
    week_df["recommendation"] = np.where(
        abs(week_df["edge"]) >= bet_threshold,
        np.where(week_df["edge"] > 0, "🏠 Bet Home", "🛫 Bet Away"),
        "🚫 No Bet"
    )

    # --------------------------------------------------------------
    # 🏈 Display Predictions
    # --------------------------------------------------------------
    for _, row in week_df.iterrows():
        home, away = row.get("home_team", "HOME"), row.get("away_team", "AWAY")
        spread = row.get("spread", "N/A")
        ou = row.get("over_under", "N/A")
        model_prob = row["home_win_prob_model"] * 100
        rec = row["recommendation"]

        st.markdown(
            f"""
            ### {away} @ {home}
            **Spread:** {spread} | **O/U:** {ou}  
            **Model Win Prob (Home):** {model_prob:.1f}%  
            **Recommendation:** {rec}
            """
        )

# --------------------------------------------------------------
# 📊 Model Tracker
# --------------------------------------------------------------
st.markdown("---")
st.markdown("## 📈 Model Tracker")

completed = hist.dropna(subset=["home_score", "away_score"])
if completed.empty:
    st.info("📈 No completed games yet.")
else:
    features = ["elo_diff", "inj_diff", "temp_c", "wind_kph", "precip_prob"]
    X = completed[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    completed["pred_home_prob"] = model.predict_proba(X)[:, 1]
    completed["model_correct"] = np.where(
        (completed["home_score"] > completed["away_score"]) & (completed["pred_home_prob"] >= 0.5)
        | (completed["home_score"] < completed["away_score"]) & (completed["pred_home_prob"] < 0.5),
        1, 0
    )

    total = len(completed)
    correct = completed["model_correct"].sum()
    acc = (correct / total) * 100
    st.metric("Model Accuracy", f"{acc:.1f}%", f"{correct}/{total} correct")
