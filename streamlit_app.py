import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from pathlib import Path
from data_fetcher import fetch_all_history
from data_updater import refresh_data, check_last_update

import schedule, time
import threading

def background_refresh():
    schedule.every().monday.at("06:00").do(refresh_data)
    while True:
        schedule.run_pending()
        time.sleep(3600)

threading.Thread(target=background_refresh, daemon=True).start()


st.set_page_config(page_title="DJBets NFL Predictor", layout="wide")

# --------------------------------------------------------------
# ⚙️ Data Handling
# --------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
DATA_PATH = DATA_DIR / "historical_odds.csv"
DATA_DIR.mkdir(exist_ok=True)

# Manual refresh button
with st.sidebar:
    st.markdown("## 🧠 Data Controls")
    if st.button("🔄 Refresh Data Now"):
        with st.spinner("Fetching new data..."):
            df_new = refresh_data()
            st.success(f"✅ Data refreshed ({len(df_new)} games). Please reload the app.")
    st.caption(f"🕒 Last updated: {check_last_update()}")

# Load existing data or fetch new
if not DATA_PATH.exists():
    st.warning("⚙️ No local data found. Fetching fresh...")
    hist = fetch_all_history()
    hist.to_csv(DATA_PATH, index=False)
else:
    hist = pd.read_csv(DATA_PATH)
    st.info(f"✅ Loaded historical data with {len(hist)} games.")

# --------------------------------------------------------------
# 🧭 Derive Season + Week
# --------------------------------------------------------------
if "date" in hist.columns:
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
else:
    hist["date"] = pd.Timestamp.now()

if "season" not in hist.columns:
    hist["season"] = hist["date"].dt.year
hist["season"] = hist["season"].fillna(2025).astype(int)

if "week" not in hist.columns or hist["week"].isna().all():
    hist = hist.sort_values("date")
    hist["week"] = hist.groupby("season").cumcount() // 16 + 1
hist["week"] = hist["week"].clip(1, 18).astype(int)

# --------------------------------------------------------------
# 🏈 Fix Team Codes
# --------------------------------------------------------------
TEAM_MAP = {
    "1": "ARI", "2": "ATL", "3": "BAL", "4": "BUF", "5": "CAR", "6": "CHI", "7": "CIN", "8": "CLE",
    "9": "DAL", "10": "DEN", "11": "DET", "12": "GB", "13": "HOU", "14": "IND", "15": "JAX", "16": "KC",
    "17": "LV", "18": "LAC", "19": "LAR", "20": "MIA", "21": "MIN", "22": "NE", "23": "NO", "24": "NYG",
    "25": "NYJ", "26": "PHI", "27": "PIT", "28": "SEA", "29": "SF", "30": "TB", "31": "TEN", "32": "WAS",
}

def normalize_team(value):
    if pd.isna(value):
        return None
    s = str(value).strip().upper()
    if s in TEAM_MAP.values():
        return s
    if s in TEAM_MAP:
        return TEAM_MAP[s]
    return s

for col in ["home_team", "away_team"]:
    if col not in hist.columns:
        hist[col] = None
    hist[col] = hist[col].apply(normalize_team)

# --------------------------------------------------------------
# 🧹 Clean
# --------------------------------------------------------------
for col in ["spread", "over_under", "elo_diff", "inj_diff", "temp_c", "wind_kph", "precip_prob"]:
    if col not in hist.columns:
        hist[col] = 0
    hist[col] = pd.to_numeric(hist[col], errors="coerce").fillna(0)

for col in ["home_score", "away_score"]:
    if col not in hist.columns:
        hist[col] = np.nan

# --------------------------------------------------------------
# 🧠 Model
# --------------------------------------------------------------
@st.cache_resource
def load_or_train_model(df):
    features = ["elo_diff", "inj_diff", "temp_c", "wind_kph", "precip_prob"]
    df["home_win"] = np.where(df["home_score"] > df["away_score"], 1, 0)
    df = df.dropna(subset=["home_win"])
    X = df[features].fillna(0)
    y = df["home_win"].astype(int)
    if len(X) < 10 or y.nunique() < 2:
        st.warning("⚠️ Not enough valid data — using fallback model.")
        model = xgb.XGBClassifier(eval_metric="logloss", n_estimators=10)
        model.fit(np.random.rand(20, len(features)), np.random.randint(0, 2, 20))
        return model
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
    return model

model = load_or_train_model(hist)
st.success("✅ Model trained successfully.")

# --------------------------------------------------------------
# 🎛️ Sidebar
# --------------------------------------------------------------
st.sidebar.markdown("## 🏈 DJBets NFL Predictor")

season = st.sidebar.selectbox("Season", sorted(hist["season"].unique(), reverse=True))
week = st.sidebar.selectbox("Week", list(range(1, 19)), index=0)

st.sidebar.markdown(
    "### ⚖️ Market Weight\n<small>How much Vegas market data influences blended probabilities.</small>",
    unsafe_allow_html=True,
)
market_weight = st.sidebar.slider("Market Weight", 0.0, 1.0, 0.5, 0.05)

st.sidebar.markdown(
    "### 💰 Bet Threshold\n<small>Minimum edge (%) before recommending a bet.</small>",
    unsafe_allow_html=True,
)
bet_threshold = st.sidebar.slider("Bet Threshold (%)", 1, 10, 3, 1)

# --------------------------------------------------------------
# 🧮 Predictions
# --------------------------------------------------------------
week_df = hist[(hist["season"] == season) & (hist["week"] == week)].copy()
if week_df.empty:
    st.warning(f"⚠️ No games found for Week {week}. Showing placeholder matchups.")
    teams = ["KC", "BUF", "PHI", "DAL", "SF", "GB"]
    week_df = pd.DataFrame({
        "home_team": np.random.choice(teams, 3, replace=False),
        "away_team": np.random.choice(teams, 3, replace=False),
        "spread": np.random.uniform(-7, 7, 3),
        "over_under": np.random.uniform(40, 50, 3),
        "elo_diff": np.random.uniform(-50, 50, 3),
        "inj_diff": np.random.uniform(-10, 10, 3),
        "temp_c": np.random.uniform(-5, 25, 3),
        "wind_kph": np.random.uniform(0, 20, 3),
        "precip_prob": np.random.uniform(0, 100, 3),
    })

features = ["elo_diff", "inj_diff", "temp_c", "wind_kph", "precip_prob"]
X = week_df[features].fillna(0)
week_df["home_win_prob_model"] = model.predict_proba(X)[:, 1]
week_df["market_prob"] = 1 / (1 + np.exp(-week_df["spread"].fillna(0)))
week_df["blended_prob"] = market_weight * week_df["market_prob"] + (1 - market_weight) * week_df["home_win_prob_model"]
week_df["edge"] = (week_df["home_win_prob_model"] - week_df["market_prob"]) * 100
week_df["recommendation"] = np.where(
    abs(week_df["edge"]) >= bet_threshold,
    np.where(week_df["edge"] > 0, "🏠 Bet Home", "🛫 Bet Away"),
    "🚫 No Bet"
)

# --------------------------------------------------------------
# 🏈 Display
# --------------------------------------------------------------
for _, row in week_df.iterrows():
    home, away = row["home_team"], row["away_team"]
    spread, ou, prob, rec = row["spread"], row["over_under"], row["home_win_prob_model"] * 100, row["recommendation"]
    home_logo, away_logo = f"logos/{home}.png", f"logos/{away}.png"

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.image(away_logo, width=80)
        st.markdown(f"**{away}**")
    with col2:
        st.markdown(
            f"**Spread:** {spread:.1f}<br>**O/U:** {ou:.1f}<br>**Home Win Prob:** {prob:.1f}%<br>**{rec}**",
            unsafe_allow_html=True,
        )
    with col3:
        st.image(home_logo, width=80)
        st.markdown(f"**{home}**")

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
    X = completed[features].fillna(0)
    completed["pred_home_prob"] = model.predict_proba(X)[:, 1]
    completed["model_correct"] = np.where(
        (completed["home_score"] > completed["away_score"]) & (completed["pred_home_prob"] >= 0.5)
        | (completed["home_score"] < completed["away_score"]) & (completed["pred_home_prob"] < 0.5),
        1, 0
    )
    correct = completed["model_correct"].sum()
    total = len(completed)
    acc = correct / total * 100
    st.metric("Model Accuracy", f"{acc:.1f}%", f"{correct}/{total} correct")
