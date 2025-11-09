import streamlit as st
import pandas as pd
import xgboost as xgb
from datetime import datetime
import matplotlib.pyplot as plt
import os

# --------------------------------------------------------------
# ⚙️ Page Configuration
# --------------------------------------------------------------
st.set_page_config(
    page_title="DJBets NFL Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    body { background-color: #0E1117; color: #FAFAFA; }
    .stApp { background-color: #0E1117; }
    h1, h2, h3, h4, h5, h6 { color: #FAFAFA !important; }
    .css-1d391kg, .css-1v3fvcr, .css-1offfwp { color: #FAFAFA !important; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------
# 📂 Data Loaders (auto-train on first run)
# --------------------------------------------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
SCHEDULE_FILE = os.path.join(DATA_DIR, "schedule.csv")
HISTORICAL_FILE = os.path.join(DATA_DIR, "historical.csv")
MODEL_FILE = os.path.join(DATA_DIR, "nfl_xgb_model.json")

@st.cache_data
def load_latest_schedule():
    if not os.path.exists(SCHEDULE_FILE):
        st.warning("⚠️ No schedule file found. Please upload schedule.csv to the data folder.")
        return pd.DataFrame()

    df = pd.read_csv(SCHEDULE_FILE)
    # Normalize column names
    df.columns = [c.lower().strip() for c in df.columns]
    if "kickoff_et" not in df.columns:
        for possible in ["kickoff", "date", "time"]:
            if possible in df.columns:
                df.rename(columns={possible: "kickoff_et"}, inplace=True)
                break
    if "kickoff_et" not in df.columns:
        st.warning("⚠️ No 'kickoff/date/time' column found — setting kickoff_et blank.")
        df["kickoff_et"] = ""
    else:
        df["kickoff_et"] = pd.to_datetime(df["kickoff_et"], errors="coerce").dt.strftime("%a %b %d, %I:%M %p")

    return df

@st.cache_data
def load_historical_data():
    if not os.path.exists(HISTORICAL_FILE):
        st.warning("⚠️ No historical data file found in /data.")
        return pd.DataFrame()
    return pd.read_csv(HISTORICAL_FILE)

@st.cache_resource
def load_or_train_model():
    model = xgb.XGBClassifier()
    if os.path.exists(MODEL_FILE):
        try:
            model.load_model(MODEL_FILE)
            return model
        except Exception:
            st.warning("⚠️ Model file invalid, retraining instead.")

    hist = load_historical_data()
    if hist.empty:
        st.warning("⚠️ No training data found; using default model.")
        return model

    # Minimal training data fallback
    hist["elo_diff"] = hist.get("elo_home", 1500) - hist.get("elo_away", 1500)
    X = hist[["elo_diff"]] if "elo_diff" in hist else pd.DataFrame([[0]], columns=["elo_diff"])
    y = (hist["home_score"] > hist["away_score"]).astype(int) if "home_score" in hist else [0]
    model.fit(X, y)
    model.save_model(MODEL_FILE)
    return model


# --------------------------------------------------------------
# 🎛️ Sidebar Controls
# --------------------------------------------------------------
st.sidebar.markdown("## 🏈 DJBets NFL Predictor")

season_options = sorted([2025, 2024, 2023], reverse=True)
st.sidebar.selectbox("Season", season_options, index=0, key="season")

MAX_WEEKS = 18
sched = load_latest_schedule()
available_weeks = sorted(sched["week"].unique().tolist()) if "week" in sched.columns and not sched.empty else list(range(1, MAX_WEEKS + 1))
st.sidebar.selectbox("Week", available_weeks, index=0, key="week")

show_elo = st.sidebar.checkbox("📊 Show Elo Ratings", value=True)
show_weather = st.sidebar.checkbox("🌦️ Show Weather", value=True)


# --------------------------------------------------------------
# 🧠 Load Model + Data
# --------------------------------------------------------------
model = load_or_train_model()
season = st.session_state["season"]
week = st.session_state["week"]

if sched.empty:
    st.stop()

week_df = sched[(sched["season"] == season) & (sched["week"] == week)] if "season" in sched.columns else pd.DataFrame()
if week_df.empty:
    st.warning("No games found for this week.")
    st.stop()


# --------------------------------------------------------------
# 🧮 Compute Predictions
# --------------------------------------------------------------
week_df["elo_home"] = week_df.get("elo_home", pd.Series([1500] * len(week_df)))
week_df["elo_away"] = week_df.get("elo_away", pd.Series([1500] * len(week_df)))
week_df["elo_diff"] = week_df["elo_home"] - week_df["elo_away"]

for col, default in {"temp_c": 20, "wind_kph": 5, "precip_prob": 0}.items():
    if col not in week_df.columns:
        week_df[col] = default

X = week_df[["elo_diff", "temp_c", "wind_kph", "precip_prob"]]
try:
    week_df["home_win_prob"] = model.predict_proba(X)[:, 1]
except Exception as e:
    st.error(f"Prediction error: {e}")
    week_df["home_win_prob"] = 0.5


# --------------------------------------------------------------
# 🏟️ Display Predictions
# --------------------------------------------------------------
st.title(f"🏈 DJBets NFL Predictor")
st.subheader(f"Week {week} — Season {season}")

for _, row in week_df.iterrows():
    home, away = row["home_team"], row["away_team"]
    home_prob = row["home_win_prob"]
    kickoff = row.get("kickoff_et", "")
    elo_h, elo_a = row["elo_home"], row["elo_away"]

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(f"### {home} 🏠 vs ✈️ {away}")
        st.caption(f"Kickoff: {kickoff}")
        st.progress(home_prob)
        st.write(f"**Predicted Winner:** {'🏠 ' + home if home_prob >= 0.5 else '✈️ ' + away}")
        st.caption(f"Home Win Probability: {home_prob:.1%}")

        if show_weather:
            st.caption(f"🌡️ {row['temp_c']}°C | 💨 {row['wind_kph']} km/h | 🌧️ {row['precip_prob']}% rain chance")

    if show_elo:
        with col2:
            fig, ax = plt.subplots(figsize=(3, 1))
            ax.barh([home, away], [elo_h, elo_a], color=["#1f77b4", "#ff7f0e"])
            ax.set_xlabel("Elo Rating")
            ax.set_xlim(1200, 1800)
            plt.tight_layout()
            st.pyplot(fig)

    st.markdown("---")


# --------------------------------------------------------------
# 📅 Footer
# --------------------------------------------------------------
st.caption(f"🔄 Data: {SCHEDULE_FILE} | Model: {MODEL_FILE} | Updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
