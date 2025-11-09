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

# --------------------------------------------------------------
# 🧠 Load or Train Model (safe + automatic fallback)
# --------------------------------------------------------------
@st.cache_resource
def load_or_train_model():
    """Load a trained XGBoost model or train one safely from local data."""
    # Try to load existing model first
    if os.path.exists(MODEL_FILE):
        try:
            model = xgb.XGBClassifier()
            model.load_model(MODEL_FILE)
            st.success("✅ Loaded trained model from disk.")
            return model
        except Exception as e:
            st.warning(f"⚠️ Could not load model: {e}. Re-training now...")

    # Fallback → train from CSV or mock
    if os.path.exists(HISTORICAL_FILE):
        try:
            df = pd.read_csv(HISTORICAL_FILE)
        except Exception as e:
            st.error(f"❌ Could not read {HISTORICAL_FILE}: {e}")
            df = pd.DataFrame()
    else:
        st.warning("⚠️ No historical data found — creating mock dataset.")
        df = pd.DataFrame({
            "elo_diff": np.random.normal(0, 100, 200),
            "temp_c": np.random.normal(15, 10, 200),
            "wind_kph": np.random.uniform(0, 30, 200),
            "precip_prob": np.random.uniform(0, 100, 200),
            "home_win": np.random.randint(0, 2, 200)
        })

    # Guarantee required columns
    required = ["elo_diff", "temp_c", "wind_kph", "precip_prob", "home_win"]
    for col in required:
        if col not in df.columns:
            df[col] = np.random.normal(0, 1, len(df))

    # Clean numeric and drop NaNs
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if df.empty or len(df) < 10:
        st.warning("⚠️ Historical dataset is too small — generating mock data.")
        df = pd.DataFrame({
            "elo_diff": np.random.normal(0, 100, 200),
            "temp_c": np.random.normal(15, 10, 200),
            "wind_kph": np.random.uniform(0, 30, 200),
            "precip_prob": np.random.uniform(0, 100, 200),
            "home_win": np.random.randint(0, 2, 200)
        })

    # Define X and y safely
    X = df[["elo_diff", "temp_c", "wind_kph", "precip_prob"]]
    y = df["home_win"].astype(int)

    try:
        model = xgb.XGBClassifier(
            eval_metric="logloss",
            use_label_encoder=False,
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08
        )
        model.fit(X.values, y.values)
        model.save_model(MODEL_FILE)
        st.success("✅ Model trained successfully and saved.")
    except Exception as e:
        st.error(f"❌ Model training failed: {e}")
        # Emergency fallback model
        model = xgb.XGBClassifier()
        model.fit(np.random.randn(20, 4), np.random.randint(0, 2, 20))

    return model


# Load or train model at startup
model = load_or_train_model()


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
