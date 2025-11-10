import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import xgboost as xgb
from datetime import datetime, timedelta

# --------------------------------------------------------------
# 🗂️ Configuration
# --------------------------------------------------------------
st.set_page_config(page_title="DJBets NFL Predictor", page_icon="🏈", layout="wide")
DATA_DIR = "data"
SCHEDULE_FILE = os.path.join(DATA_DIR, "schedule.csv")
HISTORY_FILE = os.path.join(DATA_DIR, "historical.csv")
MODEL_FILE = os.path.join(DATA_DIR, "model.json")
MAX_WEEKS = 18
MODEL_FEATURES = ["elo_diff", "temp_c", "wind_kph", "precip_prob"]

os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------------------------------------------
# 🧠 Load or Train Model
# --------------------------------------------------------------
@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_FILE):
        model = xgb.XGBClassifier()
        model.load_model(MODEL_FILE)
        st.success("✅ Loaded existing model.")
        return model

    st.warning("⚙️ No model found — training a fresh one...")

    # Generate mock historical data for first-time runs
    np.random.seed(42)
    df = pd.DataFrame({
        "elo_diff": np.random.normal(0, 100, 200),
        "temp_c": np.random.uniform(-5, 25, 200),
        "wind_kph": np.random.uniform(0, 30, 200),
        "precip_prob": np.random.uniform(0, 1, 200),
        "home_win": np.random.binomial(1, 0.5, 200),
    })

    X, y = df[MODEL_FEATURES], df["home_win"]

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X.values, y.values)
    model.save_model(MODEL_FILE)
    st.success("🎯 Model trained and saved.")
    return model

# --------------------------------------------------------------
# 📅 ESPN Schedule Scraper (auto-refresh weekly)
# --------------------------------------------------------------
@st.cache_data(ttl=604800)  # refresh every 7 days
def scrape_espn_schedule(season: int, force_refresh: bool = False):
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(SCHEDULE_FILE) and not force_refresh:
        file_age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(SCHEDULE_FILE))).days
        if file_age_days < 7:
            st.info(f"📁 Using cached schedule (last updated {file_age_days} days ago).")
            return pd.read_csv(SCHEDULE_FILE)

    st.warning("♻️ Schedule missing or outdated — fetching from ESPN...")
    all_games = []

    for week in range(1, 19):
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?year={season}&seasontype=2&week={week}"
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            st.warning(f"Week {week} fetch failed: {e}")
            continue

        for ev in data.get("events", []):
            comp = ev.get("competitions", [{}])[0]
            if not comp.get("competitors"):
                continue

            home, away, home_logo, away_logo = None, None, None, None
            for team in comp["competitors"]:
                if team["homeAway"] == "home":
                    home = team["team"]["abbreviation"]
                    home_logo = team["team"].get("logo")
                else:
                    away = team["team"]["abbreviation"]
                    away_logo = team["team"].get("logo")

            odds = comp.get("odds", [{}])[0] if comp.get("odds") else {}
            spread = odds.get("details", "N/A")

            all_games.append({
                "season": season,
                "week": week,
                "home_team": home,
                "away_team": away,
                "kickoff_et": comp.get("date"),
                "spread": spread,
                "home_logo": home_logo,
                "away_logo": away_logo
            })

    df = pd.DataFrame(all_games)
    if not df.empty:
        df.to_csv(SCHEDULE_FILE, index=False)
        st.success(f"✅ ESPN schedule updated ({len(df)} games).")
    else:
        st.error("❌ ESPN returned no schedule data.")
    return df

# --------------------------------------------------------------
# 🎛️ Sidebar Controls
# --------------------------------------------------------------
st.sidebar.markdown("## 🏈 DJBets NFL Predictor")
season = st.sidebar.selectbox("Season", [2025, 2024, 2023], index=0)
weeks = list(range(1, MAX_WEEKS + 1))
week = st.sidebar.selectbox("Week", weeks, index=0)
if st.sidebar.button("♻️ Refresh ESPN Schedule"):
    scrape_espn_schedule(season, force_refresh=True)
    st.experimental_rerun()

# --------------------------------------------------------------
# 📅 Load Schedule + Model
# --------------------------------------------------------------
sched = scrape_espn_schedule(season)
model = load_or_train_model()

if sched.empty:
    st.error("⚠️ No schedule data found.")
    st.stop()

week_df = sched[sched["week"] == week]
if week_df.empty:
    st.warning(f"⚠️ No games found for Week {week}.")
    st.stop()

st.title(f"🏈 DJBets NFL Predictor — Week {week} ({season})")

# --------------------------------------------------------------
# 📊 Game Cards Display
# --------------------------------------------------------------
for _, row in week_df.iterrows():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if pd.notna(row["away_logo"]):
            st.image(row["away_logo"], width=60)
        st.markdown(f"**{row['away_team']}**")
    with col3:
        if pd.notna(row["home_logo"]):
            st.image(row["home_logo"], width=60)
        st.markdown(f"**{row['home_team']}**")

    with col2:
        elo_diff = np.random.normal(0, 100)  # placeholder until real ELO added
        wx = {"temp_c": np.random.uniform(-5, 25),
              "wind_kph": np.random.uniform(0, 20),
              "precip_prob": np.random.uniform(0, 1)}

        # Ensure feature alignment
        X = pd.DataFrame([{
            "elo_diff": elo_diff,
            "temp_c": wx["temp_c"],
            "wind_kph": wx["wind_kph"],
            "precip_prob": wx["precip_prob"],
        }])[MODEL_FEATURES].astype(float)

        try:
            prob = model.predict_proba(X)[0][1]
        except ValueError:
            st.warning("⚠️ Feature mismatch — retraining model...")
            model = load_or_train_model()
            prob = model.predict_proba(X)[0][1]

        kickoff_str = (
            pd.to_datetime(row["kickoff_et"]).strftime("%a %b %d, %I:%M %p")
            if pd.notna(row["kickoff_et"]) else "TBD"
        )

        st.markdown(f"**Kickoff:** {kickoff_str}")
        st.markdown(f"**Spread:** {row['spread']}")
        st.progress(float(prob))
        st.caption(f"🏠 Home win probability: {prob*100:.1f}%")

# --------------------------------------------------------------
# 📈 Footer & Info
# --------------------------------------------------------------
last_update = datetime.fromtimestamp(os.path.getmtime(SCHEDULE_FILE))
st.sidebar.caption(f"🕒 Last update: {last_update.strftime('%b %d, %Y %I:%M %p')}")
st.sidebar.caption("🔄 Auto-refreshes weekly | Built with ❤️ by DJBets")
