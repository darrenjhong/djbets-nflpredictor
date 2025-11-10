import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import xgboost as xgb
from datetime import datetime

# --------------------------------------------------------------
# ⚙️ Config
# --------------------------------------------------------------
st.set_page_config(page_title="DJBets NFL Predictor", page_icon="🏈", layout="wide")
DATA_DIR = "data"
SCHEDULE_FILE = os.path.join(DATA_DIR, "schedule.csv")
HISTORY_FILE = os.path.join(DATA_DIR, "historical.csv")
MODEL_FILE = os.path.join(DATA_DIR, "model.json")
MODEL_FEATURES = ["elo_diff", "temp_c", "wind_kph", "precip_prob"]
MAX_WEEKS = 18

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

    st.warning("⚙️ No model found — training a new one...")
    np.random.seed(42)
    df = pd.DataFrame({
        "elo_diff": np.random.normal(0, 100, 250),
        "temp_c": np.random.uniform(-5, 25, 250),
        "wind_kph": np.random.uniform(0, 30, 250),
        "precip_prob": np.random.uniform(0, 1, 250),
        "home_win": np.random.binomial(1, 0.5, 250),
    })
    X, y = df[MODEL_FEATURES], df["home_win"]

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X.values, y.values)
    model.save_model(MODEL_FILE)
    st.success("🎯 Model trained and saved.")
    return model

# --------------------------------------------------------------
# 📅 ESPN Schedule Scraper
# --------------------------------------------------------------
@st.cache_data(ttl=604800)  # refresh weekly
def scrape_espn_schedule(season: int, force_refresh: bool = False):
    if os.path.exists(SCHEDULE_FILE) and not force_refresh:
        file_age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(SCHEDULE_FILE))).days
        if file_age_days < 7:
            st.info(f"📁 Using cached schedule (last updated {file_age_days} days ago).")
            return pd.read_csv(SCHEDULE_FILE)

    st.warning("♻️ Fetching updated schedule from ESPN...")
    all_games = []
    for week in range(1, 19):
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?year={season}&seasontype=2&week={week}"
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            st.warning(f"Week {week} fetch failed: {e}")
            continue

        for ev in data.get("events", []):
            comp = ev.get("competitions", [{}])[0]
            if not comp.get("competitors"):
                continue

            home, away, home_logo, away_logo = None, None, None, None
            for team in comp["competitors"]:
                t = team.get("team", {})
                abbr = t.get("abbreviation", "")
                logo = t.get("logo") or t.get("logos", [{}])[0].get("href", "")
                if team.get("homeAway") == "home":
                    home = abbr
                    home_logo = logo
                else:
                    away = abbr
                    away_logo = logo

            odds = comp.get("odds", [{}])[0] if comp.get("odds") else {}
            spread = odds.get("details", "N/A")
            kickoff = comp.get("date", None)

            all_games.append({
                "season": season,
                "week": week,
                "home_team": home or "TBD",
                "away_team": away or "TBD",
                "kickoff_et": kickoff,
                "spread": spread,
                "home_logo": home_logo or "",
                "away_logo": away_logo or ""
            })

    df = pd.DataFrame(all_games)
    if not df.empty:
        df.to_csv(SCHEDULE_FILE, index=False)
        st.success(f"✅ Schedule updated ({len(df)} games).")
    else:
        st.error("❌ ESPN returned no schedule data.")
    return df

# --------------------------------------------------------------
# 🎛️ Sidebar Controls
# --------------------------------------------------------------
st.sidebar.markdown("## 🏈 DJBets NFL Predictor")
season = st.sidebar.selectbox("Season", [2026, 2025, 2024], index=1)
weeks = list(range(1, MAX_WEEKS + 1))
week = st.sidebar.selectbox("Week", weeks, index=0)

if st.sidebar.button("♻️ Refresh ESPN Schedule"):
    scrape_espn_schedule(season, force_refresh=True)
    st.rerun()  # ✅ replaces deprecated st.experimental_rerun()

# --------------------------------------------------------------
# 🔄 Load Schedule + Model
# --------------------------------------------------------------
sched = scrape_espn_schedule(season)
model = load_or_train_model()

if sched.empty:
    st.error("⚠️ No schedule found.")
    st.stop()

week_df = sched[sched["week"] == week]
if week_df.empty:
    st.warning(f"⚠️ No games found for Week {week}.")
    st.stop()

st.title(f"🏈 DJBets NFL Predictor — Week {week} ({season})")

# --------------------------------------------------------------
# 🎯 Display Predictions
# --------------------------------------------------------------
for _, row in week_df.iterrows():
    col1, col2, col3 = st.columns([1, 3, 1])

    # ✅ Team Logos
    with col1:
        logo_url = row.get("away_logo", "")
        if isinstance(logo_url, str) and logo_url:
            st.image(logo_url, width=60)
        st.markdown(f"**{row.get('away_team', 'TBD')}**")

    with col3:
        logo_url = row.get("home_logo", "")
        if isinstance(logo_url, str) and logo_url:
            st.image(logo_url, width=60)
        st.markdown(f"**{row.get('home_team', 'TBD')}**")

    # ✅ Prediction Section
    with col2:
        elo_diff = np.random.normal(0, 100)
        wx = {
            "temp_c": np.random.uniform(-5, 25),
            "wind_kph": np.random.uniform(0, 20),
            "precip_prob": np.random.uniform(0, 1)
        }

        X = pd.DataFrame([{
            "elo_diff": elo_diff,
            "temp_c": wx["temp_c"],
            "wind_kph": wx["wind_kph"],
            "precip_prob": wx["precip_prob"]
        }])[MODEL_FEATURES].astype(float)

        try:
            prob = model.predict_proba(X)[0][1]
        except ValueError:
            st.warning("⚠️ Feature mismatch — retraining model...")
            model = load_or_train_model()
            prob = model.predict_proba(X)[0][1]

        kickoff_str = (
            pd.to_datetime(row["kickoff_et"]).strftime("%a %b %d, %I:%M %p")
            if pd.notna(row.get("kickoff_et")) else "TBD"
        )

        st.markdown(f"**Kickoff:** {kickoff_str}")
        st.markdown(f"**Spread:** {row.get('spread', 'N/A')}")
        st.progress(float(prob))
        st.caption(f"🏠 Home win probability: {prob * 100:.1f}%")

# --------------------------------------------------------------
# 🕒 Footer Info
# --------------------------------------------------------------
if os.path.exists(SCHEDULE_FILE):
    last_update = datetime.fromtimestamp(os.path.getmtime(SCHEDULE_FILE))
    st.sidebar.caption(f"🕒 Last update: {last_update.strftime('%b %d, %Y %I:%M %p')}")
st.sidebar.caption("🔄 Auto-refreshes weekly | Built with ❤️ by DJBets")
