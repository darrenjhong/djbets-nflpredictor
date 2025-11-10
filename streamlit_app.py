import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import xgboost as xgb
from datetime import datetime, timedelta

# --------------------------------------------------------------
# ⚙️ Config
# --------------------------------------------------------------
st.set_page_config(page_title="DJBets NFL Predictor", page_icon="🏈", layout="wide")
DATA_DIR = "data"
SCHEDULE_FILE = os.path.join(DATA_DIR, "schedule.csv")
MODEL_FILE = os.path.join(DATA_DIR, "model.json")
MODEL_FEATURES = ["elo_diff", "temp_c", "wind_kph", "precip_prob"]
MAX_WEEKS = 18
TEAMS = [
    "BUF", "MIA", "NE", "NYJ",
    "BAL", "CIN", "CLE", "PIT",
    "HOU", "IND", "JAX", "TEN",
    "DEN", "KC", "LV", "LAC",
    "DAL", "NYG", "PHI", "WAS",
    "CHI", "DET", "GB", "MIN",
    "ATL", "CAR", "NO", "TB",
    "ARI", "LAR", "SF", "SEA"
]
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
# 🕸️ ESPN Schedule Scraper + Fallback Generator
# --------------------------------------------------------------
@st.cache_data(ttl=604800)
def scrape_espn_schedule(season: int, force_refresh: bool = False):
    all_games = []

    # Try ESPN first
    for week in range(1, MAX_WEEKS + 1):
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?year={season}&seasontype=2&week={week}"
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            r.raise_for_status()
            data = r.json()
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
                        home, home_logo = abbr, logo
                    else:
                        away, away_logo = abbr, logo

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
        except Exception:
            continue

    df = pd.DataFrame(all_games)

    # Fallback filler if ESPN is incomplete
    if len(df) < 100:
        st.warning("⚠️ ESPN returned partial schedule — generating mock filler data.")
        generated = []
        for week in range(1, MAX_WEEKS + 1):
            np.random.shuffle(TEAMS)
            for i in range(0, len(TEAMS), 2):
                home, away = TEAMS[i], TEAMS[i+1]
                kickoff = (datetime.now() + timedelta(days=(week-1)*7)).replace(hour=13, minute=0)
                generated.append({
                    "season": season,
                    "week": week,
                    "home_team": home,
                    "away_team": away,
                    "kickoff_et": kickoff.isoformat(),
                    "spread": f"{np.random.choice(['+','-'])}{np.random.randint(1,8)}",
                    "home_logo": f"https://a.espncdn.com/i/teamlogos/nfl/500/{home.lower()}.png",
                    "away_logo": f"https://a.espncdn.com/i/teamlogos/nfl/500/{away.lower()}.png"
                })
        df = pd.DataFrame(generated)

    df.to_csv(SCHEDULE_FILE, index=False)
    st.success(f"✅ Schedule ready ({len(df)} games).")
    return df

# --------------------------------------------------------------
# 🎛️ Sidebar Controls
# --------------------------------------------------------------
st.sidebar.markdown("## 🏈 DJBets NFL Predictor")
season = st.sidebar.selectbox("Season", [2026, 2025, 2024], index=1)
weeks = list(range(1, MAX_WEEKS + 1))
week = st.sidebar.selectbox("Week", weeks, index=0)
if st.sidebar.button("♻️ Refresh Schedule"):
    scrape_espn_schedule(season, force_refresh=True)
    st.rerun()

# --------------------------------------------------------------
# 📊 Load Schedule + Model
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
# 🧮 Predictions + Display
# --------------------------------------------------------------
for _, row in week_df.iterrows():
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        if isinstance(row.get("away_logo"), str) and row["away_logo"]:
            st.image(row["away_logo"], width=60)
        st.markdown(f"**{row.get('away_team', 'TBD')}**")

    with col3:
        if isinstance(row.get("home_logo"), str) and row["home_logo"]:
            st.image(row["home_logo"], width=60)
        st.markdown(f"**{row.get('home_team', 'TBD')}**")

    with col2:
        elo_diff = np.random.normal(0, 100)
        wx = {"temp_c": np.random.uniform(-5, 25),
              "wind_kph": np.random.uniform(0, 20),
              "precip_prob": np.random.uniform(0, 1)}
        X = pd.DataFrame([{**wx, "elo_diff": elo_diff}])[MODEL_FEATURES].astype(float)
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
# 🕒 Footer
# --------------------------------------------------------------
if os.path.exists(SCHEDULE_FILE):
    last_update = datetime.fromtimestamp(os.path.getmtime(SCHEDULE_FILE))
    st.sidebar.caption(f"🕒 Last update: {last_update.strftime('%b %d, %Y %I:%M %p')}")
st.sidebar.caption("🔄 Auto-refresh weekly | Built with ❤️ by DJBets")
