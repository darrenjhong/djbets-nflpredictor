import streamlit as st
import pandas as pd
import numpy as np
import os, requests, xgboost as xgb
from datetime import datetime

# --------------------------------------------------------------
# 📂 Paths and Config
# --------------------------------------------------------------
DATA_DIR = "data"
SCHEDULE_FILE = os.path.join(DATA_DIR, "schedule.csv")
HISTORICAL_FILE = os.path.join(DATA_DIR, "historical.csv")
ELO_FILE = os.path.join(DATA_DIR, "elo_latest.csv")
MODEL_FILE = os.path.join(DATA_DIR, "xgb_model.json")
MAX_WEEKS = 18


# --------------------------------------------------------------
# 🏈 ESPN Scraper for Schedule
# --------------------------------------------------------------
from datetime import datetime, timedelta

def scrape_espn_schedule(season: int, force_refresh: bool = False):
    """
    Scrape full NFL season (weeks 1–18) from ESPN, auto-refresh weekly.
    If a local schedule.csv exists and is less than 7 days old, skip re-fetch.
    """

    os.makedirs(DATA_DIR, exist_ok=True)

    # ----------------------------------------------------------
    # 🧭 Step 1: Check if schedule exists and is fresh
    # ----------------------------------------------------------
    if os.path.exists(SCHEDULE_FILE) and not force_refresh:
        file_age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(SCHEDULE_FILE))).days
        if file_age_days < 7:
            st.info(f"📁 Using cached schedule (last updated {file_age_days} days ago).")
            return pd.read_csv(SCHEDULE_FILE)

    # ----------------------------------------------------------
    # 🏈 Step 2: Fetch from ESPN if missing or stale
    # ----------------------------------------------------------
    st.warning("♻️ Schedule missing or outdated — fetching from ESPN...")

    all_games = []
    for week in range(1, 19):
        url = (
            f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?"
            f"year={season}&seasontype=2&week={week}"
        )
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

    # ----------------------------------------------------------
    # 💾 Step 3: Save to CSV
    # ----------------------------------------------------------
    df = pd.DataFrame(all_games)
    if not df.empty:
        df.to_csv(SCHEDULE_FILE, index=False)
        st.success(f"✅ ESPN schedule updated ({len(df)} games total).")
    else:
        st.error("❌ ESPN returned no schedule data.")
    return df



# --------------------------------------------------------------
# 🌤️ Weather Scraper via wttr.in
# --------------------------------------------------------------
def get_weather_for_team(team_abbr):
    """Scrape basic weather (temp, wind, precip) using wttr.in."""
    try:
        resp = requests.get(f"https://wttr.in/{team_abbr}?format=j1", timeout=5)
        data = resp.json()
        current = data["current_condition"][0]
        return {
            "temp_c": float(current["temp_C"]),
            "wind_kph": float(current["windspeedKmph"]),
            "precip_prob": float(current["precipMM"])
        }
    except Exception:
        return {"temp_c": np.nan, "wind_kph": np.nan, "precip_prob": np.nan}


# --------------------------------------------------------------
# 📈 ELO Loader (FiveThirtyEight)
# --------------------------------------------------------------
def load_elo_ratings():
    """Download latest NFL Elo ratings from FiveThirtyEight."""
    url = "https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv"
    try:
        df = pd.read_csv(url)
        df_latest = df.sort_values("date").groupby("team1").tail(1)
        df_latest = df_latest[["team1", "elo1_pre"]].rename(columns={"team1": "team", "elo1_pre": "elo"})
        df_latest.to_csv(ELO_FILE, index=False)
        return df_latest
    except Exception:
        if os.path.exists(ELO_FILE):
            return pd.read_csv(ELO_FILE)
        else:
            return pd.DataFrame(columns=["team", "elo"])


# --------------------------------------------------------------
# 🧠 Model Loader/Trainer
# --------------------------------------------------------------
@st.cache_resource
def load_or_train_model():
    """Safely load or train an XGBoost model."""
    if os.path.exists(MODEL_FILE):
        try:
            model = xgb.XGBClassifier()
            model.load_model(MODEL_FILE)
            return model
        except Exception:
            st.warning("⚠️ Model load failed, retraining...")

    df = pd.DataFrame({
        "elo_diff": np.random.normal(0, 100, 200),
        "temp_c": np.random.normal(15, 10, 200),
        "wind_kph": np.random.uniform(0, 30, 200),
        "precip_prob": np.random.uniform(0, 100, 200),
        "home_win": np.random.randint(0, 2, 200)
    })
    X, y = df[["elo_diff", "temp_c", "wind_kph", "precip_prob"]], df["home_win"]
    model = xgb.XGBClassifier(eval_metric="logloss", n_estimators=150, max_depth=4)
    model.fit(X, y)
    model.save_model(MODEL_FILE)
    return model


# --------------------------------------------------------------
# 🎛️ Sidebar Controls
# --------------------------------------------------------------
st.sidebar.markdown("## 🏈 DJBets NFL Predictor")
season = st.sidebar.selectbox("Season", [2025, 2024, 2023], index=0)
week = st.sidebar.selectbox("Week", list(range(1, MAX_WEEKS + 1)), index=0)
if st.sidebar.button("🔄 Retrain model now"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.experimental_rerun()

if st.sidebar.button("♻️ Refresh ESPN Schedule"):
    with st.spinner("Refreshing schedule from ESPN..."):
        scrape_espn_schedule(season, force_refresh=True)
        st.experimental_rerun()


# --------------------------------------------------------------
# 🚀 Main
# --------------------------------------------------------------
st.title("🏈 DJBets NFL Predictor — Real-Time ESPN + Weather + Elo")

sched = scrape_espn_schedule(season)
elo = load_elo_ratings()
model = load_or_train_model()

if not sched.empty:
    st.success(f"✅ Loaded schedule for {season}. Showing Week {week}.")
    week_df = sched[sched["week"] == week].copy()

    if week_df.empty:
        st.warning("⚠️ No games found for this week.")
    else:
        games_out = []
        for _, row in week_df.iterrows():
            home_elo = elo.loc[elo["team"] == row["home_team"], "elo"].values
            away_elo = elo.loc[elo["team"] == row["away_team"], "elo"].values
            elo_diff = (home_elo[0] - away_elo[0]) if len(home_elo) and len(away_elo) else np.random.randint(-50, 50)
            wx = get_weather_for_team(row["home_team"])

            X = pd.DataFrame([{
                "elo_diff": elo_diff,
                "temp_c": wx["temp_c"],
                "wind_kph": wx["wind_kph"],
                "precip_prob": wx["precip_prob"]
            }])
            prob = model.predict_proba(X)[0][1]
            winner = row["home_team"] if prob >= 0.5 else row["away_team"]

            games_out.append({
                "matchup": f"{row['away_team']} @ {row['home_team']}",
                "kickoff": row["kickoff_et"],
                "spread": row["spread"],
                "predicted_winner": winner,
                "home_win_prob": f"{prob*100:.1f}%",
                **wx
            })

        df_display = pd.DataFrame(games_out)
        st.dataframe(df_display)

else:
    st.error("❌ No schedule data could be loaded.")
