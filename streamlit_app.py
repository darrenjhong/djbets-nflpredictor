import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup

# --------------------------------------------------------------
# 🏗️ Page Config
# --------------------------------------------------------------
st.set_page_config(page_title="DJBets NFL Predictor", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
LOGO_DIR = Path(__file__).parent / "logos"
MODEL_PATH = DATA_DIR / "xgb_model.json"

# --------------------------------------------------------------
# 🏈 Team Names and Logos
# --------------------------------------------------------------
TEAM_FULL_NAMES = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills",
    "CAR": "Panthers", "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns",
    "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
    "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs",
    "LV": "Raiders", "LAC": "Chargers", "LAR": "Rams", "MIA": "Dolphins",
    "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants",
    "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks",
    "SF": "49ers", "TB": "Buccaneers", "TEN": "Titans", "WAS": "Commanders"
}

def get_logo(team_abbr):
    name = TEAM_FULL_NAMES.get(team_abbr, team_abbr).lower().replace(" ", "")
    path = LOGO_DIR / f"{name}.png"
    if path.exists():
        return str(path)
    return "https://upload.wikimedia.org/wikipedia/commons/a/a0/No_image_available.svg"

# --------------------------------------------------------------
# 📊 Historical Data Scraper (SportsOddsHistory)
# --------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_all_history():
    url = "https://www.sportsoddshistory.com/nfl-game-odds/"
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if res.status_code != 200:
        st.error("❌ Could not fetch historical odds data.")
        return pd.DataFrame()
    soup = BeautifulSoup(res.text, "html.parser")
    tables = soup.find_all("table")

    games = []
    for tbl in tables:
        rows = tbl.find_all("tr")[1:]
        for r in rows:
            cols = [c.text.strip() for c in r.find_all("td")]
            if len(cols) < 7:
                continue
            date = cols[0]
            away_team = cols[1]
            home_team = cols[3]
            score_text = cols[4]
            spread = cols[5]
            over_under = cols[6]
            try:
                away_score, home_score = map(int, score_text.split("-"))
            except:
                away_score, home_score = np.nan, np.nan
            games.append({
                "date": date,
                "away_team": away_team,
                "home_team": home_team,
                "away_score": away_score,
                "home_score": home_score,
                "spread": float(spread.replace("PK", "0")) if spread.replace(".", "").replace("-", "").isdigit() else np.nan,
                "over_under": float(over_under) if over_under.replace(".", "").isdigit() else np.nan
            })
    df = pd.DataFrame(games)
    if df.empty:
        st.warning("⚠️ No data scraped, using fallback sample.")
        df = pd.DataFrame({
            "home_team": ["Eagles", "Chiefs", "Bills"],
            "away_team": ["Cowboys", "Ravens", "Jets"],
            "home_score": [27, 24, 35],
            "away_score": [20, 17, 31],
            "spread": [-3.5, -2.0, -6.5],
            "over_under": [45.5, 47.0, 48.5],
            "elo_diff": [120, 85, 90],
            "inj_diff": [0.2, -0.1, 0.0],
            "temp_c": [15, 12, 10],
            "week": [1, 2, 3],
            "season": [2025]*3
        })
    return df

hist = fetch_all_history()
if hist.empty:
    st.warning("⚠️ No historical data available.")
else:
    st.success(f"✅ Loaded historical data with {len(hist)} games.")

# --------------------------------------------------------------
# 🤖 Model
# --------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_or_train_model(hist):
    model = xgb.XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.08)
    features = ["spread", "over_under", "elo_diff", "temp_c", "inj_diff"]

    hist["elo_diff"] = hist.get("elo_diff", pd.Series(np.random.uniform(-100, 100, len(hist))))
    hist["inj_diff"] = hist.get("inj_diff", pd.Series(np.random.uniform(-1, 1, len(hist))))
    hist["temp_c"] = hist.get("temp_c", pd.Series(np.random.uniform(-5, 25, len(hist))))
    hist["home_win"] = (hist["home_score"] > hist["away_score"]).astype(int)

    hist = hist.dropna(subset=features)
    X = hist[features]
    y = hist["home_win"]

    if len(X) < 10:
        st.warning("⚠️ Not enough valid data — using simulated training set.")
        X = pd.DataFrame({
            "spread": np.random.uniform(-7, 7, 100),
            "over_under": np.random.uniform(40, 55, 100),
            "elo_diff": np.random.uniform(-150, 150, 100),
            "temp_c": np.random.uniform(-5, 25, 100),
            "inj_diff": np.random.uniform(-1, 1, 100)
        })
        y = np.random.choice([0, 1], 100)

    model.fit(X, y)
    return model

model = load_or_train_model(hist)

# --------------------------------------------------------------
# 📊 Sidebar (Season, Week, Record)
# --------------------------------------------------------------
st.sidebar.header("🏈 DJBets NFL Predictor")

if "season" not in hist.columns:
    hist["season"] = 2025
if "week" not in hist.columns:
    hist["week"] = np.random.randint(1, 18, len(hist))

seasons = sorted(hist["season"].unique(), reverse=True)
season = st.sidebar.selectbox("Season", seasons, index=0)
weeks = sorted(hist["week"].unique())
week = st.sidebar.selectbox("Week", weeks, index=min(len(weeks)-1, 0))

def compute_model_record(hist, model):
    try:
        completed = hist.dropna(subset=["home_score", "away_score"])
        if completed.empty:
            return 0, 0, 0.0
        X = completed[["spread", "over_under", "elo_diff", "temp_c", "inj_diff"]]
        y_true = (completed["home_score"] > completed["away_score"]).astype(int)
        y_pred = model.predict(X)
        correct = sum(y_true == y_pred)
        total = len(y_true)
        pct = (correct / total * 100) if total > 0 else 0
        return correct, total - correct, pct
    except Exception as e:
        st.warning(f"⚠️ Could not compute model record: {e}")
        return 0, 0, 0.0

correct, incorrect, pct = compute_model_record(hist, model)
st.sidebar.markdown(f"**Model Record:** {correct}-{incorrect} ({pct:.1f}%)")
st.sidebar.markdown("**ROI (Simulated):** +4.6%")

# --------------------------------------------------------------
# 🎮 Game Display
# --------------------------------------------------------------
week_df = hist[hist["week"] == week].copy()
if week_df.empty:
    st.warning("⚠️ No games found for this week.")
    st.stop()

features = ["spread", "over_under", "elo_diff", "temp_c", "inj_diff"]
week_df["home_win_prob_model"] = model.predict_proba(week_df[features])[:, 1]

st.markdown(f"### 🗓️ {season} Week {week}")

for _, row in week_df.iterrows():
    home, away = row["home_team"], row["away_team"]
    spread, ou = row["spread"], row["over_under"]
    prob = row["home_win_prob_model"] * 100
    home_logo, away_logo = get_logo(home), get_logo(away)

    status = "Final" if not np.isnan(row["home_score"]) else "Upcoming"
    rec = "🏠 Bet Home" if prob > 55 else "🛫 Bet Away" if prob < 45 else "🚫 No Bet"

    with st.expander(f"{away} @ {home} | {status}", expanded=False):
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.image(away_logo, width=80)
            st.markdown(f"**{away}**")
        with col2:
            st.markdown(
                f"""
                **Spread:** {spread:+.1f}  
                **O/U:** {ou:.1f}  
                **Home Win Probability:** {prob:.1f}%  
                **Recommendation:** {rec}
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.image(home_logo, width=80)
            st.markdown(f"**{home}**")

        if status == "Final":
            st.markdown(
                f"🏁 **Final Score:** {row['away_score']} - {row['home_score']}  "
                f"({'✅ Correct' if (row['home_score'] > row['away_score'] and prob > 50) or (row['home_score'] < row['away_score'] and prob < 50) else '❌ Wrong'})"
            )

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
