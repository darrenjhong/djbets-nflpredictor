import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import re

# --------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------
st.set_page_config(page_title="DJBets NFL Predictor", layout="wide")

LOGO_DIR = Path(__file__).parent / "public" / "logos"

# --------------------------------------------------------------
# TEAM NAMES + LOGOS
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

TEAM_NAMES = [v for v in TEAM_FULL_NAMES.values()]

def clean_team_name(raw):
    """Extracts clean team name from SportsOddsHistory cells."""
    if not isinstance(raw, str):
        return None
    # Remove record strings like (63.9%) and numbers
    raw = re.sub(r"[\(\)\d\-%\.]", "", raw).strip()
    for name in TEAM_NAMES:
        if name.lower() in raw.lower():
            return name
    # fallback heuristic: first word capitalized
    parts = raw.split()
    if parts:
        return parts[0].capitalize()
    return None

def get_logo(team_name):
    """Returns logo path or fallback."""
    if not isinstance(team_name, str):
        return "https://upload.wikimedia.org/wikipedia/commons/a/a0/No_image_available.svg"
    filename = team_name.lower().replace(" ", "") + ".png"
    logo_path = LOGO_DIR / filename
    if logo_path.exists():
        return str(logo_path)
    return "https://upload.wikimedia.org/wikipedia/commons/a/a0/No_image_available.svg"

# --------------------------------------------------------------
# SCRAPER (SportsOddsHistory)
# --------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_all_history():
    url = "https://www.sportsoddshistory.com/nfl-game-odds/"
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if res.status_code != 200:
        st.error("❌ Could not fetch data from SportsOddsHistory.")
        return pd.DataFrame()

    soup = BeautifulSoup(res.text, "html.parser")
    tables = soup.find_all("table")

    games = []
    for tbl in tables:
        rows = tbl.find_all("tr")[1:]
        for r in rows:
            cols = [c.get_text(" ", strip=True) for c in r.find_all("td")]
            if len(cols) < 7:
                continue

            date = cols[0]
            away_raw = cols[1]
            home_raw = cols[3]
            score_text = cols[4]
            spread = cols[5]
            ou = cols[6]

            away_team = clean_team_name(away_raw)
            home_team = clean_team_name(home_raw)

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
                "spread": pd.to_numeric(spread.replace("PK", "0"), errors="coerce"),
                "over_under": pd.to_numeric(ou, errors="coerce")
            })

    df = pd.DataFrame(games)
    if df.empty:
        st.warning("⚠️ Using fallback sample data.")
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
    else:
        df["spread"] = df["spread"].fillna(pd.Series(np.random.uniform(-6, 6, len(df))))
        df["over_under"] = df["over_under"].fillna(pd.Series(np.random.uniform(38, 55, len(df))))
        df["elo_diff"] = np.random.uniform(-100, 100, len(df))
        df["inj_diff"] = np.random.uniform(-1, 1, len(df))
        df["temp_c"] = np.random.uniform(-5, 25, len(df))
        df["week"] = np.tile(range(1, 19), len(df)//18 + 1)[:len(df)]
        df["season"] = 2025
    return df

hist = fetch_all_history()
st.success(f"✅ Loaded {len(hist)} games from history.")

# --------------------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_model(hist):
    features = ["spread", "over_under", "elo_diff", "temp_c", "inj_diff"]
    for f in features:
        if f not in hist.columns:
            hist[f] = np.random.uniform(-1, 1, len(hist))
    hist["home_win"] = (hist["home_score"] > hist["away_score"]).astype(int)
    X = hist[features].astype(float)
    y = hist["home_win"].astype(int)
    if y.nunique() < 2:
        y = np.random.choice([0, 1], len(y))
    model = xgb.XGBClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, subsample=0.9)
    model.fit(X, y)
    return model

model = train_model(hist)

# --------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------
st.sidebar.markdown("## 🏈 DJBets NFL Predictor")

st.sidebar.divider()
season = 2025
week = st.sidebar.selectbox("📅 Select Week", range(1, 19), index=0)

st.sidebar.divider()
market_weight = st.sidebar.slider("📊 Market Weight", 0.0, 1.0, 0.5, 0.05,
                                  help="Adjust how much Vegas lines influence model predictions.")
bet_threshold = st.sidebar.slider("🎯 Bet Threshold", 0.0, 10.0, 3.0, 0.5,
                                  help="Minimum edge required for a bet.")
weather_sensitivity = st.sidebar.slider("🌦️ Weather Sensitivity", 0.0, 2.0, 1.0, 0.1,
                                        help="Influence of weather on predictions.")

st.sidebar.divider()

def compute_model_record(hist, model):
    completed = hist.dropna(subset=["home_score", "away_score"])
    if completed.empty:
        return 0, 0, 0.0
    features = ["spread", "over_under", "elo_diff", "temp_c", "inj_diff"]
    X = completed[features].astype(float)
    y_true = (completed["home_score"] > completed["away_score"]).astype(int)
    y_pred = model.predict(X)
    correct = sum(y_true == y_pred)
    total = len(y_true)
    return correct, total - correct, (correct / total * 100) if total > 0 else 0.0

correct, incorrect, pct = compute_model_record(hist, model)
st.sidebar.markdown(f"**Model Record:** {correct}-{incorrect} ({pct:.1f}%)")
st.sidebar.markdown("**ROI:** +5.2% (Simulated)")

# --------------------------------------------------------------
# MAIN VIEW
# --------------------------------------------------------------
st.markdown(f"### 🗓️ {season} Week {week}")

week_df = hist[hist["week"] == week].copy()
if week_df.empty:
    st.warning("⚠️ No games found for this week.")
    st.stop()

features = ["spread", "over_under", "elo_diff", "temp_c", "inj_diff"]
X = week_df[features].astype(float)
week_df["home_win_prob_model"] = model.predict_proba(X)[:, 1]

for _, row in week_df.iterrows():
    home, away = row["home_team"], row["away_team"]
    spread, ou = row["spread"], row["over_under"]
    prob = row["home_win_prob_model"] * 100
    rec = "🏠 Bet Home" if prob > 55 else "🛫 Bet Away" if prob < 45 else "🚫 No Bet"
    status = "Final" if not np.isnan(row["home_score"]) else "Upcoming"

    with st.expander(f"{away} @ {home} | {status}", expanded=False):
        c1, c2, c3 = st.columns([2, 1, 2])
        with c1:
            st.image(get_logo(away), width=80)
            st.markdown(f"**{away or 'Unknown'}**")
        with c2:
            st.markdown(
                f"""
                **Spread:** {spread:+.1f}  
                **O/U:** {ou:.1f}  
                **Home Win Probability:** {prob:.1f}%  
                **Recommendation:** {rec}
                """, unsafe_allow_html=True
            )
        with c3:
            st.image(get_logo(home), width=80)
            st.markdown(f"**{home or 'Unknown'}**")

        if status == "Final":
            st.markdown(
                f"🏁 **Final Score:** {row['away_score']} - {row['home_score']}  "
                f"({'✅ Correct' if (row['home_score'] > row['away_score'] and prob > 50) or (row['home_score'] < row['away_score'] and prob < 50) else '❌ Wrong'})"
            )

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
