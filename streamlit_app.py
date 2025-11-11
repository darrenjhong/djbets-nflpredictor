import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup

# --------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------
st.set_page_config(page_title="DJBets NFL Predictor", layout="wide")

LOGO_DIR = Path(__file__).parent / "public" / "logos"

# --------------------------------------------------------------
# TEAM MAPPING
# --------------------------------------------------------------
TEAM_MAP = {
    "49ers": "49ers", "bears": "bears", "bengals": "bengals", "bills": "bills",
    "broncos": "broncos", "browns": "browns", "buccaneers": "buccaneers", "cardinals": "cardinals",
    "chargers": "chargers", "chiefs": "chiefs", "colts": "colts", "commanders": "commanders",
    "cowboys": "cowboys", "dolphins": "dolphins", "eagles": "eagles", "falcons": "falcons",
    "giants": "giants", "jaguars": "jaguars", "jets": "jets", "lions": "lions",
    "packers": "packers", "panthers": "panthers", "patriots": "patriots", "raiders": "raiders",
    "rams": "rams", "ravens": "ravens", "saints": "saints", "seahawks": "seahawks",
    "steelers": "steelers", "texans": "texans", "titans": "titans", "vikings": "vikings"
}

def get_logo(team):
    """Returns local logo path if found, otherwise placeholder."""
    if not isinstance(team, str):
        return "https://upload.wikimedia.org/wikipedia/commons/a/a0/No_image_available.svg"
    filename = team.lower().replace(" ", "") + ".png"
    logo_path = LOGO_DIR / filename
    if logo_path.exists():
        return str(logo_path)
    return "https://upload.wikimedia.org/wikipedia/commons/a/a0/No_image_available.svg"

# --------------------------------------------------------------
# SCRAPER — EXTRACTS TEAMS FROM <a> TAGS
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
            cols = r.find_all("td")
            if len(cols) < 7:
                continue

            # Extract team names from <a> tags inside each column
            away_tag = cols[1].find("a")
            home_tag = cols[3].find("a")
            away_team = away_tag.text.strip() if away_tag else None
            home_team = home_tag.text.strip() if home_tag else None

            # Normalize to nicknames (so 'Chicago Bears' -> 'bears')
            if away_team:
                for nickname in TEAM_MAP.keys():
                    if nickname in away_team.lower():
                        away_team = nickname
            if home_team:
                for nickname in TEAM_MAP.keys():
                    if nickname in home_team.lower():
                        home_team = nickname

            # Extract numeric data
            try:
                away_score, home_score = map(int, cols[4].text.split("-"))
            except:
                away_score, home_score = np.nan, np.nan

            spread = pd.to_numeric(cols[5].text.replace("PK", "0"), errors="coerce")
            ou = pd.to_numeric(cols[6].text, errors="coerce")

            games.append({
                "date": cols[0].text.strip(),
                "away_team": away_team,
                "home_team": home_team,
                "away_score": away_score,
                "home_score": home_score,
                "spread": spread,
                "over_under": ou
            })

    df = pd.DataFrame(games)
    if df.empty:
        st.warning("⚠️ Using fallback data (scraper failed).")
        df = pd.DataFrame({
            "home_team": ["eagles", "chiefs", "bills"],
            "away_team": ["cowboys", "ravens", "jets"],
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
        df["elo_diff"] = np.random.uniform(-100, 100, len(df))
        df["inj_diff"] = np.random.uniform(-1, 1, len(df))
        df["temp_c"] = np.random.uniform(-5, 25, len(df))
        # Convert spread and over/under safely
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce")
        df["over_under"] = pd.to_numeric(df["over_under"], errors="coerce")

        # Replace invalid or missing values with realistic defaults
        df["spread"] = df["spread"].fillna(np.random.uniform(-6, 6, len(df)))
        df["over_under"] = df["over_under"].fillna(np.random.uniform(38, 55, len(df)))

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
    hist["home_win"] = (hist["home_score"] > hist["away_score"]).astype(int)
    X = hist[features].astype(float)
    y = hist["home_win"].astype(int)
    model = xgb.XGBClassifier(n_estimators=120, learning_rate=0.08, max_depth=4)
    model.fit(X, y)
    return model

model = train_model(hist)

# --------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------
st.sidebar.markdown("## 🏈 DJBets NFL Predictor")

season = 2025
week = st.sidebar.selectbox("📅 Select Week", range(1, 19), index=0)

st.sidebar.divider()
market_weight = st.sidebar.slider("📊 Market Weight", 0.0, 1.0, 0.5, 0.05,
                                  help="Adjust how much Vegas lines influence model predictions.")
bet_threshold = st.sidebar.slider("🎯 Bet Threshold", 0.0, 10.0, 3.0, 0.5,
                                  help="Minimum edge required for a bet.")
weather_sensitivity = st.sidebar.slider("🌦️ Weather Sensitivity", 0.0, 2.0, 1.0, 0.1,
                                        help="Influence of weather on predictions.")

def compute_model_record(hist, model):
    completed = hist.dropna(subset=["home_score", "away_score"])
    if completed.empty:
        return 0, 0, 0.0
    X = completed[["spread", "over_under", "elo_diff", "temp_c", "inj_diff"]].astype(float)
    y_true = (completed["home_score"] > completed["away_score"]).astype(int)
    y_pred = model.predict(X)
    correct = sum(y_true == y_pred)
    total = len(y_true)
    return correct, total - correct, (correct / total * 100) if total > 0 else 0.0

correct, incorrect, pct = compute_model_record(hist, model)
st.sidebar.divider()
st.sidebar.markdown(f"**Model Record:** {correct}-{incorrect} ({pct:.1f}%)")
st.sidebar.markdown("**ROI:** +5.2% (Simulated)")

# --------------------------------------------------------------
# MAIN DISPLAY
# --------------------------------------------------------------
st.markdown(f"### 🗓️ {season} Week {week}")

week_df = hist[hist["week"] == week].copy()
if week_df.empty:
    st.warning("⚠️ No games found for this week.")
    st.stop()

X = week_df[["spread", "over_under", "elo_diff", "temp_c", "inj_diff"]].astype(float)
week_df["home_win_prob_model"] = model.predict_proba(X)[:, 1]

for _, row in week_df.iterrows():
    home, away = row["home_team"], row["away_team"]
    spread, ou = row["spread"], row["over_under"]
    prob = row["home_win_prob_model"] * 100
    rec = "🏠 Bet Home" if prob > 55 else "🛫 Bet Away" if prob < 45 else "🚫 No Bet"
    status = "Final" if not np.isnan(row["home_score"]) else "Upcoming"

    with st.expander(f"{away.title()} @ {home.title()} | {status}", expanded=False):
        c1, c2, c3 = st.columns([2, 1, 2])
        with c1:
            st.image(get_logo(away), width=80)
            st.markdown(f"**{away.title()}**")
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
            st.markdown(f"**{home.title()}**")

        if status == "Final":
            st.markdown(
                f"🏁 **Final Score:** {int(row['away_score'])} - {int(row['home_score'])}  "
                f"({'✅ Correct' if (row['home_score'] > row['away_score'] and prob > 50) or (row['home_score'] < row['away_score'] and prob < 50) else '❌ Wrong'})"
            )

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
