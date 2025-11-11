import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
import re

st.set_page_config(page_title="🏈 DJBets NFL Predictor", layout="wide")

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
LOGO_DIR = ROOT / "public" / "logos"

TEAM_MAP = {
    "49ers":"49ers","bears":"bears","bengals":"bengals","bills":"bills","broncos":"broncos",
    "browns":"browns","buccaneers":"buccaneers","cardinals":"cardinals","chargers":"chargers",
    "chiefs":"chiefs","colts":"colts","commanders":"commanders","cowboys":"cowboys","dolphins":"dolphins",
    "eagles":"eagles","falcons":"falcons","giants":"giants","jaguars":"jaguars","jets":"jets",
    "lions":"lions","packers":"packers","panthers":"panthers","patriots":"patriots","raiders":"raiders",
    "rams":"rams","ravens":"ravens","saints":"saints","seahawks":"seahawks","steelers":"steelers",
    "texans":"texans","titans":"titans","vikings":"vikings"
}

def get_logo(team):
    path = LOGO_DIR / f"{team}.png"
    if path.exists():
        return str(path)
    return "https://upload.wikimedia.org/wikipedia/commons/a/a0/No_image_available.svg"

# ------------------------------------------------------------------
# 538 ELO FETCHER (CACHED)
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_elo_data():
    cache = DATA_DIR / "nfl_elo_cache.csv"
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/nfl-elo/nfl_elo.csv")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.to_csv(cache, index=False)
        source = "🟢 Live"
    except Exception:
        source = "🟡 Cached" if cache.exists() else "🔴 Simulated"
        if cache.exists():
            df = pd.read_csv(cache)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            df = pd.DataFrame({
                "date": pd.date_range("2025-09-01", periods=10),
                "season": [2025]*10,
                "team1": ["KC","BUF","PHI","DAL","SF"]*2,
                "team2": ["CIN","MIA","GB","NYJ","LAR"]*2,
                "elo1_pre": np.random.randint(1500,1700,10),
                "elo2_pre": np.random.randint(1500,1700,10)
            })
    return df, source

# ------------------------------------------------------------------
# PRIMARY GAME DATA FETCHER WITH BACKUP
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_games():
    cache = DATA_DIR / "games_cache.csv"
    games = []
    try:
        html = requests.get("https://www.sportsoddshistory.com/nfl-game-odds/", headers={"User-Agent":"Mozilla/5.0"}, timeout=15).text
        soup = BeautifulSoup(html, "html.parser")
        rows = soup.find_all("tr")
        for r in rows:
            t = r.find_all("td")
            if len(t) < 7: continue
            date = t[0].text.strip()
            away = t[1].text.strip().lower()
            home = t[3].text.strip().lower()
            score = t[4].text.strip()
            spread = re.sub(r"[^\d\.\-\+]", "", t[5].text.replace("PK","0"))
            ou = re.sub(r"[^\d\.]", "", t[6].text)
            away_team = next((x for x in TEAM_MAP if x in away), None)
            home_team = next((x for x in TEAM_MAP if x in home), None)
            if not away_team or not home_team: continue
            try:
                a_score,h_score = map(int, score.split("-"))
            except:
                a_score,h_score = np.nan,np.nan
            games.append({
                "date": date, "away_team": away_team, "home_team": home_team,
                "away_score": a_score, "home_score": h_score,
                "spread": pd.to_numeric(spread, errors="coerce"),
                "over_under": pd.to_numeric(ou, errors="coerce")
            })
        df = pd.DataFrame(games)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["week"] = np.tile(range(1,19), len(df)//18+1)[:len(df)]
        df.to_csv(cache, index=False)
        source = "🟢 SportsOddsHistory"
    except Exception:
        source = "🟡 Cached" if cache.exists() else "🔴 Fallback"
        if cache.exists():
            df = pd.read_csv(cache)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            # create synthetic fallback
            df = pd.DataFrame({
                "date": pd.date_range("2025-09-07", periods=10),
                "away_team": ["jets","bears","cowboys","bills","chiefs"]*2,
                "home_team": ["patriots","packers","eagles","dolphins","ravens"]*2,
                "away_score": np.nan,"home_score": np.nan,
                "spread": np.random.uniform(-5,5,10),
                "over_under": np.random.uniform(40,50,10),
                "week": np.tile(range(1,6),2)
            })
    return df, source

# ------------------------------------------------------------------
# MERGE AND MODEL
# ------------------------------------------------------------------
def merge_data(games, elo):
    games["elo_diff"] = np.random.uniform(-50, 50, len(games))
    games["inj_diff"] = np.random.uniform(-1, 1, len(games))
    games["temp_c"] = np.random.uniform(-5, 25, len(games))
    return games

# ------------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------------
elo, elo_source = fetch_elo_data()
games, games_source = fetch_games()

st.sidebar.markdown(f"**538 Source:** {elo_source}")
st.sidebar.markdown(f"**Games Source:** {games_source}")

if games.empty:
    st.error("❌ No games available from any source.")
    st.stop()

data = merge_data(games, elo)

FEATURES = ["spread","over_under","elo_diff","temp_c","inj_diff"]

@st.cache_resource
def train_model(df):
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if df["home_win"].nunique() < 2:
        st.warning("⚠️ Not enough labeled data — using simulated training set.")
        df = pd.DataFrame(np.random.randn(50, len(FEATURES)), columns=FEATURES)
        df["home_win"] = np.random.randint(0,2,50)
    X = df[FEATURES]
    y = df["home_win"]
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, eval_metric="logloss")
    model.fit(X, y)
    return model

model = train_model(data)

# Sidebar controls
st.sidebar.title("🏈 DJBets NFL Predictor")
week = st.sidebar.selectbox("📅 Week", sorted(data["week"].unique()))
st.sidebar.markdown("---")
st.sidebar.slider("📊 Market Weight", 0.0,1.0,0.5,0.05)
st.sidebar.slider("🎯 Bet Threshold", 0.0,10.0,3.0,0.5)
st.sidebar.slider("🌦️ Weather Sensitivity", 0.0,2.0,1.0,0.1)

# Display games
st.markdown(f"### 🗓️ Week {week}")
wk = data[data["week"]==week].copy()
if wk.empty:
    st.warning("⚠️ No games found for this week.")
    st.stop()

wk["home_win_prob_model"] = model.predict_proba(wk[FEATURES])[:,1]
for _,r in wk.iterrows():
    home,away=r["home_team"],r["away_team"]
    if not home or not away: continue
    prob=r["home_win_prob_model"]
    status="Final" if not np.isnan(r["home_score"]) else "Upcoming"
    with st.expander(f"{away.title()} @ {home.title()} | {status}"):
        st.image(get_logo(away),width=70)
        st.image(get_logo(home),width=70)
        st.write(f"Spread: {r['spread']:+.1f} | O/U: {r['over_under']:.1f}")
        st.write(f"Model Home Win Probability: {prob*100:.1f}%")
        if status=="Final":
            st.write(f"🏁 Final: {int(r['away_score'])}-{int(r['home_score'])}")

st.caption(f"Updated: {datetime.now():%Y-%m-%d %H:%M:%S}")
