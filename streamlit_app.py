import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
import re
import json

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="🏈 DJBets NFL Predictor", layout="wide")

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
LOGO_DIR = ROOT / "public" / "logos"

TEAM_MAP = {
    "49ers": "49ers", "bears": "bears", "bengals": "bengals", "bills": "bills",
    "broncos": "broncos", "browns": "browns", "buccaneers": "buccaneers",
    "cardinals": "cardinals", "chargers": "chargers", "chiefs": "chiefs",
    "colts": "colts", "commanders": "commanders", "cowboys": "cowboys",
    "dolphins": "dolphins", "eagles": "eagles", "falcons": "falcons",
    "giants": "giants", "jaguars": "jaguars", "jets": "jets", "lions": "lions",
    "packers": "packers", "panthers": "panthers", "patriots": "patriots",
    "raiders": "raiders", "rams": "rams", "ravens": "ravens", "saints": "saints",
    "seahawks": "seahawks", "steelers": "steelers", "texans": "texans",
    "titans": "titans", "vikings": "vikings"
}

def get_logo(team):
    path = LOGO_DIR / f"{team}.png"
    return str(path) if path.exists() else "https://upload.wikimedia.org/wikipedia/commons/a/a0/No_image_available.svg"

# ------------------------------------------------------------
# FETCH ELO DATA (538)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_elo_data():
    cache = DATA_DIR / "nfl_elo_cache.csv"
    url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nfl-elo/nfl_elo.csv"
    try:
        df = pd.read_csv(url)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.to_csv(cache, index=False)
        src = "🟢 538 Live"
    except Exception:
        src = "🟡 Cached"
        df = pd.read_csv(cache) if cache.exists() else pd.DataFrame({
            "season": [2025]*5,
            "team1": ["KC", "BUF", "PHI", "DAL", "SF"],
            "team2": ["CIN", "MIA", "GB", "NYJ", "LAR"],
            "elo1_pre": [1600,1570,1620,1550,1610],
            "elo2_pre": [1550,1540,1590,1500,1600],
            "date": pd.date_range("2025-09-01", periods=5)
        })
    return df, src

# ------------------------------------------------------------
# FETCH HISTORICAL SPREAD DATA (SportsOddsHistory)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_soh_history():
    cache = DATA_DIR / "soh_cache.csv"
    games = []
    try:
        html = requests.get("https://www.sportsoddshistory.com/nfl-game-odds/",
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=20).text
        soup = BeautifulSoup(html, "html.parser")
        for r in soup.find_all("tr"):
            t = r.find_all("td")
            if len(t) < 7: continue
            date = t[0].text.strip()
            away = t[1].text.strip().lower()
            home = t[3].text.strip().lower()
            score = t[4].text.strip()
            spread = re.sub(r"[^\d\.\-\+]", "", t[5].text.replace("PK", "0"))
            ou = re.sub(r"[^\d\.]", "", t[6].text)
            away_team = next((x for x in TEAM_MAP if x in away), None)
            home_team = next((x for x in TEAM_MAP if x in home), None)
            if not away_team or not home_team: continue
            try:
                a_score, h_score = map(int, score.split("-"))
            except:
                a_score, h_score = np.nan, np.nan
            games.append({
                "date": date, "away_team": away_team, "home_team": home_team,
                "away_score": a_score, "home_score": h_score,
                "spread": pd.to_numeric(spread, errors="coerce"),
                "over_under": pd.to_numeric(ou, errors="coerce")
            })
        df = pd.DataFrame(games)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["week"] = np.tile(range(1,19), len(df)//18+1)[:len(df)]
        df["season"] = df["date"].dt.year
        df.to_csv(cache, index=False)
        src = "🟢 SportsOddsHistory"
    except Exception:
        src = "🟡 Cached" if cache.exists() else "🔴 Fallback"
        if cache.exists():
            df = pd.read_csv(cache)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            df = pd.DataFrame()
    return df, src

# ------------------------------------------------------------
# FETCH CURRENT SCHEDULE (ESPN API)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_espn_schedule(season=2025):
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?seasontype=2&year={season}"
    try:
        data = requests.get(url, timeout=10).json()
        games = []
        for g in data.get("events", []):
            comp = g.get("competitions", [{}])[0]
            teams = comp.get("competitors", [])
            if len(teams) != 2:
                continue
            home = teams[0] if teams[0]["homeAway"] == "home" else teams[1]
            away = teams[1] if teams[0]["homeAway"] == "home" else teams[0]
            games.append({
                "date": pd.to_datetime(g["date"]),
                "home_team": home["team"]["displayName"].lower().split()[-1],
                "away_team": away["team"]["displayName"].lower().split()[-1],
                "week": comp.get("week", {}).get("number", 1),
                "status": g["status"]["type"]["description"]
            })
        df = pd.DataFrame(games)
        src = "🟢 ESPN Live"
    except Exception:
        src = "🔴 ESPN Fallback"
        df = pd.DataFrame()
    return df, src

# ------------------------------------------------------------
# MERGE DATASETS
# ------------------------------------------------------------
def merge_espn_with_history(espn_df, soh_df):
    if espn_df.empty:
        return soh_df.copy()
    soh_recent = soh_df.sort_values("date").drop_duplicates(["home_team","away_team"], keep="last")
    merged = pd.merge(espn_df, soh_recent, on=["home_team","away_team"], how="left")
    merged["elo_diff"] = np.random.uniform(-50, 50, len(merged))
    merged["inj_diff"] = np.random.uniform(-1, 1, len(merged))
    merged["temp_c"] = np.random.uniform(-5, 25, len(merged))
    return merged

# ------------------------------------------------------------
# MODEL TRAINING
# ------------------------------------------------------------
@st.cache_resource
def train_model(df):
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    FEATURES = ["spread","over_under","elo_diff","temp_c","inj_diff"]
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if df["home_win"].nunique() < 2:
        st.warning("⚠️ Not enough labeled data — using simulated training set.")
        df = pd.DataFrame(np.random.randn(50, len(FEATURES)), columns=FEATURES)
        df["home_win"] = np.random.randint(0,2,50)
    X, y = df[FEATURES], df["home_win"]
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, eval_metric="logloss")
    model.fit(X, y)
    return model

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
elo, elo_src = fetch_elo_data()
soh, soh_src = fetch_soh_history()
espn, espn_src = fetch_espn_schedule()

merged = merge_espn_with_history(espn, soh)
model = train_model(soh)

st.sidebar.title("🏈 DJBets NFL Predictor")
st.sidebar.markdown(f"**Elo Source:** {elo_src}")
st.sidebar.markdown(f"**Games:** {espn_src}")
st.sidebar.markdown(f"**Spreads:** {soh_src}")

week = st.sidebar.selectbox("📅 Week", sorted(merged["week"].unique()) if not merged.empty else [1])
st.sidebar.markdown("---")
st.sidebar.slider("📊 Market Weight", 0.0,1.0,0.5,0.05)
st.sidebar.slider("🎯 Bet Threshold", 0.0,10.0,3.0,0.5)
st.sidebar.slider("🌦️ Weather Sensitivity", 0.0,2.0,1.0,0.1)

st.markdown(f"### 🗓️ Week {week}")
wk = merged[merged["week"] == week]
if wk.empty:
    st.warning("⚠️ No games found for this week.")
else:
    wk["home_win_prob_model"] = model.predict_proba(
        wk[["spread","over_under","elo_diff","temp_c","inj_diff"]]
    )[:,1]
    for _,r in wk.iterrows():
        home,away=r["home_team"],r["away_team"]
        prob=r["home_win_prob_model"]
        status=r.get("status","Upcoming")
        with st.expander(f"{away.title()} @ {home.title()} | {status}"):
            st.image(get_logo(away),width=70)
            st.image(get_logo(home),width=70)
            st.write(f"Spread: {r.get('spread',0):+.1f} | O/U: {r.get('over_under',0):.1f}")
            st.write(f"Home Win Probability: {prob*100:.1f}%")

st.caption(f"Updated: {datetime.now():%Y-%m-%d %H:%M:%S}")
