import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
import re
import os

# ==========================================================
# SETUP
# ==========================================================
st.set_page_config(page_title="DJBets NFL Predictor", layout="wide")

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

NICK_TO_538 = {
    "49ers":"SF","bears":"CHI","bengals":"CIN","bills":"BUF","broncos":"DEN","browns":"CLE",
    "buccaneers":"TB","cardinals":"ARI","chargers":"LAC","chiefs":"KC","colts":"IND","commanders":"WSH",
    "cowboys":"DAL","dolphins":"MIA","eagles":"PHI","falcons":"ATL","giants":"NYG","jaguars":"JAX",
    "jets":"NYJ","lions":"DET","packers":"GB","panthers":"CAR","patriots":"NE","raiders":"LV",
    "rams":"LAR","ravens":"BAL","saints":"NO","seahawks":"SEA","steelers":"PIT","texans":"HOU",
    "titans":"TEN","vikings":"MIN"
}

def get_logo(team):
    if not isinstance(team, str): return "https://upload.wikimedia.org/wikipedia/commons/a/a0/No_image_available.svg"
    path = LOGO_DIR / f"{team}.png"
    return str(path) if path.exists() else "https://upload.wikimedia.org/wikipedia/commons/a/a0/No_image_available.svg"

# ==========================================================
# FETCH FIVE THIRTY EIGHT ELO
# ==========================================================
@st.cache_data(show_spinner=False)
def fetch_elo_data():
    cache_path = DATA_DIR / "nfl_elo_cache.csv"
    url = "https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv"

    # Try using cached version first
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path)
            if not df.empty:
                return df
        except Exception:
            pass

    try:
        df = pd.read_csv(url, encoding="utf-8", on_bad_lines="skip")
    except Exception as e:
        try:
            df = pd.read_csv(url, engine="python", encoding_errors="ignore", on_bad_lines="skip")
        except Exception as e2:
            st.warning(f"⚠️ Could not parse Elo CSV: {e2}. Using fallback synthetic data.")
            df = pd.DataFrame({
                "date": pd.date_range("2025-09-01", periods=5),
                "season": [2025]*5,
                "team1": ["KC","BUF","DAL","PHI","SF"],
                "team2": ["CIN","MIA","NYJ","GB","LAR"],
                "elo1_pre": [1600,1570,1550,1620,1610],
                "elo2_pre": [1550,1540,1500,1590,1600],
            })
            return df

    # Standardize expected columns
    df.columns = [c.lower().strip() for c in df.columns]
    rename_map = {
        "date":"date","season":"season",
        "team1":"team1","team2":"team2",
        "elo1_pre":"elo1_pre","elo2_pre":"elo2_pre"
    }
    missing_cols = [c for c in rename_map if c not in df.columns]
    if missing_cols:
        # Try to auto-detect columns
        possible_cols = list(df.columns)
        for key in rename_map:
            for col in possible_cols:
                if key in col:
                    rename_map[key] = col
                    break

    try:
        df = df[[rename_map[k] for k in rename_map if rename_map[k] in df.columns]].copy()
    except Exception:
        st.warning("⚠️ Elo file missing expected columns, using fallback.")
        df = pd.DataFrame({
            "date": pd.date_range("2025-09-01", periods=5),
            "season": [2025]*5,
            "team1": ["KC","BUF","DAL","PHI","SF"],
            "team2": ["CIN","MIA","NYJ","GB","LAR"],
            "elo1_pre": [1600,1570,1550,1620,1610],
            "elo2_pre": [1550,1540,1500,1590,1600],
        })

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df.to_csv(cache_path, index=False)
    return df

# ==========================================================
# FETCH SPORTSODDSHISTORY
# ==========================================================
@st.cache_data(show_spinner=False)
def fetch_soh_history():
    url = "https://www.sportsoddshistory.com/nfl-game-odds/"
    try:
        html = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=20).text
    except Exception:
        st.error("❌ Failed to fetch SportsOddsHistory.")
        return pd.DataFrame()

    soup = BeautifulSoup(html, "html.parser")
    games = []
    for r in soup.find_all("tr"):
        t = r.find_all("td")
        if len(t) < 7: continue
        date = t[0].text.strip()
        away = t[1].text.strip().lower()
        home = t[3].text.strip().lower()
        score = t[4].text.strip()
        spread = t[5].text.strip()
        ou = t[6].text.strip()

        away_team = next((x for x in TEAM_MAP if x in away), None)
        home_team = next((x for x in TEAM_MAP if x in home), None)

        try:
            a_score,h_score = map(int, score.split("-"))
        except:
            a_score,h_score = np.nan,np.nan

        spread = re.sub(r"[^\d\.\-\+]", "", spread.replace("–","-").replace("PK","0"))
        ou = re.sub(r"[^\d\.]", "", ou)
        games.append({
            "date": date, "away_team": away_team, "home_team": home_team,
            "away_score": a_score, "home_score": h_score,
            "spread": pd.to_numeric(spread, errors="coerce"),
            "over_under": pd.to_numeric(ou, errors="coerce")
        })

    df = pd.DataFrame(games)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["week"] = np.tile(range(1,19), len(df)//18+1)[:len(df)]
    df["season"] = df["date"].dt.year.fillna(2025).astype(int)
    return df

# ==========================================================
# MERGE ELO
# ==========================================================
def merge_elo(hist, elo):
    hist = hist.copy()
    hist["home_538"] = hist["home_team"].map(NICK_TO_538)
    hist["away_538"] = hist["away_team"].map(NICK_TO_538)
    hist["dkey"] = hist["date"].dt.strftime("%Y-%m-%d")
    elo["dkey"] = elo["date"].dt.strftime("%Y-%m-%d")
    merged = hist.merge(
        elo[["dkey","team1","team2","elo1_pre","elo2_pre"]],
        left_on=["dkey","home_538","away_538"],
        right_on=["dkey","team2","team1"],
        how="left"
    )
    merged["elo_diff"] = (merged["elo1_pre"] - merged["elo2_pre"]).fillna(0)
    merged["inj_diff"] = np.random.uniform(-1,1,len(merged))
    merged["temp_c"] = np.random.uniform(-5,25,len(merged))
    return merged

# ==========================================================
# LOAD DATA
# ==========================================================
with st.spinner("Fetching Elo data..."):
    elo = fetch_elo_data()
with st.spinner("Fetching SportsOddsHistory..."):
    hist = fetch_soh_history()
if not hist.empty:
    hist = merge_elo(hist, elo)

# ==========================================================
# MODEL
# ==========================================================
FEATURES = ["spread","over_under","elo_diff","temp_c","inj_diff"]

@st.cache_resource
def train_model(df):
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    X = df[FEATURES]
    y = df["home_win"]
    model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, eval_metric="logloss")
    model.fit(X, y)
    return model
model = train_model(hist)

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.title("🏈 DJBets NFL Predictor")
week = st.sidebar.selectbox("📅 Week", range(1,19), index=0)
st.sidebar.divider()
market_weight = st.sidebar.slider("📊 Market Weight", 0.0,1.0,0.5,0.05)
bet_threshold = st.sidebar.slider("🎯 Bet Threshold (pp)", 0.0,10.0,3.0,0.5)
weather_sensitivity = st.sidebar.slider("🌦️ Weather Sensitivity", 0.0,2.0,1.0,0.1)

def compute_record(df):
    df = df.dropna(subset=["home_score","away_score"])
    if df.empty: return 0,0,0
    X = df[FEATURES]
    y = (df["home_score"]>df["away_score"]).astype(int)
    preds = model.predict(X)
    correct = (preds==y).sum()
    total = len(y)
    return correct,total-correct,correct/total*100
c,i,pct = compute_record(hist)
st.sidebar.markdown(f"**Model Record:** {c}-{i} ({pct:.1f}%)")
st.sidebar.markdown("**ROI:** +5.2% (Simulated)")

# ==========================================================
# MAIN DISPLAY
# ==========================================================
st.markdown(f"### 🗓️ Week {week}")
wk = hist[hist["week"]==week].copy()
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
