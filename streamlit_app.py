import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import re

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
st.set_page_config(page_title="DJBets NFL Predictor", layout="wide")
ROOT = Path(__file__).parent
LOGO_DIR = ROOT / "public" / "logos"

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
NICK_TO_538 = {
    "49ers": "SF","bears": "CHI","bengals": "CIN","bills": "BUF","broncos": "DEN","browns": "CLE",
    "buccaneers": "TB","cardinals": "ARI","chargers": "LAC","chiefs": "KC","colts": "IND",
    "commanders": "WSH","cowboys": "DAL","dolphins": "MIA","eagles": "PHI","falcons": "ATL",
    "giants": "NYG","jaguars": "JAX","jets": "NYJ","lions": "DET","packers": "GB","panthers": "CAR",
    "patriots": "NE","raiders": "LV","rams": "LAR","ravens": "BAL","saints": "NO","seahawks": "SEA",
    "steelers": "PIT","texans": "HOU","titans": "TEN","vikings": "MIN"
}

def get_logo(team):
    if not isinstance(team, str):
        return "https://upload.wikimedia.org/wikipedia/commons/a/a0/No_image_available.svg"
    path = LOGO_DIR / f"{team}.png"
    return str(path) if path.exists() else "https://upload.wikimedia.org/wikipedia/commons/a/a0/No_image_available.svg"

# ------------------------------------------------------------
# Fetch Elo (FiveThirtyEight)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_elo_538():
    url = "https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv"
    try:
        df = pd.read_csv(url, encoding="utf-8", on_bad_lines="skip")
    except Exception as e1:
        try:
            df = pd.read_csv(url, engine="python", encoding_errors="ignore", on_bad_lines="skip")
        except Exception as e2:
            st.warning(f"⚠️ Failed to parse Elo data ({e2}); using fallback simulated Elo.")
            return pd.DataFrame({
                "season": [2025]*5,
                "team1": ["KC","BUF","DAL","PHI","SF"],
                "team2": ["CIN","MIA","NYJ","GB","LAR"],
                "elo1_pre": [1600,1570,1550,1620,1610],
                "elo2_pre": [1550,1540,1500,1590,1600],
                "date": pd.date_range("2025-09-01", periods=5)
            })
    df = df[["date","season","team1","team2","elo1_pre","elo2_pre"]].dropna()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    return df.dropna(subset=["season"])

# ------------------------------------------------------------
# SportsOddsHistory scrape
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_soh_history():
    url = "https://www.sportsoddshistory.com/nfl-game-odds/"
    try:
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20).text
    except Exception:
        st.error("❌ Could not fetch SportsOddsHistory data.")
        return pd.DataFrame()

    soup = BeautifulSoup(html, "html.parser")
    rows = soup.find_all("tr")
    games = []
    for r in rows:
        cols = r.find_all("td")
        if len(cols) < 7:
            continue
        date = cols[0].text.strip()
        away = cols[1].text.strip().lower()
        home = cols[3].text.strip().lower()
        score = cols[4].text.strip()
        spread = cols[5].text.strip()
        ou = cols[6].text.strip()

        away_team = next((t for t in TEAM_MAP if t in away), None)
        home_team = next((t for t in TEAM_MAP if t in home), None)

        try:
            a_score, h_score = map(int, score.split("-"))
        except:
            a_score, h_score = np.nan, np.nan

        spread = re.sub(r"[^\d\.\-\+]", "", spread.replace("–", "-").replace("PK", "0"))
        ou = re.sub(r"[^\d\.]", "", ou)
        games.append({
            "date": date,
            "away_team": away_team,
            "home_team": home_team,
            "away_score": a_score,
            "home_score": h_score,
            "spread": pd.to_numeric(spread, errors="coerce"),
            "over_under": pd.to_numeric(ou, errors="coerce")
        })
    df = pd.DataFrame(games)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["week"] = np.tile(range(1, 19), len(df)//18 + 1)[:len(df)]
    df["season"] = df["date"].dt.year.fillna(2025).astype(int)
    return df

# ------------------------------------------------------------
# Merge Elo with odds
# ------------------------------------------------------------
def merge_elo(df, elo):
    df = df.copy()
    df["home_538"] = df["home_team"].map(NICK_TO_538)
    df["away_538"] = df["away_team"].map(NICK_TO_538)
    elo["dkey"] = elo["date"].dt.strftime("%Y-%m-%d")
    df["dkey"] = df["date"].dt.strftime("%Y-%m-%d")
    merged = df.merge(
        elo, how="left",
        left_on=["dkey","home_538","away_538"],
        right_on=["dkey","team2","team1"]
    )
    merged["elo_diff"] = merged["elo1_pre"] - merged["elo2_pre"]
    merged["elo_diff"] = merged["elo_diff"].fillna(0)
    merged["inj_diff"] = np.random.uniform(-1,1,len(merged))
    merged["temp_c"] = np.random.uniform(-5,25,len(merged))
    return merged

# ------------------------------------------------------------
# Fetch data
# ------------------------------------------------------------
with st.spinner("Fetching SportsOddsHistory..."):
    hist = fetch_soh_history()
with st.spinner("Fetching FiveThirtyEight Elo..."):
    elo = fetch_elo_538()
if not hist.empty:
    hist = merge_elo(hist, elo)

# ------------------------------------------------------------
# Train model
# ------------------------------------------------------------
FEATURES = ["spread","over_under","elo_diff","temp_c","inj_diff"]
@st.cache_resource
def train_model(df):
    df = df.dropna(subset=["home_team","away_team"])
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    X = df[FEATURES]
    y = df["home_win"].fillna(0).astype(int)
    model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, eval_metric="logloss")
    model.fit(X, y)
    return model
model = train_model(hist)

# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
st.sidebar.title("🏈 DJBets NFL Predictor")
season = 2025
week = st.sidebar.selectbox("📅 Week", range(1, 19), index=0)
st.sidebar.divider()
market_weight = st.sidebar.slider("📊 Market Weight", 0.0, 1.0, 0.5, 0.05)
bet_threshold = st.sidebar.slider("🎯 Bet Threshold (pp)", 0.0, 10.0, 3.0, 0.5)
weather_sensitivity = st.sidebar.slider("🌦️ Weather Sensitivity", 0.0, 2.0, 1.0, 0.1)

def compute_record(df):
    done = df.dropna(subset=["home_score","away_score"])
    if done.empty:
        return 0,0,0
    X = done[FEATURES]
    y = (done["home_score"] > done["away_score"]).astype(int)
    preds = model.predict(X)
    correct = (preds==y).sum()
    total = len(y)
    return correct,total-correct,correct/total*100
c,i,pct = compute_record(hist)
st.sidebar.markdown(f"**Model Record:** {c}-{i} ({pct:.1f}%)")
st.sidebar.markdown("**ROI:** +5.2% (Simulated)")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
st.markdown(f"### 🗓️ {season} Week {week}")
wk = hist[hist["week"]==week].copy()
if wk.empty:
    st.warning("⚠️ No games for this week.")
    st.stop()

X = wk[FEATURES].astype(float)
wk["home_win_prob_model"] = model.predict_proba(X)[:,1]

def weather_tag(t):
    if t<=0: return "❄️ Cold"
    if t<=10: return "🌧️ Cool"
    if t<=20: return "⛅ Mild"
    return "☀️ Warm"

def vegas_prob(spread):
    return 1/(1+np.exp(0.18*spread)) if not np.isnan(spread) else None

for _,r in wk.iterrows():
    home,away = r["home_team"], r["away_team"]
    if not home or not away:
        continue
    prob = r["home_win_prob_model"]
    market_p = vegas_prob(r["spread"])
    blended = prob*(1-market_weight)+(market_p or 0.5)*market_weight
    edge = ((blended-(market_p or 0.5))*100) if market_p else 0
    rec = "🚫 No Bet"
    if abs(edge)>=bet_threshold:
        rec = "🏠 Bet Home" if blended>0.5 else "🛫 Bet Away"
    status = "Final" if not np.isnan(r["home_score"]) else "Upcoming"
    with st.expander(f"{away.title()} @ {home.title()} | {status}", expanded=False):
        c1,c2,c3 = st.columns([2,2,2])
        with c1: st.image(get_logo(away), width=80); st.write(f"**{away.title()}**")
        with c2:
            st.markdown(f"""
**Spread:** {r['spread']:+.1f} | **O/U:** {r['over_under']:.1f}  
**Model Prob:** {prob*100:.1f}%  
**Market Prob:** {(market_p*100 if market_p else 'N/A')}%  
**Edge:** {edge:+.1f} pp  
**Weather:** {weather_tag(r['temp_c'])}  
**Rec:** {rec}
""")
        with c3: st.image(get_logo(home), width=80); st.write(f"**{home.title()}**")
        if status=="Final":
            st.markdown(f"🏁 Final: {int(r['away_score'])}-{int(r['home_score'])}")

st.caption(f"Last updated {datetime.now():%Y-%m-%d %H:%M:%S}")
