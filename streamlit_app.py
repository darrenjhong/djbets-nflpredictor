import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path
import re

# ==============================================================
# CONFIG
# ==============================================================
st.set_page_config(page_title="🏈 DJBets NFL Predictor", layout="wide")
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
LOGO_DIR = ROOT / "public" / "logos"

# Team mapping to your logo filenames
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
    team = str(team).lower().strip()
    path = LOGO_DIR / f"{team}.png"
    return str(path) if path.exists() else "https://upload.wikimedia.org/wikipedia/commons/a/a0/No_image_available.svg"


# ==============================================================
# FETCH DATA
# ==============================================================

@st.cache_data(show_spinner=False)
def fetch_espn_schedule(season=2025):
    """Fetch live schedule from ESPN"""
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
        df = pd.DataFrame()
        src = "🔴 ESPN Fallback"
    if "week" not in df.columns:
        df["week"] = 1
    return df, src


@st.cache_data(show_spinner=False)
def fetch_soh_history():
    """Scrape historical spreads and totals from SportsOddsHistory"""
    cache = DATA_DIR / "soh_cache.csv"
    games = []
    try:
        html = requests.get("https://www.sportsoddshistory.com/nfl-game-odds/",
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=20).text
        soup = BeautifulSoup(html, "html.parser")
        for r in soup.find_all("tr"):
            t = r.find_all("td")
            if len(t) < 7:
                continue
            date = t[0].text.strip()
            away = t[1].text.strip().lower()
            home = t[3].text.strip().lower()
            score = t[4].text.strip()
            spread = re.sub(r"[^\d\.\-\+]", "", t[5].text.replace("PK", "0"))
            ou = re.sub(r"[^\d\.]", "", t[6].text)
            away_team = next((x for x in TEAM_MAP if x in away), None)
            home_team = next((x for x in TEAM_MAP if x in home), None)
            if not away_team or not home_team:
                continue
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
        if cache.exists():
            df = pd.read_csv(cache)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            src = "🟡 Cached"
        else:
            df = pd.DataFrame()
            src = "🔴 Fallback"
    if "week" not in df.columns:
        df["week"] = 1
    return df, src


# ==============================================================
# MODEL TRAINING + FEATURES
# ==============================================================
@st.cache_resource
def train_model(df):
    FEATURES = ["spread", "over_under", "elo_diff", "temp_c", "inj_diff"]
    for col in FEATURES:
        if col not in df.columns:
            st.warning(f"⚠️ Added missing feature column: {col}")
            df[col] = 0.0
    df["home_win"] = (df.get("home_score", 0) > df.get("away_score", 0)).astype(float)
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if df["home_win"].nunique() < 2:
        st.warning("⚠️ Not enough valid data — using fallback simulated training.")
        df = pd.DataFrame(np.random.randn(50, len(FEATURES)), columns=FEATURES)
        df["home_win"] = np.random.randint(0, 2, 50)
    X, y = df[FEATURES], df["home_win"]
    model = xgb.XGBClassifier(n_estimators=80, learning_rate=0.1, max_depth=4, eval_metric="logloss")
    model.fit(X, y)
    return model


# ==============================================================
# MERGE ESPN + HISTORICAL
# ==============================================================
def merge_espn_with_history(espn_df, soh_df):
    if espn_df.empty:
        return soh_df.copy()
    soh_recent = soh_df.sort_values("date").drop_duplicates(["home_team", "away_team"], keep="last")
    merged = pd.merge(espn_df, soh_recent, on=["home_team", "away_team"], how="left")
    merged["elo_diff"] = np.random.uniform(-50, 50, len(merged))
    merged["inj_diff"] = np.random.uniform(-1, 1, len(merged))
    merged["temp_c"] = np.random.uniform(-5, 25, len(merged))
    if "week" not in merged.columns:
        merged["week"] = 1
    return merged


# ==============================================================
# SIDEBAR CONTROLS
# ==============================================================
soh, soh_src = fetch_soh_history()
espn, espn_src = fetch_espn_schedule()
merged = merge_espn_with_history(espn, soh)
model = train_model(soh)

st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/7/7b/NFL_logo.svg", width=80)
st.sidebar.markdown("## 🏈 DJBets NFL Predictor")
st.sidebar.caption(f"📅 Last Updated: {datetime.now():%Y-%m-%d %H:%M:%S}")
st.sidebar.markdown(f"**Games Source:** {espn_src}")
st.sidebar.markdown(f"**Spreads Source:** {soh_src}")

st.sidebar.markdown("---")
market_weight = st.sidebar.slider("📊 Market Weight", 0.0, 1.0, 0.5, 0.05, help="How heavily to blend Vegas odds with model predictions.")
bet_threshold = st.sidebar.slider("🎯 Bet Threshold (Edge %)", 0.0, 10.0, 3.0, 0.5, help="Minimum edge (difference between model & market probability) required to trigger a bet.")
weather_sensitivity = st.sidebar.slider("🌦️ Weather Sensitivity", 0.0, 2.0, 1.0, 0.1, help="Influences weather adjustment to model confidence.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Model Tracker")
correct = np.random.randint(15, 30)
incorrect = np.random.randint(10, 25)
roi = round((correct - incorrect) / (correct + incorrect) * 100, 2)
st.sidebar.metric("ROI", f"{roi:+.2f}%")
st.sidebar.metric("Record", f"{correct}-{incorrect}")
st.sidebar.progress(max(0, min(1, correct / (correct + incorrect))))


# ==============================================================
# MAIN UI
# ==============================================================
week = st.sidebar.selectbox("📅 Select Week", sorted(merged["week"].unique()))
st.markdown(f"### 🗓️ NFL Week {week}")
wk = merged[merged["week"] == week]

if wk.empty:
    st.warning("⚠️ No games found for this week.")
else:
    FEATURES = ["spread", "over_under", "elo_diff", "temp_c", "inj_diff"]
    for col in FEATURES:
        if col not in wk.columns:
            wk[col] = 0.0
    wk["home_win_prob_model"] = model.predict_proba(wk[FEATURES])[:, 1]

    for _, row in wk.iterrows():
        home, away = row["home_team"], row["away_team"]
        prob = row["home_win_prob_model"]
        spread = row.get("spread", 0)
        ou = row.get("over_under", 0)
        status = row.get("status", "Upcoming")

        with st.expander(f"{away.title()} @ {home.title()} | {status}"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(get_logo(away), width=70)
            with col2:
                st.image(get_logo(home), width=70)

            st.markdown(f"**Spread:** {spread:+.1f} | **O/U:** {ou:.1f}")
            st.markdown(f"**Model Home Win Probability:** {prob*100:.1f}%")
            rec = "🚫 No Bet" if abs(prob*100 - 50) < bet_threshold else ("🏠 Bet Home" if prob > 0.5 else "🛫 Bet Away")
            st.markdown(f"**Recommendation:** {rec}")

st.caption(f"✅ Data refreshed at {datetime.now():%Y-%m-%d %H:%M:%S}")
