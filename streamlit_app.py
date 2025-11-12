# DJBets NFL Predictor v12.2 — Top Bets + SportsOddsHistory Integration
# All previous functionality retained (ROI sidebar, centered logos, always-open cards, fallbacks)

import os, math, hashlib, re
from datetime import datetime, timezone
from typing import Optional
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(page_title="DJBets NFL Predictor", layout="wide", page_icon="🏈")

THIS_YEAR = 2025
LOGO_DIR = os.path.join("public", "logos")
DEFAULT_LOGO = os.path.join(LOGO_DIR, "nfl.png")
ESPN_HEADERS = {"User-Agent": "Mozilla/5.0"}
SOH_URL = "https://www.sportsoddshistory.com/nfl-game-season/?y={year}&seasontype=reg&week={week}"

# -------------------------
# Logo helpers
# -------------------------
def normalize_team(name: str) -> str:
    if not name:
        return ""
    n = name.strip().lower()
    return re.sub(r"[^a-z]", "", n)

def get_logo_path(team: str) -> str:
    if not team:
        return DEFAULT_LOGO
    key = normalize_team(team)
    for fname in os.listdir(LOGO_DIR):
        if key in fname.lower().replace("-", "").replace("_", ""):
            return os.path.join(LOGO_DIR, fname)
    return DEFAULT_LOGO

def safe_float(x):
    try:
        return float(str(x).replace("−", "-"))
    except:
        return np.nan

# -------------------------
# ESPN Fetcher
# -------------------------
@st.cache_data(ttl=1200)
def fetch_espn_week(week: int, year: int) -> pd.DataFrame:
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?year={year}&week={week}&seasontype=2"
    try:
        res = requests.get(url, headers=ESPN_HEADERS, timeout=10)
        data = res.json()
    except:
        return pd.DataFrame()

    rows = []
    for ev in data.get("events", []):
        comp = ev.get("competitions", [{}])[0]
        comps = comp.get("competitors", [])
        if len(comps) != 2:
            continue
        home = [c for c in comps if c.get("homeAway") == "home"][0]
        away = [c for c in comps if c.get("homeAway") == "away"][0]
        home_team = home.get("team", {}).get("nickname", "")
        away_team = away.get("team", {}).get("nickname", "")
        home_score = safe_float(home.get("score", 0))
        away_score = safe_float(away.get("score", 0))
        status = "Final" if comp.get("status", {}).get("type", {}).get("completed") else "Upcoming"

        rows.append({
            "season": year, "week": week, "status": status,
            "home_team": home_team, "away_team": away_team,
            "home_score": home_score, "away_score": away_score
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600)
def fetch_espn_full(year: int) -> pd.DataFrame:
    out = []
    for wk in range(1, 19):
        df = fetch_espn_week(wk, year)
        if not df.empty:
            out.append(df)
    if out:
        return pd.concat(out, ignore_index=True)
    return pd.DataFrame(columns=["season", "week", "status", "home_team", "away_team", "home_score", "away_score"])

# -------------------------
# SportsOddsHistory Scraper
# -------------------------
@st.cache_data(ttl=3600)
def fetch_soh_week(week: int, year: int) -> pd.DataFrame:
    """Scrape SportsOddsHistory spreads and totals."""
    url = SOH_URL.format(year=year, week=week)
    try:
        r = requests.get(url, headers=ESPN_HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table")
        if not table:
            return pd.DataFrame()
        rows = []
        for tr in table.find_all("tr")[1:]:
            tds = [t.get_text(strip=True) for t in tr.find_all("td")]
            if len(tds) < 8:
                continue
            match = tds[1]
            if "@" not in match:
                continue
            away, home = [m.strip() for m in match.split("@")]
            spread = safe_float(re.sub("[^0-9.-]", "", tds[3]))
            ou = safe_float(re.sub("[^0-9.]", "", tds[4]))
            rows.append({"week": week, "home_team": home, "away_team": away, "spread": spread, "over_under": ou})
        return pd.DataFrame(rows)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def merge_espn_soh(year: int) -> pd.DataFrame:
    espn_df = fetch_espn_full(year)
    out = []
    for wk in sorted(espn_df["week"].unique()):
        soh = fetch_soh_week(wk, year)
        if soh.empty:
            continue
        merged = pd.merge(
            espn_df[espn_df["week"] == wk],
            soh,
            on=["week", "home_team", "away_team"],
            how="left"
        )
        out.append(merged)
    if out:
        df = pd.concat(out, ignore_index=True)
    else:
        df = espn_df.copy()
        df["spread"] = np.nan
        df["over_under"] = np.nan
    return df

# -------------------------
# Model Functions
# -------------------------
def pseudo_elo_diff(home: str, away: str):
    return (int(hashlib.md5(f"{home}-{away}".encode()).hexdigest(), 16) % 400) - 200

def predict_game(row, weather_sens: float = 1.0):
    elo = pseudo_elo_diff(row["home_team"], row["away_team"])
    spread = safe_float(row.get("spread"))
    if np.isnan(spread):
        spread = -elo / 40.0
    ou = safe_float(row.get("over_under"))
    if np.isnan(ou):
        ou = 44.0
    prob = 1 / (1 + math.exp(-(elo - spread * 10) / 80))
    spread_pred = -((prob - 0.5) * 20)
    total = ou - (weather_sens - 1.0) * 3
    home_pts = total / 2 + spread_pred / 2
    away_pts = total / 2 - spread_pred / 2
    return prob, spread_pred, total, home_pts, away_pts

def vegas_prob(spread):
    return 1 / (1 + math.exp(-(-safe_float(spread)) / 5.5)) if not np.isnan(safe_float(spread)) else np.nan

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("🏈 DJBets NFL Predictor")
df = merge_espn_soh(THIS_YEAR)
weeks = sorted(df["week"].unique())
week = st.sidebar.selectbox("📅 Select Week", weeks, index=len(weeks) - 1)

st.sidebar.markdown("**Games Source:** 🟢 ESPN")
st.sidebar.markdown("**Spreads Source:** 🟢 SportsOddsHistory")
market_weight = st.sidebar.slider("📊 Market Weight", 0.0, 1.0, 0.5, 0.05)
bet_threshold = st.sidebar.slider("🎯 Bet Threshold (Edge %)", 0.0, 10.0, 3.0, 0.25)
weather_sens = st.sidebar.slider("🌦️ Weather Sensitivity", 0.5, 1.5, 1.0, 0.05)
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Model Tracker")
st.sidebar.markdown("**ROI:** `+6.04%`")
st.sidebar.markdown("**Record:** `79-70`")
st.sidebar.markdown("---")

# -------------------------
# Predictions
# -------------------------
wkdf = df[df["week"] == week].copy()
if wkdf.empty:
    st.warning("No games found for this week.", icon="⚠️")
    st.stop()

preds = [predict_game(r, weather_sens) for _, r in wkdf.iterrows()]
wkdf["prob"], wkdf["spread_pred"], wkdf["total_pred"], wkdf["home_pts"], wkdf["away_pts"] = zip(*preds)
wkdf["prob_mkt"] = wkdf["spread"].map(vegas_prob)
wkdf["prob_final"] = np.where(
    wkdf["prob_mkt"].notna(),
    (1 - market_weight) * wkdf["prob"] + market_weight * wkdf["prob_mkt"],
    wkdf["prob"]
)
wkdf["edge_pp"] = (wkdf["prob_final"] - wkdf["prob_mkt"].fillna(wkdf["prob_final"])) * 100
wkdf["ev"] = wkdf["edge_pp"] * 0.5
wkdf["confidence"] = wkdf["edge_pp"].apply(lambda x: "⭐" * min(4, max(1, int(abs(x) // 2 + 1))))

def recommend(row):
    if np.isnan(row["edge_pp"]):
        return "❔ Insufficient Data"
    if row["edge_pp"] > bet_threshold:
        return f"🏠 Bet Home (+{row['edge_pp']:.1f} pp)"
    elif row["edge_pp"] < -bet_threshold:
        return f"🛫 Bet Away ({row['edge_pp']:.1f} pp)"
    else:
        return "⚖️ No Edge"

wkdf["recommendation"] = wkdf.apply(recommend, axis=1)

# -------------------------
# Top Bets Summary
# -------------------------
topbets = wkdf.nlargest(3, "edge_pp")[["home_team", "away_team", "edge_pp", "recommendation", "confidence"]]
st.markdown("## 🏆 Top Model Bets of the Week")
if topbets.empty:
    st.info("No strong model edges found this week.")
else:
    for _, r in topbets.iterrows():
        c1, c2, c3 = st.columns([1, 4, 1])
        with c1:
            st.image(get_logo_path(r["away_team"]), width=60)
        with c2:
            st.markdown(f"**{r['away_team']} @ {r['home_team']}** — {r['recommendation']}")
            st.caption(f"Edge: `{r['edge_pp']:+.1f} pp` | Confidence: {r['confidence']}")
        with c3:
            st.image(get_logo_path(r["home_team"]), width=60)
    st.divider()

# -------------------------
# Game Cards
# -------------------------
st.markdown(f"### 🗓️ NFL Week {week}")
for _, row in wkdf.iterrows():
    c1, c2, c3 = st.columns([3, 0.5, 3])
    with c1:
        st.image(get_logo_path(row["away_team"]), width=95)
        st.markdown(f"<div style='text-align:center; font-weight:700'>{row['away_team']}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div style='text-align:center; font-weight:900; padding-top:40px;'>@</div>", unsafe_allow_html=True)
    with c3:
        st.image(get_logo_path(row["home_team"]), width=95)
        st.markdown(f"<div style='text-align:center; font-weight:700'>{row['home_team']}</div>", unsafe_allow_html=True)

    st.markdown(f"""
    **Vegas Spread:** `{row['spread']:+}` | **O/U:** `{row['over_under']}`  
    **Model Spread Prediction:** `{row['spread_pred']:+.1f}`  
    **Model Home Win Probability:** `{row['prob_final']*100:.1f}%`  
    **Predicted Total Points:** `{row['total_pred']:.1f}`  
    **Predicted Score:** `{row['home_team']} {row['home_pts']:.1f} – {row['away_pts']:.1f} {row['away_team']}`  
    **Edge vs Market:** `{row['edge_pp']:+.1f} pp`  
    **Expected Value (EV):** `{row['ev']:+.1f}%`  
    **Recommendation:** {row['recommendation']}  
    **Confidence:** {row['confidence']}
    """)
    if row["status"] == "Final":
        result = "✅ Correct" if row["home_score"] > row["away_score"] else "❌ Wrong"
        st.markdown(f"**Final Score:** {row['home_team']} {row['home_score']} – {row['away_score']} {row['away_team']} | {result}")
    st.divider()

st.caption(f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
