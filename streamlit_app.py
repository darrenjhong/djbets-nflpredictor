# DJBets NFL Predictor v12.3.1
# Stable with Safe Merge for SportsOddsHistory integration

import os, re, math, hashlib, io
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image

# ------------------------
# App Config
# ------------------------
st.set_page_config(page_title="DJBets NFL Predictor", layout="wide", page_icon="🏈")
THIS_YEAR = 2025
LOGO_DIR = os.path.join("public", "logos")
DEFAULT_LOGO = os.path.join(LOGO_DIR, "nfl.png")
ESPN_HEADERS = {"User-Agent": "Mozilla/5.0"}
SOH_URL = "https://www.sportsoddshistory.com/nfl-game-season/?y={year}&seasontype=reg&week={week}"

# ------------------------
# Helpers
# ------------------------
def safe_float(x):
    try:
        return float(str(x).replace("−", "-"))
    except:
        return np.nan

def normalize_team(name: str) -> str:
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r"[^a-z ]", "", name)
    aliases = {
        "washington": "commanders",
        "ny giants": "giants",
        "ny jets": "jets",
        "tampa bay": "buccaneers",
        "san francisco": "49ers",
        "new england": "patriots",
        "green bay": "packers",
        "la rams": "rams",
        "la chargers": "chargers",
        "oakland": "raiders",
        "las vegas": "raiders"
    }
    for key, val in aliases.items():
        if key in name:
            name = val
    return name.strip().replace(" ", "")

def get_logo_path(team: str):
    if not team:
        return None
    team_key = normalize_team(team)
    if os.path.isdir(LOGO_DIR):
        for f in os.listdir(LOGO_DIR):
            f_lower = f.lower().replace("-", "").replace("_", "")
            if team_key in f_lower and f.endswith((".png", ".jpg", ".svg")):
                p = os.path.join(LOGO_DIR, f)
                if os.path.exists(p):
                    return p
    if os.path.exists(DEFAULT_LOGO):
        return DEFAULT_LOGO
    img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def pseudo_elo_diff(home: str, away: str):
    return (int(hashlib.md5(f"{home}-{away}".encode()).hexdigest(), 16) % 400) - 200

# ------------------------
# ESPN Fetching
# ------------------------
@st.cache_data(ttl=1200)
def fetch_espn_week(week: int, year: int):
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?year={year}&week={week}&seasontype=2"
    try:
        r = requests.get(url, headers=ESPN_HEADERS, timeout=10)
        data = r.json()
    except:
        return pd.DataFrame()

    games = []
    for ev in data.get("events", []):
        comp = ev.get("competitions", [{}])[0]
        comps = comp.get("competitors", [])
        if len(comps) != 2:
            continue
        home = [c for c in comps if c.get("homeAway") == "home"][0]
        away = [c for c in comps if c.get("homeAway") == "away"][0]
        games.append({
            "week": week,
            "season": year,
            "status": "Final" if comp.get("status", {}).get("type", {}).get("completed") else "Upcoming",
            "home_team": home.get("team", {}).get("nickname", ""),
            "away_team": away.get("team", {}).get("nickname", ""),
            "home_score": safe_float(home.get("score")),
            "away_score": safe_float(away.get("score"))
        })
    return pd.DataFrame(games)

@st.cache_data(ttl=3600)
def fetch_espn_full(year: int):
    allw = []
    for wk in range(1, 19):
        df = fetch_espn_week(wk, year)
        if not df.empty:
            allw.append(df)
    if not allw:
        return pd.DataFrame()
    return pd.concat(allw, ignore_index=True)

# ------------------------
# SportsOddsHistory Scraper
# ------------------------
@st.cache_data(ttl=3600)
def fetch_soh_week(week: int, year: int):
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
            if len(tds) < 5 or "@" not in tds[1]:
                continue
            away, home = [t.strip() for t in tds[1].split("@")]
            spread = safe_float(re.sub("[^0-9.-]", "", tds[3]))
            ou = safe_float(re.sub("[^0-9.]", "", tds[4]))
            rows.append({"week": week, "home_team": home, "away_team": away, "spread": spread, "over_under": ou})
        df = pd.DataFrame(rows)
        if not df.empty:
            df["home_team"] = df["home_team"].astype(str)
            df["away_team"] = df["away_team"].astype(str)
        return df
    except Exception as e:
        st.warning(f"⚠️ SOH fetch failed for week {week}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def merge_espn_soh(year: int):
    espn = fetch_espn_full(year)
    if espn.empty:
        return pd.DataFrame()
    parts = []
    for wk in sorted(espn["week"].unique()):
        soh = fetch_soh_week(wk, year)
        # ✅ Check columns before merging
        required_cols = {"week", "home_team", "away_team"}
        if soh.empty or not required_cols.issubset(soh.columns):
            st.warning(f"⚠️ SOH data missing required columns for week {wk}. Using ESPN-only data.")
            merged = espn[espn["week"] == wk].copy()
            merged["spread"] = np.nan
            merged["over_under"] = np.nan
        else:
            try:
                merged = pd.merge(
                    espn[espn["week"] == wk],
                    soh,
                    on=["week", "home_team", "away_team"],
                    how="left"
                )
            except Exception as e:
                st.warning(f"⚠️ Merge failed for week {wk}: {e}")
                merged = espn[espn["week"] == wk].copy()
                merged["spread"] = np.nan
                merged["over_under"] = np.nan
        parts.append(merged)
    return pd.concat(parts, ignore_index=True)

# ------------------------
# Model Prediction Logic
# ------------------------
def vegas_prob(spread):
    s = safe_float(spread)
    return 1 / (1 + math.exp(-(-s) / 5.5)) if not np.isnan(s) else np.nan

def predict_game(row, weather_sens=1.0):
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

# ------------------------
# Sidebar + UI + Cards
# ------------------------
# (same as before – no changes)
