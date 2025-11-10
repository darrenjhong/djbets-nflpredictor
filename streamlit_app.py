# DJBets NFL Predictor v10.0
# - Full-season weeks dropdown
# - ESPN schedule scrape with offline fallback generator
# - Auto first-run model training
# - ELO & weather visualizations
# - Per-game "Details" expander
# - Dark-friendly UI (works great on Streamlit dark theme)

import os
import json
import time
import math
import requests
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

import streamlit as st
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

# --------------------------------------------------------------
# ⚙️ App Config
# --------------------------------------------------------------
st.set_page_config(
    page_title="DJBets NFL Predictor",
    page_icon="🏈",
    layout="wide"
)

DATA_DIR = "data"
SCHEDULE_FILE = os.path.join(DATA_DIR, "schedule.csv")
MODEL_FILE = os.path.join(DATA_DIR, "model.json")
ELO_CSV = os.path.join(DATA_DIR, "elo_ratings.csv")         # optional override
WEATHER_CSV = os.path.join(DATA_DIR, "weather.csv")         # optional override

MAX_WEEKS = 18
DEFAULT_SEASONS = [2026, 2025, 2024]

TEAMS = [
    "BUF", "MIA", "NE", "NYJ",
    "BAL", "CIN", "CLE", "PIT",
    "HOU", "IND", "JAX", "TEN",
    "DEN", "KC", "LV", "LAC",
    "DAL", "NYG", "PHI", "WAS",
    "CHI", "DET", "GB", "MIN",
    "ATL", "CAR", "NO", "TB",
    "ARI", "LAR", "SF", "SEA"
]

MODEL_FEATURES = ["elo_diff", "temp_c", "wind_kph", "precip_prob"]

os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------------------------------------------
# 🧰 Utilities
# --------------------------------------------------------------
def _safe_iso(dt) -> str:
    try:
        return pd.to_datetime(dt).isoformat()
    except Exception:
        return ""

def _kick_str(v) -> str:
    try:
        ts = pd.to_datetime(v)
        return ts.strftime("%a %b %d, %I:%M %p")
    except Exception:
        return "TBD"

# --------------------------------------------------------------
# 🧠 Model (train on first run if missing)
# --------------------------------------------------------------
@st.cache_resource
def load_or_train_model() -> xgb.XGBClassifier:
    if os.path.exists(MODEL_FILE):
        model = xgb.XGBClassifier()
        model.load_model(MODEL_FILE)
        return model

    # train a small model on synthetic historical-style features
    rng = np.random.default_rng(42)
    n = 400
    df = pd.DataFrame({
        "elo_diff": rng.normal(0, 100, n),
        "temp_c": rng.uniform(-5, 25, n),
        "wind_kph": rng.uniform(0, 25, n),
        "precip_prob": rng.uniform(0, 1, n),
    })
    # True prob via simple sigmoid on elo_diff and weather noise
    logits = 0.012*df["elo_diff"] + 0.02*(20 - df["temp_c"]) - 0.01*df["wind_kph"] - 0.2*(df["precip_prob"]-0.4)
    p = 1/(1+np.exp(-logits))
    y = (rng.uniform(0,1,n) < p).astype(int)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        eval_metric="logloss"
    )
    model.fit(df[MODEL_FEATURES].values, y.values)
    model.save_model(MODEL_FILE)
    return model

# --------------------------------------------------------------
# 🕸️ ESPN schedule (with robust fallback generator)
# --------------------------------------------------------------
@st.cache_data(ttl=7*24*3600)
def scrape_espn_schedule(season: int) -> pd.DataFrame:
    games: List[Dict] = []
    for week in range(1, MAX_WEEKS + 1):
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?year={season}&seasontype=2&week={week}"
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            r.raise_for_status()
            data = r.json()
            for ev in data.get("events", []):
                comp = (ev.get("competitions") or [{}])[0]
                if not comp:
                    continue
                home_abbr, away_abbr, home_logo, away_logo = "TBD", "TBD", "", ""
                for c in comp.get("competitors", []):
                    t = c.get("team", {})
                    abbr = t.get("abbreviation") or ""
                    logo = t.get("logo") or (t.get("logos", [{}])[0].get("href", "") if t.get("logos") else "")
                    if c.get("homeAway") == "home":
                        home_abbr, home_logo = abbr, logo
                    else:
                        away_abbr, away_logo = abbr, logo
                odds = (comp.get("odds") or [{}])[0]
                spread = odds.get("details", "N/A")
                kickoff = comp.get("date") or ev.get("date")

                games.append({
                    "season": season,
                    "week": week,
                    "home_team": home_abbr or "TBD",
                    "away_team": away_abbr or "TBD",
                    "kickoff_et": _safe_iso(kickoff),
                    "spread": spread,
                    "home_logo": home_logo or f"https://a.espncdn.com/i/teamlogos/nfl/500/{(home_abbr or 'tbd').lower()}.png",
                    "away_logo": away_logo or f"https://a.espncdn.com/i/teamlogos/nfl/500/{(away_abbr or 'tbd').lower()}.png",
                })
        except Exception:
            # ignore a single-week failure
            continue

    df = pd.DataFrame(games)

    # ESPN might be partial; fill with generator if needed
    if df["week"].nunique() < MAX_WEEKS or len(df) < 100:
        gen = []
        # stable pairing template per week
        for week in range(1, MAX_WEEKS + 1):
            # deterministic pairing per week for consistency
            order = TEAMS.copy()
            rng = np.random.default_rng(season*100 + week)
            rng.shuffle(order)
            # 16 games
            for i in range(0, len(order), 2):
                home, away = order[i], order[i+1]
                kickoff = (datetime(season, 9, 1) + timedelta(weeks=week-1))
                kickoff = kickoff + timedelta(days=rng.integers(0,3))  # Thu/Sun/Mon-ish
                kickoff = kickoff.replace(hour=int(rng.integers(12, 21)), minute=0)
                gen.append({
                    "season": season,
                    "week": week,
                    "home_team": home,
                    "away_team": away,
                    "kickoff_et": kickoff.isoformat(),
                    "spread": f"{rng.choice(['+','-'])}{int(rng.integers(1,8))}",
                    "home_logo": f"https://a.espncdn.com/i/teamlogos/nfl/500/{home.lower()}.png",
                    "away_logo": f"https://a.espncdn.com/i/teamlogos/nfl/500/{away.lower()}.png",
                })
        gen_df = pd.DataFrame(gen)

        # merge ESPN where available, else generator
        if not df.empty:
            key_cols = ["season","week","home_team","away_team"]
            merged = pd.merge(gen_df, df, on=key_cols, how="left", suffixes=("","_espn"))
            # prefer espn values where present
            def coalesce(a,b): return b if (isinstance(b,str) and b) else a
            for col in ["kickoff_et","spread","home_logo","away_logo"]:
                merged[col] = merged[f"{col}_espn"].combine_first(merged[col])
            keep = ["season","week","home_team","away_team","kickoff_et","spread","home_logo","away_logo"]
            df = merged[keep].copy()
        else:
            df = gen_df

    # persist for transparency
    df.to_csv(SCHEDULE_FILE, index=False)
    return df

# --------------------------------------------------------------
# 🧮 Simple ELO + Weather (mock if no CSVs present)
# --------------------------------------------------------------
@st.cache_data
def load_elo(season: int) -> pd.DataFrame:
    if os.path.exists(ELO_CSV):
        out = pd.read_csv(ELO_CSV)
        return out
    # mock per-team elo centered ~1500
    rng = np.random.default_rng(season)
    rows = []
    for t in TEAMS:
        rows.append({"season": season, "team": t, "elo": 1500 + rng.normal(0, 50)})
    return pd.DataFrame(rows)

@st.cache_data
def load_weather(season: int) -> pd.DataFrame:
    if os.path.exists(WEATHER_CSV):
        out = pd.read_csv(WEATHER_CSV)
        if "kickoff_et" in out.columns:
            out["kickoff_et"] = pd.to_datetime(out["kickoff_et"], errors="coerce")
        return out
    # mock weekly weather per team
    rng = np.random.default_rng(season+123)
    rows = []
    for w in range(1, MAX_WEEKS+1):
        for t in TEAMS:
            rows.append({
                "season": season,
                "week": w,
                "team": t,
                "temp_c": float(rng.uniform(-2, 22)),
                "wind_kph": float(rng.uniform(0, 25)),
                "precip_prob": float(rng.uniform(0, 1)),
            })
    return pd.DataFrame(rows)

def attach_elo_weather(sched: pd.DataFrame, season: int) -> pd.DataFrame:
    elo = load_elo(season)
    wx  = load_weather(season)

    # attach elo
    df = sched.copy()
    df = df.merge(elo.rename(columns={"team":"home_team","elo":"elo_home"}),
                  on=["season","home_team"], how="left")
    df = df.merge(elo.rename(columns={"team":"away_team","elo":"elo_away"}),
                  on=["season","away_team"], how="left")
    # attach weather (use team-week averages)
    df = df.merge(wx.rename(columns={"team":"home_team"}), on=["season","week","home_team"], how="left")
    df = df.merge(wx.rename(columns={
        "team":"away_team",
        "temp_c":"temp_c_away",
        "wind_kph":"wind_kph_away",
        "precip_prob":"precip_prob_away"
    }), on=["season","week","away_team"], how="left")
    # Build model features (home-centric)
    df["elo_diff"] = df["elo_home"].fillna(1500) - df["elo_away"].fillna(1500)
    df["temp_c"] = df["temp_c"].fillna(10.0)           # home weather proxy
    df["wind_kph"] = df["wind_kph"].fillna(10.0)
    df["precip_prob"] = df["precip_prob"].fillna(0.3)
    return df

# --------------------------------------------------------------
# 🎛️ Sidebar Controls
# --------------------------------------------------------------
st.sidebar.markdown("## 🏈 DJBets NFL Predictor")

# Build schedule (and persist) first so weeks are known
season = st.sidebar.selectbox("Season", DEFAULT_SEASONS, index=1)
sched = scrape_espn_schedule(season)

if sched.empty:
    st.error("No schedule available. Try again later.")
    st.stop()

weeks_available = sorted(sched["week"].dropna().unique().tolist())
# ensure full 1..18 dropdown
full_weeks = list(range(1, MAX_WEEKS+1))
week_idx_default = 0 if 1 in weeks_available else (weeks_available[0]-1 if weeks_available else 0)
week = st.sidebar.selectbox("Week", full_weeks, index=week_idx_default)

if st.sidebar.button("♻️ Refresh from ESPN"):
    scrape_espn_schedule.clear()  # clear cache
    st.experimental_rerun()

# --------------------------------------------------------------
# 📦 Data join + model
# --------------------------------------------------------------
sched["kickoff_et"] = pd.to_datetime(sched["kickoff_et"], errors="coerce")
week_df = sched[sched["week"] == week].copy()
week_df = attach_elo_weather(week_df, season)

model = load_or_train_model()

st.title(f"🏈 DJBets NFL Predictor — Week {week} ({season})")
st.caption("ESPN + offline fallback • auto-train on first run • demo ELO & weather effects")

if week_df.empty:
    st.warning("No games found for this week.")
    st.stop()

# --------------------------------------------------------------
# 📈 Overview charts (ELO & Weather)
# --------------------------------------------------------------
# ELO scatter: elo_home vs elo_away
st.subheader("League Context (Week level)")

left, right = st.columns(2)

with left:
    fig = plt.figure(figsize=(5.5,4))
    x = week_df["elo_away"].fillna(1500)
    y = week_df["elo_home"].fillna(1500)
    plt.scatter(x, y, alpha=0.7)
    plt.xlabel("Away ELO")
    plt.ylabel("Home ELO")
    plt.title("ELO: Home vs Away (Week)")
    st.pyplot(fig, clear_figure=True)

with right:
    fig2 = plt.figure(figsize=(5.5,4))
    temps = week_df["temp_c"].fillna(10)
    winds = week_df["wind_kph"].fillna(10)
    plt.scatter(temps, winds, alpha=0.7)
    plt.xlabel("Temp °C (home proxy)")
    plt.ylabel("Wind (kph)")
    plt.title("Weather: Temp vs Wind (Week)")
    st.pyplot(fig2, clear_figure=True)

st.markdown("---")

# --------------------------------------------------------------
# 🧮 Predictions + Game Cards
# --------------------------------------------------------------
for _, row in week_df.sort_values("kickoff_et").iterrows():
    c1, c2, c3 = st.columns([1.5, 3, 1.5])

    with c1:
        if isinstance(row.get("away_logo"), str) and row["away_logo"]:
            st.image(row["away_logo"], width=70)
        st.markdown(f"**{row.get('away_team','TBD')}**")

    with c3:
        if isinstance(row.get("home_logo"), str) and row["home_logo"]:
            st.image(row["home_logo"], width=70)
        st.markdown(f"**{row.get('home_team','TBD')}**")

    # Build features
    feats = {
        "elo_diff": float(row.get("elo_diff", 0.0)),
        "temp_c": float(row.get("temp_c", 10.0)),
        "wind_kph": float(row.get("wind_kph", 10.0)),
        "precip_prob": float(row.get("precip_prob", 0.3)),
    }
    X = pd.DataFrame([feats])[MODEL_FEATURES].astype(float)
    prob = float(model.predict_proba(X)[0,1])
    kickoff_str = _kick_str(row.get("kickoff_et"))

    with c2:
        st.markdown(f"**Kickoff:** {kickoff_str}  |  **Spread:** {row.get('spread','N/A')}")
        st.progress(prob, text=f"Home win probability: {prob*100:.1f}%")

        with st.expander("Details & Features"):
            metrics = pd.DataFrame([
                {"Feature":"elo_diff (home-away)","Value": feats["elo_diff"]},
                {"Feature":"temp_c (home proxy)","Value": feats["temp_c"]},
                {"Feature":"wind_kph (home proxy)","Value": feats["wind_kph"]},
                {"Feature":"precip_prob (home proxy)","Value": feats["precip_prob"]},
            ])
            st.dataframe(metrics, use_container_width=True, hide_index=True)

            # tiny feature effect bar (single-sample)
            fig3 = plt.figure(figsize=(5.5,2.6))
            plt.bar(MODEL_FEATURES, [feats[k] for k in MODEL_FEATURES])
            plt.xticks(rotation=15)
            plt.title("Model Inputs (this matchup)")
            st.pyplot(fig3, clear_figure=True)

st.markdown("---")

# --------------------------------------------------------------
# ⬇️ Export Week Predictions
# --------------------------------------------------------------
def make_week_predictions_frame(week_df: pd.DataFrame, model: xgb.XGBClassifier) -> pd.DataFrame:
    sub = week_df.copy()
    sub["elo_diff"] = sub["elo_diff"].fillna(0)
    sub["temp_c"] = sub["temp_c"].fillna(10.0)
    sub["wind_kph"] = sub["wind_kph"].fillna(10.0)
    sub["precip_prob"] = sub["precip_prob"].fillna(0.3)

    X = sub[MODEL_FEATURES].astype(float).values
    sub["home_win_prob"] = model.predict_proba(X)[:,1]
    sub["predicted_winner"] = np.where(sub["home_win_prob"] >= 0.5, sub["home_team"], sub["away_team"])
    return sub[[
        "season","week","away_team","home_team","kickoff_et","spread",
        "elo_home","elo_away","elo_diff","temp_c","wind_kph","precip_prob","home_win_prob","predicted_winner"
    ]]

pred_df = make_week_predictions_frame(week_df, model)
csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Week Predictions (CSV)", data=csv_bytes, file_name=f"predictions_{season}_wk{week}.csv", mime="text/csv")

# --------------------------------------------------------------
# 🕒 Footer
# --------------------------------------------------------------
if os.path.exists(SCHEDULE_FILE):
    ts = datetime.fromtimestamp(os.path.getmtime(SCHEDULE_FILE))
    st.caption(f"🕒 Schedule last updated: {ts.strftime('%b %d, %Y %I:%M %p')}")
st.caption("🔄 ESPN + offline fallback • Demo ELO & weather • Built with ❤️ by DJBets")
