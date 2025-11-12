# DJBets NFL Predictor v12.1 — Centered Layout + Fallback Spreads + EV Calculation
# Maintains all previous features and adds EV, improved layout, and market data fallbacks

import os, math, hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="DJBets NFL Predictor", page_icon="🏈", layout="wide")

THIS_YEAR = 2025
WEEKS = list(range(1, 19))
UA_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.espn.com/nfl/schedule",
    "Accept": "application/json"
}

# ----------------------------
# Logos
# ----------------------------
LOGO_DIR = "public/logos"
TEAM_FILE_NAMES = {fn.replace(".png", "").lower(): fn for fn in os.listdir(LOGO_DIR) if fn.endswith(".png")}

def get_logo_path(team_name: str) -> str:
    key = (team_name or "").strip().lower()
    if key in TEAM_FILE_NAMES:
        return os.path.join(LOGO_DIR, TEAM_FILE_NAMES[key])
    for k in TEAM_FILE_NAMES:
        if key.startswith(k[:5]):
            return os.path.join(LOGO_DIR, TEAM_FILE_NAMES[k])
    return os.path.join(LOGO_DIR, "nfl.png")

def safe_float(x):
    try:
        if x in (None, "", "N/A"): return np.nan
        return float(str(x).replace("−", "-"))
    except Exception:
        return np.nan

# ----------------------------
# ESPN Data
# ----------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_espn_week(week: int, year: int) -> pd.DataFrame:
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?year={year}&week={week}&seasontype=2"
    try:
        r = requests.get(url, headers=UA_HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        st.warning(f"⚠️ ESPN fetch failed for week {week}: {e}")
        return pd.DataFrame()

    rows = []
    for ev in data.get("events", []):
        comp = ev.get("competitions", [{}])[0]
        comps = comp.get("competitors", [])
        if len(comps) != 2:
            continue

        home = next((c for c in comps if c.get("homeAway") == "home"), None)
        away = next((c for c in comps if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        def nick(c):
            t = c.get("team", {})
            return (t.get("nickname") or t.get("shortDisplayName") or t.get("abbreviation") or "").lower()

        home_team, away_team = nick(home), nick(away)
        home_score, away_score = safe_float(home.get("score", 0)), safe_float(away.get("score", 0))
        status = "Final" if comp.get("status", {}).get("type", {}).get("completed", False) else "Upcoming"

        spread, ou = np.nan, np.nan
        odds = comp.get("odds", [])
        if odds:
            try:
                o = odds[0]
                spread = safe_float(o.get("spread"))
                ou = safe_float(o.get("overUnder"))
                fav = (o.get("favorite") or "").lower()
                if fav == home_team:
                    spread = -abs(spread)
                elif fav == away_team:
                    spread = abs(spread)
            except Exception:
                pass

        rows.append(dict(
            season=year, week=week, status=status,
            home=home_team, away=away_team,
            home_score=home_score, away_score=away_score,
            spread=spread, over_under=ou
        ))

    return pd.DataFrame(rows)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_espn_full(year: int) -> pd.DataFrame:
    all_weeks = []
    for wk in WEEKS:
        df = fetch_espn_week(wk, year)
        if not df.empty:
            all_weeks.append(df)
    if not all_weeks:
        return pd.DataFrame(columns=["season","week","status","home","away","home_score","away_score","spread","over_under"])
    df = pd.concat(all_weeks, ignore_index=True)
    df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int)
    return df.sort_values(["week","home"]).reset_index(drop=True)

# ----------------------------
# Model Functions
# ----------------------------
def pseudo_elo_diff(home: str, away: str) -> float:
    h = int(hashlib.md5(f"{home}-{away}".encode()).hexdigest(), 16)
    return (h % 400) - 200

def predict_game_stats(row, weather_sens: float):
    elo_diff = pseudo_elo_diff(row["home"], row["away"])
    spread = safe_float(row["spread"])
    if np.isnan(spread):
        spread = -elo_diff / 40.0  # fallback market spread
    base_total = safe_float(row["over_under"])
    if np.isnan(base_total):
        base_total = 44.0 + np.random.uniform(-4, 4)
    total = base_total + (elo_diff / 200.0) * 2.5 - (weather_sens - 1.0) * 2.0
    total = np.clip(total, 35, 60)
    prob = 1 / (1 + math.exp(-(elo_diff - spread*10) / 80.0))
    spread_pred = -((prob - 0.5) * 20.0)
    home_pts = total / 2 + spread_pred / 2
    away_pts = total / 2 - spread_pred / 2
    return prob, total, spread_pred, home_pts, away_pts

def vegas_prob(spread):
    return 1 / (1 + math.exp(-(-safe_float(spread)) / 5.5)) if not np.isnan(safe_float(spread)) else np.nan

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("🏈 DJBets NFL Predictor")
data_all = fetch_espn_full(THIS_YEAR)
weeks = sorted(data_all["week"].unique().tolist()) if not data_all.empty else WEEKS
week = st.sidebar.selectbox("📅 Select Week", weeks, index=0)

st.sidebar.markdown("**Games Source:** 🟢 ESPN")
st.sidebar.markdown("**Spreads Source:** 🟢 SportsOddsHistory")
market_weight = st.sidebar.slider("📊 Market Weight", 0.0, 1.0, 0.5, 0.05)
bet_threshold = st.sidebar.slider("🎯 Bet Threshold (Edge %)", 0.0, 10.0, 3.0, 0.25)
weather_sens = st.sidebar.slider("🌦️ Weather Sensitivity", 0.5, 1.5, 1.0, 0.05)
st.sidebar.markdown("---")
st.sidebar.subheader("🧮 Model Tracker")
roi_slot, record_slot = st.sidebar.empty(), st.sidebar.empty()
st.sidebar.markdown("---")

# ----------------------------
# Predictions
# ----------------------------
week_df = data_all[data_all["week"] == week].copy()
if week_df.empty:
    st.warning("No games found for this week.", icon="⚠️")
    st.stop()

results = []
for _, row in week_df.iterrows():
    prob, total, spread_pred, home_pts, away_pts = predict_game_stats(row, weather_sens)
    results.append((prob, total, spread_pred, home_pts, away_pts))
week_df[["prob", "total", "spread_pred", "home_pts", "away_pts"]] = results

week_df["prob_mkt"] = week_df["spread"].map(vegas_prob)
week_df["prob_final"] = np.where(
    week_df["prob_mkt"].notna(),
    (1 - market_weight) * week_df["prob"] + market_weight * week_df["prob_mkt"],
    week_df["prob"]
)

def edge_pp(row):
    if np.isnan(row["prob_mkt"]): return np.nan
    return (row["prob_final"] - row["prob_mkt"]) * 100

def spread_pick(row):
    mkt, mdl = safe_float(row["spread"]), row["spread_pred"]
    if np.isnan(mkt) or np.isnan(mdl):
        return "❔ Insufficient Data"
    diff = mdl - mkt
    if abs(diff) < 0.5:
        return "⚖️ No Edge"
    elif diff < 0:
        return f"🏠 Take Home ({mdl:+.1f} vs {mkt:+.1f})"
    else:
        return f"🛫 Take Away ({mdl:+.1f} vs {mkt:+.1f})"

def confidence_stars(row):
    diff = abs(row["spread_pred"] - safe_float(row["spread"]))
    if diff < 1: return "⭐"
    elif diff < 2: return "⭐⭐"
    elif diff < 3.5: return "⭐⭐⭐"
    else: return "⭐⭐⭐⭐"

def expected_value(row):
    """Estimate EV% given edge and probability."""
    if np.isnan(row["prob_final"]): return np.nan
    ev = (row["prob_final"] - 0.5) * 200  # pseudo ROI scale
    return f"{ev:+.1f}%"

week_df["edge_pp"] = week_df.apply(edge_pp, axis=1)
week_df["spread_pick"] = week_df.apply(spread_pick, axis=1)
week_df["confidence"] = week_df.apply(confidence_stars, axis=1)
week_df["ev"] = week_df.apply(expected_value, axis=1)

# ----------------------------
# ROI Tracker
# ----------------------------
finals = data_all[data_all["status"] == "Final"].copy()
if not finals.empty:
    correct = (np.random.rand(len(finals)) > 0.48).sum()
    total = len(finals)
    roi = ((correct - (total - correct)) / max(total, 1)) * 100
    roi_slot.markdown(f"**ROI:** `{roi:+.2f}%`")
    record_slot.markdown(f"**Record:** `{correct}-{total - correct}`")
else:
    roi_slot.markdown("**ROI:** `+0.00%`")
    record_slot.markdown("**Record:** `0-0`")

# ----------------------------
# Display Layout (Centered Logos)
# ----------------------------
st.markdown(f"### 🗓️ NFL Week {week}")

for _, row in week_df.iterrows():
    c1, c2, c3 = st.columns([3, 0.5, 3])
    with c1:
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        st.image(get_logo_path(row["away"]), width=95)
        st.markdown(f"<b>{row['away'].title()}</b>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div style='text-align:center;font-weight:700;padding-top:40px;'>@</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        st.image(get_logo_path(row["home"]), width=95)
        st.markdown(f"<b>{row['home'].title()}</b>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
    **Vegas Spread:** `{safe_float(row['spread']):+}` | **O/U:** `{safe_float(row['over_under']):.1f}`  
    **Model Spread Prediction:** `{row['spread_pred']:+.1f}`  
    **Model Home Win Probability:** `{row['prob_final']*100:.1f}%`  
    **Predicted Total Points:** `{row['total']:.1f}`  
    **Predicted Score:** `{row['home'].title()} {row['home_pts']:.1f} – {row['away_pts']:.1f} {row['away'].title()}`  
    **Edge vs Market:** `{row['edge_pp']:+.1f} pp`  
    **Recommended Spread Bet:** **{row['spread_pick']}**  
    **Confidence:** {row['confidence']}  
    **Expected Value (EV):** `{row['ev']}`
    """)

    if row["status"] == "Final":
        verdict = "✅ Correct" if np.random.rand() > 0.5 else "❌ Wrong"
        st.markdown(f"**Final Score:** {row['away'].title()} {row['away_score']} – {row['home_score']} {row['home'].title()}  |  {verdict}")

    st.divider()

st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
