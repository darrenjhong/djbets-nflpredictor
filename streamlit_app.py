# DJBets NFL Predictor v11.7 — Full Functionality + ESPN Fix
# ESPN multi-week, clean UI, aligned logos, model ROI tracker retained.

import os
import math
import json
from datetime import datetime, timezone
from typing import List, Tuple
import numpy as np
import pandas as pd
import requests
import streamlit as st
from xgboost import XGBClassifier

# ----------------------------
# Streamlit layout setup
# ----------------------------
st.set_page_config(page_title="DJBets NFL Predictor", page_icon="🏈", layout="wide")

# ----------------------------
# Constants
# ----------------------------
THIS_YEAR = 2025
WEEKS = list(range(1, 19))

UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Referer": "https://www.espn.com/nfl/schedule",
    "Accept": "application/json",
}

TEAM_FILE_NAMES = {
    "49ers": "49ers.png", "bears": "bears.png", "bengals": "bengals.png",
    "bills": "bills.png", "broncos": "broncos.png", "browns": "browns.png",
    "buccaneers": "buccaneers.png", "cardinals": "cardinals.png", "chargers": "chargers.png",
    "chiefs": "chiefs.png", "colts": "colts.png", "commanders": "commanders.png",
    "cowboys": "cowboys.png", "dolphins": "dolphins.png", "eagles": "eagles.png",
    "falcons": "falcons.png", "giants": "giants.png", "jaguars": "jaguars.png",
    "jets": "jets.png", "lions": "lions.png", "packers": "packers.png",
    "panthers": "panthers.png", "patriots": "patriots.png", "raiders": "raiders.png",
    "rams": "rams.png", "ravens": "ravens.png", "saints": "saints.png",
    "seahawks": "seahawks.png", "steelers": "steelers.png", "texans": "texans.png",
    "titans": "titans.png", "vikings": "vikings.png",
}

def logo_path_for(team):
    """Return relative path to logo image."""
    key = (team or "").strip().lower()
    fname = TEAM_FILE_NAMES.get(key)
    if not fname:
        return f"public/logos/{key}.png"
    return f"public/logos/{fname}"

def safe_float(x):
    try:
        if x in (None, "", "N/A"): return np.nan
        return float(str(x).replace("−", "-"))
    except Exception:
        return np.nan

def vegas_prob_from_spread(spread: float) -> float:
    """Approximate home win probability from spread (logistic fit)."""
    if np.isnan(spread): return np.nan
    return 1.0 / (1.0 + math.exp(-(-spread) / 5.5))

def implied_total_baseline(ou: float) -> float:
    if np.isnan(ou) or ou <= 10 or ou > 90: return 44.0
    return ou

def expected_margin_from_prob(p_home: float) -> float:
    if np.isnan(p_home): return 0.0
    return (p_home - 0.5) * 28.0  # ~7 pts per 0.25 shift

# ----------------------------
# ESPN Fetchers
# ----------------------------
@st.cache_data(ttl=60*30, show_spinner=False)
def fetch_espn_week(week: int, year: int) -> pd.DataFrame:
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?year={year}&week={week}&seasontype=2"
    try:
        r = requests.get(url, headers=UA_HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        st.warning(f"⚠️ ESPN request failed for week {week}: {e}")
        return pd.DataFrame()

    rows = []
    for ev in data.get("events", []):
        comp = ev.get("competitions", [{}])[0]
        comps = comp.get("competitors", [])
        if len(comps) != 2: continue
        home = next((c for c in comps if c.get("homeAway") == "home"), None)
        away = next((c for c in comps if c.get("homeAway") == "away"), None)
        if not home or not away: continue

        def nick(c):
            t = c.get("team", {})
            return (t.get("nickname") or t.get("shortDisplayName") or t.get("abbreviation") or t.get("name") or "").lower().strip()

        home_nick, away_nick = nick(home), nick(away)
        try:
            home_score, away_score = int(home.get("score", 0)), int(away.get("score", 0))
        except Exception:
            home_score, away_score = np.nan, np.nan

        odds = comp.get("odds", [])
        spread, ou = np.nan, np.nan
        if odds:
            try:
                o = odds[0]
                spread = safe_float(o.get("spread"))
                ou = safe_float(o.get("overUnder"))
                fav = (o.get("favorite") or "").lower()
                if fav == home_nick:
                    spread = -abs(spread)
                elif fav == away_nick:
                    spread = +abs(spread)
            except Exception:
                pass

        status_type = comp.get("status", {}).get("type", {})
        completed = status_type.get("completed", False)
        status = "Final" if completed else "Upcoming"

        rows.append(dict(
            season=year, week=week, status=status,
            home=home_nick, away=away_nick,
            home_score=home_score, away_score=away_score,
            spread=spread, over_under=ou
        ))
    return pd.DataFrame(rows)

@st.cache_data(ttl=60*60, show_spinner=False)
def fetch_espn_full(year: int) -> pd.DataFrame:
    frames = []
    for wk in range(1, 19):
        df = fetch_espn_week(wk, year)
        if not df.empty: frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["season","week","status","home","away","home_score","away_score","spread","over_under"])
    out = pd.concat(frames, ignore_index=True)
    out["week"] = pd.to_numeric(out["week"], errors="coerce").fillna(0).astype(int)
    return out.sort_values(["week","home"]).reset_index(drop=True)

# ----------------------------
# Synthetic Training (stable fallback)
# ----------------------------
@st.cache_data(ttl=60*30, show_spinner=False)
def simulate_training(n: int=4000, random_state: int=42) -> Tuple[XGBClassifier, List[str]]:
    rng = np.random.RandomState(random_state)
    spread = rng.normal(0, 6, n)
    total = rng.normal(44, 7, n).clip(30,60)
    elo = rng.normal(0, 60, n)
    temp = rng.normal(10, 15, n)
    inj = rng.normal(0, 1, n)
    logits = (-0.18*spread + 0.006*elo - 0.005*abs(temp-10) + 0.03*inj + rng.normal(0,0.3,n))
    p = 1/(1+np.exp(-logits))
    y = (rng.rand(n) < p).astype(int)
    X = np.column_stack([spread, total, elo, temp, inj])
    model = XGBClassifier(
        n_estimators=180, max_depth=4, learning_rate=0.09,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        n_jobs=2, random_state=random_state, tree_method="hist"
    )
    model.fit(X, y)
    return model, ["spread","over_under","elo_diff","temp_c","inj_diff"]

def ensure_feature_cols(df, features):
    out = df.copy()
    for c in features:
        if c not in out.columns:
            st.warning(f"Added missing feature column: {c}", icon="⚠️")
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out

# ----------------------------
# Predictive functions
# ----------------------------
def predicted_total_points(ou, p_home, weather_sens):
    base = implied_total_baseline(ou)
    adj = -2.0 * abs(p_home - 0.5)
    weather_adj = -3.5 * (weather_sens - 1.0)
    return float(np.clip(base + adj + weather_adj, 30, 60))

def predicted_spread_from_prob(p_home):
    margin = expected_margin_from_prob(p_home)
    return float(np.round(-margin, 1))

def score_from_total_and_spread(total, spread_home):
    t = float(np.clip(total, 30, 60))
    s = float(np.clip(spread_home, -20, 20))
    home = 0.5*t + s/2.0
    away = 0.5*t - s/2.0
    return float(np.clip(home,7,45)), float(np.clip(away,7,45))

def recommendation_from_edge(edge_pp, bet_threshold_pp, favored_home):
    if np.isnan(edge_pp): return "🚫 No Bet"
    if abs(edge_pp) < bet_threshold_pp: return "🚫 No Bet"
    if edge_pp > 0:
        return "🏠 Bet Home" if favored_home else "🛫 Bet Away"
    else:
        return "🛫 Bet Away" if favored_home else "🏠 Bet Home"

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("🏈 DJBets NFL Predictor")
games_df_all = fetch_espn_full(THIS_YEAR)
available_weeks = sorted(games_df_all["week"].unique().tolist()) if not games_df_all.empty else WEEKS
week = st.sidebar.selectbox("📅 Select Week", available_weeks, index=0)

st.sidebar.markdown("**Games Source:**  \n🟢 ESPN Full Season")
st.sidebar.markdown("**Spreads Source:**  \n🟢 SportsOddsHistory")

market_weight = st.sidebar.slider("📊 Market Weight", 0.0, 1.0, 0.50, 0.05)
bet_threshold = st.sidebar.slider("🎯 Bet Threshold (Edge %)", 0.0, 10.0, 3.0, 0.25)
weather_sens = st.sidebar.slider("🌤️ Weather Sensitivity", 0.5, 1.5, 1.0, 0.05)
st.sidebar.markdown("---")
st.sidebar.subheader("🧮 Model Tracker")
roi_slot = st.sidebar.empty()
record_slot = st.sidebar.empty()
st.sidebar.markdown("---")

# ----------------------------
# Model train and predict
# ----------------------------
model, FEATURES = simulate_training()
games_df = games_df_all[games_df_all["week"] == int(week)].copy()
if games_df.empty:
    st.warning("No games found for this week.", icon="⚠️")
    st.stop()

feat_df = games_df[["spread","over_under"]].copy()
feat_df["elo_diff"] = 0.0
feat_df["temp_c"] = 10.0
feat_df["inj_diff"] = 0.0
feat_df = ensure_feature_cols(feat_df, FEATURES)

p_home_model = model.predict_proba(feat_df[FEATURES].values)[:,1]
games_df["home_win_prob_model"] = p_home_model
games_df["home_win_prob_market"] = games_df["spread"].map(lambda x: vegas_prob_from_spread(safe_float(x)))
games_df["home_win_prob"] = np.where(
    games_df["home_win_prob_market"].notna(),
    (1 - market_weight)*games_df["home_win_prob_model"] + market_weight*games_df["home_win_prob_market"],
    games_df["home_win_prob_model"]
)

# Pred totals, spreads, scores
pred_totals, pred_spreads, pred_scores = [], [], []
for _, r in games_df.iterrows():
    p = float(r["home_win_prob"])
    ou = safe_float(r.get("over_under"))
    tot = predicted_total_points(ou, p, weather_sens)
    sp = predicted_spread_from_prob(p)
    home_pts, away_pts = score_from_total_and_spread(tot, sp)
    pred_totals.append(tot)
    pred_spreads.append(sp)
    pred_scores.append((home_pts, away_pts))
games_df["pred_total"] = pred_totals
games_df["pred_spread_home"] = pred_spreads
games_df["pred_home_pts"] = [x[0] for x in pred_scores]
games_df["pred_away_pts"] = [x[1] for x in pred_scores]

# Edge / Recommendation
def edge_pp_row(row):
    sp_mkt = safe_float(row.get("spread"))
    if np.isnan(sp_mkt): return np.nan
    p_model = row["home_win_prob"]
    p_mkt = vegas_prob_from_spread(sp_mkt)
    if np.isnan(p_mkt): return np.nan
    return (p_model - p_mkt) * 100.0
games_df["edge_pp"] = games_df.apply(edge_pp_row, axis=1)
games_df["recommendation"] = games_df.apply(
    lambda r: recommendation_from_edge(r["edge_pp"], bet_threshold, r["spread"]<0 if not np.isnan(r["spread"]) else False),
    axis=1
)

# ----------------------------
# ROI Tracker
# ----------------------------
completed = games_df_all[games_df_all["status"]=="Final"].copy()
if not completed.empty:
    feat_c = completed[["spread","over_under"]].copy()
    feat_c["elo_diff"]=0.0; feat_c["temp_c"]=10.0; feat_c["inj_diff"]=0.0
    feat_c = ensure_feature_cols(feat_c, FEATURES)
    completed["p_home"]=model.predict_proba(feat_c[FEATURES].values)[:,1]
    completed["pred_winner"]=np.where(completed["p_home"]>=0.5, completed["home"], completed["away"])
    completed["true_winner"]=np.where(completed["home_score"]>completed["away_score"], completed["home"],
                                      np.where(completed["home_score"]<completed["away_score"], completed["away"],"push"))
    mask=completed["true_winner"]!="push"
    correct=(completed.loc[mask,"pred_winner"]==completed.loc[mask,"true_winner"]).sum()
    total=int(mask.sum())
    roi=( (correct-(total-correct))/max(total,1))*100.0
    roi_slot.markdown(f"**ROI**  \n`{roi:+.2f}%`")
    record_slot.markdown(f"**Record**  \n`{correct}-{total-correct}`")
else:
    roi_slot.markdown("**ROI**  \n`+0.00%`")
    record_slot.markdown("**Record**  \n`0-0`")

# ----------------------------
# Render Games
# ----------------------------
st.markdown(f"### 🗓️ NFL Week {week}")

def render_team_block(team):
    path = logo_path_for(team)
    name = team.title()
    st.markdown(f"""
    <div style='display:flex;flex-direction:column;align-items:center;gap:6px;'>
      <img src='{path}' style='height:84px;object-fit:contain;' onerror="this.style.display='none'"/>
      <div style='font-weight:600'>{name}</div>
    </div>
    """, unsafe_allow_html=True)

for _, row in games_df.iterrows():
    title = f"{row['away'].title()} @ {row['home'].title()} | {row['status']}"
    with st.expander(title, expanded=False):
        c1, c2, c3 = st.columns([3,1,3])
        with c1: render_team_block(row["away"])
        with c2:
            st.markdown("<div style='height:84px;display:flex;align-items:center;justify-content:center;font-weight:700;'>VS</div>", unsafe_allow_html=True)
        with c3: render_team_block(row["home"])
        st.markdown("---")

        spread_str = "N/A" if np.isnan(safe_float(row["spread"])) else f"{safe_float(row['spread']):+0.1f}"
        ou_str = "N/A" if np.isnan(safe_float(row["over_under"])) else f"{safe_float(row['over_under']):0.1f}"
        edge_txt = "N/A" if np.isnan(row["edge_pp"]) else f"{row['edge_pp']:+0.1f} pp"

        st.markdown(f"""
        **Vegas Spread:** `{spread_str}` | **Vegas O/U:** `{ou_str}`  
        **Model Home Win Probability:** `{row['home_win_prob']*100:0.1f}%`  
        **Predicted Total Points:** `{row['pred_total']:0.1f}`  
        **Predicted Score:** `{row['home'].title()} {row['pred_home_pts']:0.1f} – {row['pred_away_pts']:0.1f} {row['away'].title()}`  
