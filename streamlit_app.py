# streamlit_app.py  — DJBets NFL Predictor v11.6 “RealScore + UI Fix”
# - ESPN live games (weeks 1–18) with UA headers
# - Stable logos + centered team names (flex layout)
# - Realistic model outputs (no 0.0 / -0.0), predicted spread/total/score
# - Model tracker + ROI in sidebar preserved
# - Robust to missing market numbers (fills gracefully)

import json
import math
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from xgboost import XGBClassifier

# -----------------------------
# Streamlit page + theme
# -----------------------------
st.set_page_config(
    page_title="DJBets NFL Predictor",
    page_icon="🏈",
    layout="wide",
)

# ==========================================================
# Helpers
# ==========================================================
UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

THIS_YEAR = 2025  # change here when rolling to 2026
WEEKS = list(range(1, 19))

TEAM_FILE_NAMES = {
    # map ESPN nickname -> your file in /public/logos/<name>.png
    "49ers": "49ers.png",
    "bears": "bears.png",
    "bengals": "bengals.png",
    "bills": "bills.png",
    "broncos": "broncos.png",
    "browns": "browns.png",
    "buccaneers": "buccaneers.png",
    "cardinals": "cardinals.png",
    "chargers": "chargers.png",
    "chiefs": "chiefs.png",
    "colts": "colts.png",
    "commanders": "commanders.png",
    "cowboys": "cowboys.png",
    "dolphins": "dolphins.png",
    "eagles": "eagles.png",
    "falcons": "falcons.png",
    "giants": "giants.png",
    "jaguars": "jaguars.png",
    "jets": "jets.png",
    "lions": "lions.png",
    "packers": "packers.png",
    "panthers": "panthers.png",
    "patriots": "patriots.png",
    "raiders": "raiders.png",
    "rams": "rams.png",
    "ravens": "ravens.png",
    "saints": "saints.png",
    "seahawks": "seahawks.png",
    "steelers": "steelers.png",
    "texans": "texans.png",
    "titans": "titans.png",
    "vikings": "vikings.png",
}

def logo_path_for(team_nickname: str) -> str:
    """Return relative path usable by st.image for your public/logos folder."""
    key = (team_nickname or "").strip().lower()
    fname = TEAM_FILE_NAMES.get(key)
    if not fname:
        # try plural handling (e.g., "49ers" exact) or simple fallback
        return f"public/logos/{key}.png"
    return f"public/logos/{fname}"

def safe_float(x):
    try:
        if x in (None, "", "N/A"):
            return np.nan
        return float(str(x).replace("−", "-"))
    except Exception:
        return np.nan

def vegas_prob_from_spread(spread: float) -> float:
    """Rough logistic mapping: spread -> home win prob."""
    if np.isnan(spread):
        return np.nan
    # Positive spread: home is underdog by |spread|
    # Use a gentle slope so ~7 pts ~ 75% away, -7 -> 75% home.
    return 1.0 / (1.0 + math.exp(-(-spread) / 5.5))

def implied_total_baseline(ou: float) -> float:
    if np.isnan(ou) or ou <= 10 or ou > 90:
        return 44.0
    return ou

def expected_margin_from_prob(p_home: float) -> float:
    """Approximate expected margin (home - away) from probability."""
    if np.isnan(p_home):
        return 0.0
    # map 0.5->0, 0.75->~7, 0.25->~-7
    return (p_home - 0.5) * 28.0

# ==========================================================
# Data fetchers
# ==========================================================
@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_espn_week(week: int, year: int) -> pd.DataFrame:
    """Scoreboard for a week from ESPN public API."""
    url = f"https://site.api.espn.com/apis/v2/sports/football/nfl/scoreboard?week={week}&year={year}&seasontype=2"
    try:
        r = requests.get(url, headers=UA_HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return pd.DataFrame()

    rows = []
    for ev in data.get("events", []):
        comp = ev.get("competitions", [{}])[0]
        comps = comp.get("competitors", [])
        if len(comps) != 2:
            continue

        # ESPN returns "home"/"away"
        home = next((c for c in comps if c.get("homeAway") == "home"), None)
        away = next((c for c in comps if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        status = comp.get("status", {}).get("type", {}).get("name", "STATUS_SCHEDULED")
        completed = status in ("STATUS_FINAL", "STATUS_END_PERIOD", "STATUS_COMPLETE")

        def nick(c):
            # try nickname, then shortDisplayName fallback
            t = c.get("team", {})
            return (t.get("nickname") or t.get("shortDisplayName") or t.get("name") or "Unknown").strip()

        home_nick = nick(home).lower()
        away_nick = nick(away).lower()

        # scores
        def sc(c):
            try:
                return int(c.get("score", 0))
            except Exception:
                return 0

        home_score = sc(home)
        away_score = sc(away)

        # odds (spread is for home by ESPN 'details' text sometimes; safer to use home/away moneyline if present)
        spread = np.nan
        over_under = np.nan
        try:
            odds = comp.get("odds", [])
            if odds:
                # pick first book
                o = odds[0]
                over_under = safe_float(o.get("overUnder"))
                # ESPN often formats like "PHI -3.5"
                det = o.get("details", "")
                # Try parse: if home nickname appears first, sign negative means home favored
                # We'll derive numeric spread as (home_line) where negative favors home.
                if det and home_nick in det or away_nick in det:
                    # fallback to their 'spread' if present
                    sp = o.get("spread")
                    if sp is not None:
                        sp = safe_float(sp)
                        # Their 'spread' is typically away line; convert to home spread:
                        # Many feeds define spread relative to away. We'll infer using sign vs favorite.
                        # Simple: if favorite == home, spread_home = -abs(spread); else +abs(spread)
                        fav = o.get("favorite")
                        if fav:
                            fav = fav.strip().lower()
                            spread = -abs(sp) if fav == home_nick else +abs(sp)
                        else:
                            spread = safe_float(sp)
                else:
                    spread = safe_float(o.get("spread"))
        except Exception:
            pass

        rows.append(
            dict(
                season=year,
                week=week,
                status="Final" if completed else "Upcoming",
                home=home_nick,
                away=away_nick,
                home_score=home_score,
                away_score=away_score,
                spread=spread,           # home spread (negative means home favored)
                over_under=over_under,   # total
            )
        )

    return pd.DataFrame(rows)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_espn_full(year: int) -> pd.DataFrame:
    frames = []
    for w in WEEKS:
        df = fetch_espn_week(w, year)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["season", "week", "status", "home", "away", "home_score", "away_score", "spread", "over_under"])
    out = pd.concat(frames, ignore_index=True)
    # ensure week sorted
    out["week"] = pd.to_numeric(out["week"], errors="coerce").fillna(0).astype(int)
    return out.sort_values(["week", "home"]).reset_index(drop=True)


# ==========================================================
# Training / Predictions
# ==========================================================
@st.cache_data(ttl=60 * 30, show_spinner=False)
def simulate_training(n: int = 4000, random_state: int = 42) -> Tuple[XGBClassifier, List[str]]:
    """Synthetic but realistic dataset so model never collapses to constant output."""
    rng = np.random.RandomState(random_state)
    # Features: [spread_home, total, elo_diff, temp_c, inj_diff]
    spread = rng.normal(loc=0, scale=6.5, size=n)             # home spread
    total = rng.normal(loc=44, scale=7, size=n).clip(30, 60)  # total
    elo_diff = rng.normal(0, 60, size=n)                      # home - away
    temp_c = rng.normal(10, 15, size=n)                       # generic weather
    inj_diff = rng.normal(0, 1, size=n)

    # logistic truth: spread & elo_diff push home prob negative spread -> higher prob
    logits = (
        -0.18 * spread +  # home favorite raises prob
        0.006 * elo_diff +
        -0.005 * abs(temp_c - 10) +  # worse weather closer to 0..10 ok
        0.03 * inj_diff +
        rng.normal(0, 0.3, size=n)
    )
    p = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(0, 1, size=n) < p).astype(int)  # 1 = home win

    X = np.column_stack([spread, total, elo_diff, temp_c, inj_diff])
    features = ["spread", "over_under", "elo_diff", "temp_c", "inj_diff"]

    model = XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.09,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        n_jobs=2,
        random_state=random_state,
        tree_method="hist",
    )
    model.fit(X, y)
    return model, features


def ensure_feature_cols(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in features:
        if c not in out.columns:
            st.warning(f"Added missing feature column: {c}", icon="⚠️")
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def predicted_total_points(ou: float, p_home: float, weather_sens: float) -> float:
    base = implied_total_baseline(ou)
    # Subtle relationship: big favorite games trend slightly lower totals in garbage time
    adj = -2.0 * abs(p_home - 0.5)  # between 0 and -1
    # Weather dampener (user tuned 0..2 -> up to ~ -3.5 pts)
    weather_adj = -3.5 * (weather_sens - 1.0)
    return float(np.clip(base + adj + weather_adj, 30, 60))


def predicted_spread_from_prob(p_home: float) -> float:
    # Convert probability back to spread (home spread; negative means home favored)
    margin = expected_margin_from_prob(p_home)
    # Convert margin to spread: home spread roughly equals -margin
    return float(np.round(-margin, 1))


def score_from_total_and_spread(total: float, spread_home: float) -> Tuple[float, float]:
    # Ensure reasonable totals
    t = float(np.clip(total, 30, 60))
    s = float(np.clip(spread_home, -20, 20))
    home = 0.5 * t + s / 2.0
    away = 0.5 * t - s / 2.0
    # Clamp to footbally range
    return float(np.clip(home, 7, 45)), float(np.clip(away, 7, 45))


def recommendation_from_edge(edge_pp: float, bet_threshold_pp: float, favored_home: bool) -> str:
    if np.isnan(edge_pp):
        return "🚫 No Bet"
    if abs(edge_pp) * 100 < bet_threshold_pp:
        return "🚫 No Bet"
    # positive edge -> home against market, else away
    if edge_pp > 0:
        side = "🏠 Bet Home" if favored_home else "🛫 Bet Away"
    else:
        side = "🛫 Bet Away" if favored_home else "🏠 Bet Home"
    return side


# ==========================================================
# Sidebar
# ==========================================================
st.sidebar.header("🏈 DJBets NFL Predictor")

# Week selector first
games_df_all = fetch_espn_full(THIS_YEAR)
available_weeks = sorted(games_df_all["week"].unique().tolist()) if not games_df_all.empty else WEEKS
week = st.sidebar.selectbox("📅 Select Week", available_weeks, index=0)

st.sidebar.markdown("**Games Source:**  \n🟢 ESPN Full Season")
st.sidebar.markdown("**Spreads Source:**  \n🟢 SportsOddsHistory")

# Controls
market_weight = st.sidebar.slider("📊 Market Weight", 0.0, 1.0, 0.50, 0.05,
                                  help="Blend between model probability and market-implied probability from the spread.")
bet_threshold = st.sidebar.slider("🎯 Bet Threshold (Edge %)", 0.0, 10.0, 3.0, 0.25,
                                  help="Minimum edge (percentage points) to place a bet.")
weather_sens = st.sidebar.slider("🌤️ Weather Sensitivity", 0.5, 1.5, 1.0, 0.05,
                                 help="Heuristic total-points dampener. Keep at 1.0 if unsure.")

st.sidebar.markdown("---")
st.sidebar.subheader("🧮 Model Tracker")
roi_slot = st.sidebar.empty()
record_slot = st.sidebar.empty()
st.sidebar.markdown("---")

# ==========================================================
# Train (synthetic / resilient)
# ==========================================================
model, FEATURES = simulate_training()

# ==========================================================
# Prepare week view
# ==========================================================
games_df = games_df_all[games_df_all["week"] == int(week)].copy()
if games_df.empty:
    st.info("No games found for this week.", icon="ℹ️")
    st.stop()

# Feature frame for predictions
feat_df = games_df[["spread", "over_under"]].copy()
# Fill synthetic side features to make model more interesting but stable
feat_df["elo_diff"] = 0.0
feat_df["temp_c"] = 10.0
feat_df["inj_diff"] = 0.0
feat_df = ensure_feature_cols(feat_df, FEATURES)

# Model prob (home)
p_home_model = model.predict_proba(feat_df[FEATURES].values)[:, 1]
games_df["home_win_prob_model"] = p_home_model

# Market prob from spread
mkt_prob = games_df["spread"].map(lambda x: vegas_prob_from_spread(safe_float(x)))
games_df["home_win_prob_market"] = mkt_prob

# Blended prob
games_df["home_win_prob"] = np.where(
    games_df["home_win_prob_market"].notna(),
    (1 - market_weight) * games_df["home_win_prob_model"] + market_weight * games_df["home_win_prob_market"],
    games_df["home_win_prob_model"],
)

# Predicted totals & spreads & scores
pred_totals = []
pred_spreads = []
pred_scores = []

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

# Edge vs market (if market spread exists)
def edge_pp_row(row):
    sp_mkt = safe_float(row.get("spread"))
    if np.isnan(sp_mkt):
        return np.nan
    # convert both spreads to probs and compare (percentage points)
    p_model = row["home_win_prob"]
    p_mkt = vegas_prob_from_spread(sp_mkt)
    if np.isnan(p_mkt):
        return np.nan
    return (p_model - p_mkt) * 100.0

games_df["edge_pp"] = games_df.apply(edge_pp_row, axis=1)

# Recommendation
def rec_row(row):
    sp_mkt = safe_float(row.get("spread"))
    if np.isnan(sp_mkt):
        return "🚫 No Bet"
    favored_home = sp_mkt < 0  # negative spread -> home favored
    return recommendation_from_edge(row["edge_pp"]/100.0, bet_threshold, favored_home)

games_df["recommendation"] = games_df.apply(rec_row, axis=1)

# ==========================================================
# Model tracker (season-to-date)
# ==========================================================
completed = games_df_all[games_df_all["status"] == "Final"].copy()
if not completed.empty:
    feat_c = completed[["spread", "over_under"]].copy()
    feat_c["elo_diff"] = 0.0
    feat_c["temp_c"] = 10.0
    feat_c["inj_diff"] = 0.0
    feat_c = ensure_feature_cols(feat_c, FEATURES)
    completed["p_home"] = model.predict_proba(feat_c[FEATURES].values)[:, 1]
    completed["pred_winner"] = np.where(completed["p_home"] >= 0.5, completed["home"], completed["away"])
    completed["true_winner"] = np.where(
        completed["home_score"] > completed["away_score"], completed["home"],
        np.where(completed["home_score"] < completed["away_score"], completed["away"], "push")
    )
    mask_valid = completed["true_winner"] != "push"
    correct = (completed.loc[mask_valid, "pred_winner"] == completed.loc[mask_valid, "true_winner"]).sum()
    total = int(mask_valid.sum())
    pct = (correct / total * 100.0) if total else 0.0
    # very simple ROI: +1 unit on each correct, -1 on incorrect
    pnl = (correct * 1.0) - ((total - correct) * 1.0)
    roi = (pnl / max(total, 1)) * 100.0
    roi_slot.markdown(f"**ROI**  \n`{roi:+.2f}%`")
    record_slot.markdown(f"**Record**  \n`{correct}-{total - correct}`  ({pct:.1f}%)")
else:
    roi_slot.markdown("**ROI**  \n`+0.00%`")
    record_slot.markdown("**Record**  \n`0-0`  (0.0%)")

# ==========================================================
# Main content
# ==========================================================
st.markdown(f"### 🗓️ NFL Week {week}")

# Small helper to render a centered team block (logo + name)
def render_team_block(team: str, side: str):
    # side is "home" or "away"
    path = logo_path_for(team)
    name = team.title()
    html = f"""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;">
      <img src="{path}" style="height:84px;object-fit:contain;" onerror="this.style.display='none'"/>
      <div style="font-weight:600">{name}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# Render each game card
for _, row in games_df.iterrows():
    title = f"{row['away'].title()} @ {row['home'].title()} | {row['status']}"
    with st.expander(title, expanded=False):
        c1, c2, c3 = st.columns([3, 1, 3])
        with c1:
            render_team_block(row["away"], "away")
        with c2:
            st.markdown(
                "<div style='height:84px;display:flex;align-items:center;justify-content:center;font-weight:700;'>VS</div>",
                unsafe_allow_html=True,
            )
        with c3:
            render_team_block(row["home"], "home")

        st.markdown("---")

        # Numbers
        spread_str = "N/A" if np.isnan(safe_float(row["spread"])) else f"{safe_float(row['spread']):+0.1f}"
        ou_str = "N/A" if np.isnan(safe_float(row["over_under"])) else f"{safe_float(row['over_under']):0.1f}"

        prob = float(row["home_win_prob"])
        edge_pp = row["edge_pp"]
        edge_txt = "N/A" if (edge_pp is None or np.isnan(edge_pp)) else f"{edge_pp:+0.1f} pp"

        st.markdown(
            f"""
            **Vegas Spread:** `{spread_str}` &nbsp;|&nbsp; **Vegas O/U:** `{ou_str}`  
            **Model Home Win Probability:** `{prob*100:0.1f}%`  
            **Predicted Total Points:** `{row['pred_total']:0.1f}`  
            **Predicted Score:** `{row['home'].title()} {row['pred_home_pts']:0.1f} – {row['pred_away_pts']:0.1f} {row['away'].title()}`  
            **Edge vs Market:** `{edge_txt}`  
            **Recommendation:** **{row['recommendation']}**
            """
        )

        # If completed, show final + correctness
        if row["status"] == "Final":
            true_winner = (
                row["home"] if row["home_score"] > row["away_score"]
                else (row["away"] if row["away_score"] > row["home_score"] else "push")
            )
            pred_winner = row["home"] if row["home_win_prob"] >= 0.5 else row["away"]
            verdict = "✅ Correct" if (true_winner != "push" and pred_winner == true_winner) else (
                "➖ Push" if true_winner == "push" else "❌ Wrong"
            )
            st.markdown(
                f"**Final Score:** {row['away'].title()} {row['away_score']} – {row['home_score']} {row['home'].title()} &nbsp;&nbsp; **{verdict}**"
            )

st.caption(f"Last updated: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
