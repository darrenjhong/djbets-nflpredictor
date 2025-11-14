# streamlit_app.py
"""
DJBets — NFL Predictor (Option A2)
- Primary schedule & odds: Covers scraping (covers_odds.fetch_covers_for_week)
- Scores enrichment (completed games): ESPN scoreboard (optional)
- Uses local historical archive (data/nfl_archive_10Y.json) if present to train
- Stable ML: sklearn LogisticRegression fallback model (no xgboost)
- Sidebar A minimal layout with week dropdown at top + small logos + sliders
- Expects team logos in public/logos/<canonical_team>.png (canonical via team_logo_map.canonical_team_name)
"""

import streamlit as st
st.set_page_config(page_title="DJBets NFL Predictor", layout="wide")

import os
import json
import time
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Local modules (must exist in repo)
# covers_odds.fetch_covers_for_week(year, week) -> DataFrame with columns ['home','away','spread','over_under']
# team_logo_map.canonical_team_name(name) -> canonical file base (e.g., chicago_bears)
from covers_odds import fetch_covers_for_week
from team_logo_map import canonical_team_name

# -----------------------
# Configuration
# -----------------------
CURRENT_SEASON = datetime.now().year
MAX_WEEKS = 18
LOGOS_DIR = "public/logos"
HISTORICAL_PATH = "data/nfl_archive_10Y.json"
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"  # used only for scores enrichment
MODEL_MIN_ROWS = 50  # min labelled rows to train properly
FALLBACK_ELO = 1500

# -----------------------
# Helper functions
# -----------------------
def get_logo_path(team_name: str) -> Optional[str]:
    """
    Return local logo path for a team name (canonical). If not found, return None.
    """
    if not team_name or str(team_name).strip() == "":
        return None
    canon = canonical_team_name(team_name)
    candidate = os.path.join(LOGOS_DIR, f"{canon}.png")
    if os.path.exists(candidate):
        return candidate
    # try jpg
    candidate2 = os.path.join(LOGOS_DIR, f"{canon}.jpg")
    if os.path.exists(candidate2):
        return candidate2
    return None

def safe_fetch_espn_scores(season:int, week:int) -> pd.DataFrame:
    """
    Optional: fetch ESPN scoreboard for a given season/week to obtain final scores/status.
    Defensive: returns empty df on error.
    """
    try:
        params = {"season": season, "seasontype": 2, "week": week}
        r = pd.read_json(f"{ESPN_SCOREBOARD_URL}?season={season}&seasontype=2&week={week}")
        # pandas read_json can be protective; but ESPN JSON structure is complex.
        # We'll defensively parse using requests if necessary.
    except Exception:
        # fallback to requests + json parsing
        try:
            import requests
            resp = requests.get(ESPN_SCOREBOARD_URL, params={"season": season, "seasontype": 2, "week": week}, timeout=6)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return pd.DataFrame()
        data = data or {}
        events = data.get("events", [])
        rows = []
        for ev in events:
            try:
                comp = ev.get("competitions", [])[0]
                competitors = comp.get("competitors", [])
                if len(competitors) != 2:
                    continue
                home = None
                away = None
                for c in competitors:
                    if c.get("homeAway") == "home":
                        home = c
                    else:
                        away = c
                if home is None or away is None:
                    continue
                # canonical names
                home_name = home.get("team", {}).get("displayName", "")
                away_name = away.get("team", {}).get("displayName", "")
                home_score = home.get("score", None)
                away_score = away.get("score", None)
                status = comp.get("status", {}).get("type", {}).get("name", "")
                rows.append({
                    "home_team": home_name,
                    "away_team": away_name,
                    "home_score": home_score,
                    "away_score": away_score,
                    "status": status
                })
            except Exception:
                continue
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # normalize names
        df["home_team"] = df["home_team"].astype(str).map(lambda x: x.lower().strip())
        df["away_team"] = df["away_team"].astype(str).map(lambda x: x.lower().strip())
        return df
    # if we reached here, pandas read_json was used but likely won't shape correctly; return empty
    return pd.DataFrame()

def load_local_history(path=HISTORICAL_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        with open(path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df.columns = [c.lower().strip() for c in df.columns]
        # make sure columns exist
        for col in ["home_team","away_team","home_score","away_score","season","week"]:
            if col not in df.columns:
                df[col] = np.nan
        # normalize team text
        df["home_team"] = df["home_team"].astype(str).map(lambda x: x.lower().strip())
        df["away_team"] = df["away_team"].astype(str).map(lambda x: x.lower().strip())
        return df
    except Exception:
        return pd.DataFrame()

def compute_simple_elo(history_df: pd.DataFrame, k=20, base=FALLBACK_ELO) -> dict:
    """
    Compute simple Elo ratings from historical games.
    history_df expected to contain: home_team, away_team, home_score, away_score
    Returns dict team -> elo
    """
    elos = {}
    # initialize
    for t in history_df["home_team"].dropna().unique().tolist() + history_df["away_team"].dropna().unique().tolist():
        if pd.notna(t):
            elos.setdefault(t, base)
    # iterate chronological if possible (we may not have dates)
    for _, row in history_df.iterrows():
        h = str(row.get("home_team","")).lower().strip()
        a = str(row.get("away_team","")).lower().strip()
        try:
            hs = float(row.get("home_score", np.nan))
            as_ = float(row.get("away_score", np.nan))
        except Exception:
            continue
        if h == "" or a == "" or np.isnan(hs) or np.isnan(as_):
            continue
        elos.setdefault(h, base)
        elos.setdefault(a, base)
        # expected
        diff = elos[h] - elos[a]
        exp_h = 1.0 / (1 + 10**((-diff)/400))
        # result
        res_h = 1.0 if hs > as_ else 0.5 if hs == as_ else 0.0
        # update
        elos[h] = elos[h] + k * (res_h - exp_h)
        elos[a] = elos[a] + k * ((1-res_h) - (1-exp_h))
    return elos

def prepare_week_schedule_from_covers(season:int, week:int) -> pd.DataFrame:
    """
    Use covers_odds.fetch_covers_for_week to build schedule rows.
    Expected output columns: season, week, home_team, away_team, spread, over_under
    home_team/away_team kept as lowercase display strings.
    """
    try:
        df = fetch_covers_for_week(season, week)
    except Exception:
        df = pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    # covers may return 'home'/'away' keys that are full names — normalize lowercase
    df = df.rename(columns={"home":"home_team","away":"away_team","over_under":"over_under","spread":"spread"})
    # normalize
    for c in ["home_team","away_team"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].astype(str).map(lambda x: x.lower().strip())
    # ensure numeric for spread/over_under
    if "spread" in df.columns:
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce")
    else:
        df["spread"] = np.nan
    if "over_under" in df.columns:
        df["over_under"] = pd.to_numeric(df["over_under"], errors="coerce")
    else:
        df["over_under"] = np.nan
    df["season"] = season
    df["week"] = week
    # add default columns
    df["home_score"] = np.nan
    df["away_score"] = np.nan
    df["status"] = "scheduled"
    return df[["season","week","home_team","away_team","spread","over_under","home_score","away_score","status"]]

# -----------------------
# Simple model training
# -----------------------
def build_feature_matrix(df: pd.DataFrame, elos: dict) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Input: dataframe with columns home_team, away_team, spread, over_under, home_score, away_score
    Output: X (n x f), y (n), feature_names (list)
    Features: elo_diff, spread, over_under
    """
    rows = []
    y = []
    for _, r in df.iterrows():
        h = str(r.get("home_team","")).lower().strip()
        a = str(r.get("away_team","")).lower().strip()
        if h == "" or a == "":
            continue
        # require completed games for label
        try:
            hs = r.get("home_score")
            as_ = r.get("away_score")
            if pd.isna(hs) or pd.isna(as_):
                continue
            hs = float(hs); as_ = float(as_)
        except Exception:
            continue
        # features
        elo_h = elos.get(h, FALLBACK_ELO)
        elo_a = elos.get(a, FALLBACK_ELO)
        elo_diff = elo_h - elo_a
        spread = float(r.get("spread", 0)) if pd.notna(r.get("spread", np.nan)) else 0.0
        ou = float(r.get("over_under", 0)) if pd.notna(r.get("over_under", np.nan)) else 0.0
        rows.append([elo_diff, spread, ou])
        y.append(1 if hs > as_ else 0)
    if not rows:
        return np.zeros((0,3)), np.zeros((0,)), ["elo_diff","spread","over_under"]
    X = np.array(rows)
    y = np.array(y)
    return X, y, ["elo_diff","spread","over_under"]

def train_model_from_history(history_df: pd.DataFrame) -> Tuple[Optional[object], dict]:
    """
    Train a LogisticRegression model from history. Returns (model, elos)
    If not enough labelled rows, returns (None, elos) to indicate fallback only.
    """
    # compute elos
    elos = compute_simple_elo(history_df)
    X, y, feats = build_feature_matrix(history_df, elos)
    if X.shape[0] < MODEL_MIN_ROWS:
        return None, elos
    # standardize and train
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=200)
    model.fit(Xs, y)
    # store meta in model object for convenience
    model._feats = feats
    model._scaler = scaler
    return model, elos

def predict_game_prob(model, elos:dict, row:pd.Series) -> Optional[float]:
    """
    Return probability that home team wins according to model.
    If model is None, return None (caller can use elo-derived probability)
    """
    h = str(row.get("home_team","")).lower().strip()
    a = str(row.get("away_team","")).lower().strip()
    elo_h = elos.get(h, FALLBACK_ELO)
    elo_a = elos.get(a, FALLBACK_ELO)
    elo_diff = elo_h - elo_a
    spread = float(row.get("spread", 0)) if pd.notna(row.get("spread", np.nan)) else 0.0
    ou = float(row.get("over_under", 0)) if pd.notna(row.get("over_under", np.nan)) else 0.0
    feat = np.array([[elo_diff, spread, ou]])
    if model is None:
        # approximate prob from elo_diff logistic function
        p = 1.0 / (1 + 10**((-elo_diff)/400))
        return p
    try:
        Xs = model._scaler.transform(feat)
        p = model.predict_proba(Xs)[0][1]
        return float(p)
    except Exception:
        # fallback to elo
        return 1.0 / (1 + 10**((-elo_diff)/400))

def format_kickoff_time(raw: Optional[str]) -> str:
    """
    Attempt to parse kickoff-like strings; fallback nicely.
    """
    if raw is None or raw == "" or pd.isna(raw):
        return ""
    try:
        # attempt ISO parse
        dt = pd.to_datetime(raw)
        return dt.strftime("%a %b %d %I:%M %p")
    except Exception:
        # return raw
        return str(raw)

# -----------------------
# ROI / record helpers
# -----------------------
def compute_model_record(history_df: pd.DataFrame, model, elos: dict) -> Tuple[int,int,float]:
    """
    Compute overall model record on completed historical games (correct/incorrect/%).
    """
    if history_df.empty:
        return 0,0,0.0
    correct = 0
    total = 0
    for _, r in history_df.iterrows():
        try:
            hs = r.get("home_score")
            as_ = r.get("away_score")
            if pd.isna(hs) or pd.isna(as_):
                continue
            total += 1
            prob = predict_game_prob(model, elos, r)
            pred_home = prob >= 0.5
            actual_home = float(hs) > float(as_)
            if pred_home == actual_home:
                correct += 1
        except Exception:
            continue
    if total == 0:
        return 0,0,0.0
    return correct, total-correct, correct/total*100.0

# -----------------------
# UI / App start
# -----------------------
st.title("🏈 DJBets — NFL Predictor")
st.markdown("Minimal, dark-ish predictions UI (Covers primary, ESPN scores optional).")

# Sidebar A: Week dropdown at top, minimal logos + model sliders + model record
with st.sidebar:
    st.markdown("## 📅 Week")
    available_weeks = list(range(1, MAX_WEEKS+1))
    current_week = st.selectbox("", available_weeks, index=0, key="week_selector")
    st.markdown("---")
    st.markdown("⚙️ **Model Controls**")
    market_weight = st.slider("Market weight (blend model <> market)", 0.0, 1.0, 0.0, step=0.05)
    bet_threshold = st.slider("Bet threshold (edge pts)", 0.0, 30.0, 8.0, step=0.5)
    st.markdown("---")
    st.markdown("📊 **Model Record**")
    # small placeholders while loading
    st.info("Loading model & history...")

# Load history (local)
history_df = load_local_history(HISTORICAL_PATH)
if history_df.empty:
    st.sidebar.warning("No local historical file found (data/nfl_archive_10Y.json). Using fallback simulated model.")
else:
    st.sidebar.success(f"Historical rows: {len(history_df)}")

# Train model (or fallback)
model, elos = None, {}
try:
    maybe_model, maybe_elos = train_model_from_history(history_df)
    if maybe_model is not None:
        model = maybe_model
        elos = maybe_elos
        st.sidebar.success("Trained model available.")
    else:
        # fallback: compute elos only
        elos = maybe_elos or compute_simple_elo(history_df)
        st.sidebar.warning("Not enough labelled historical games — Elo fallback active.")
except Exception as e:
    elos = compute_simple_elo(history_df) if not history_df.empty else {}
    st.sidebar.error("Model training failed — using Elo fallback.")

# compute model record and show
correct, incorrect, pct = compute_model_record(history_df, model, elos)
if correct + incorrect > 0:
    st.sidebar.markdown(f"✅ Correct: **{correct}**  | ❌ Incorrect: **{incorrect}**  | 🎯 Hit%: **{pct:.1f}%**")
else:
    st.sidebar.markdown("No completed historical games available for record.")

# Main: build schedule for selected week via Covers
st.subheader(f"DJBets — Season {CURRENT_SEASON} — Week {current_week}")

with st.spinner("Loading schedule & odds from Covers..."):
    week_sched = prepare_week_schedule_from_covers(CURRENT_SEASON, int(current_week))

# If covers failed, try ESPN (minor fallback)
if week_sched.empty:
    st.warning("No schedule found from Covers for this week. Attempting ESPN scoreboard for schedule (fallback).")
    try:
        espn = safe_fetch_espn_scores(CURRENT_SEASON, int(current_week))
        if not espn.empty:
            # create schedule structure
            rows=[]
            for _, r in espn.iterrows():
                rows.append({
                    "season": CURRENT_SEASON,
                    "week": int(current_week),
                    "home_team": str(r.get("home_team","")).lower().strip(),
                    "away_team": str(r.get("away_team","")).lower().strip(),
                    "spread": np.nan,
                    "over_under": np.nan,
                    "home_score": r.get("home_score", np.nan),
                    "away_score": r.get("away_score", np.nan),
                    "status": r.get("status","scheduled")
                })
            if rows:
                week_sched = pd.DataFrame(rows)
    except Exception:
        week_sched = pd.DataFrame()

if week_sched.empty:
    st.error("No games found for this week (Covers and ESPN failed). Please ensure schedule.csv in data or upstream sources are reachable.")
    st.stop()

# Enrich with logos, predictions, and ESPN final scores (if any)
# attempt to fetch ESPN scores to mark completed games
espn_scores = safe_fetch_espn_scores(CURRENT_SEASON, int(current_week))

# join espn_scores to week_sched where possible to get scores/status
if not espn_scores.empty:
    # normalize keys lowercase
    espn_scores["home_team"] = espn_scores["home_team"].astype(str).map(lambda x: x.lower().strip())
    espn_scores["away_team"] = espn_scores["away_team"].astype(str).map(lambda x: x.lower().strip())
    def enrich_with_espn(row):
        h = row["home_team"]
        a = row["away_team"]
        match = espn_scores[(espn_scores["home_team"] == h) & (espn_scores["away_team"] == a)]
        if not match.empty:
            row["home_score"] = match.iloc[0].get("home_score", row.get("home_score", np.nan))
            row["away_score"] = match.iloc[0].get("away_score", row.get("away_score", np.nan))
            row["status"] = match.iloc[0].get("status", row.get("status", "scheduled"))
        return row
    week_sched = week_sched.apply(enrich_with_espn, axis=1)

# Prepare display
def round_or_na(x, fmt="{:.1f}"):
    try:
        if pd.isna(x):
            return "N/A"
        return fmt.format(float(x))
    except Exception:
        return "N/A"

cards = []
for idx, row in week_sched.iterrows():
    # logos
    home_logo = get_logo_path(row["home_team"])
    away_logo = get_logo_path(row["away_team"])
    # predicted probabilities
    prob = predict_game_prob(model, elos, row)
    prob_text = "N/A" if prob is None else f"{prob*100:.1f}%"
    # market probability from spread (approx): convert spread to implied market prob (basic)
    spread = row.get("spread", np.nan)
    market_prob = np.nan
    if pd.notna(spread):
        try:
            # crude conversion: translate spread to win prob using normal cdf approximation
            # P(home wins) ~ 1 - Phi((spread)/13)  ; 13 is typical points sd. This is heuristic.
            from math import erf, sqrt
            z = (spread) / 13.0
            # normal cdf
            market_prob = 0.5 * (1 + erf(-z / sqrt(2)))
        except Exception:
            market_prob = np.nan
    # blended:
    blended = None
    if prob is not None and not np.isnan(market_prob):
        blended = market_weight * market_prob + (1 - market_weight) * prob
    elif prob is not None:
        blended = prob
    elif not np.isnan(market_prob):
        blended = market_prob
    # edge in percentage points (pp)
    edge_pp = None
    if blended is not None and prob is not None:
        edge_pp = (prob - blended) * 100.0
    # recommendation
    rec = "🚫 No Bet"
    if edge_pp is not None and abs(edge_pp) >= bet_threshold:
        # recommend side: positive edge -> bet home; negative -> bet away
        if edge_pp > 0:
            rec = f"🛫 Bet Home (+{edge_pp:.1f}pp)"
        else:
            rec = f"🛫 Bet Away ({edge_pp:.1f}pp)"
    # final score display
    hs = row.get("home_score", np.nan)
    as_ = row.get("away_score", np.nan)
    completed = False
    final_score_text = ""
    if pd.notna(hs) and pd.notna(as_):
        completed = True
        final_score_text = f"{int(hs)} - {int(as_)}"
    else:
        final_score_text = "Not started"
    # correctness for completed games
    model_correct_text = ""
    if completed:
        actual_home = float(hs) > float(as_)
        pred_home = (prob is not None and prob >= 0.5)
        model_correct_text = "✅ Model Correct" if pred_home == actual_home else "❌ Model Wrong"
    # push card info
    cards.append({
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "home_logo": home_logo,
        "away_logo": away_logo,
        "spread": spread,
        "over_under": row.get("over_under", np.nan),
        "prob": prob,
        "prob_text": prob_text,
        "market_prob": market_prob,
        "blended": blended,
        "edge_pp": edge_pp,
        "rec": rec,
        "final_score_text": final_score_text,
        "completed": completed,
        "model_correct_text": model_correct_text,
        "status": row.get("status","scheduled")
    })

# Layout: show all cards expanded by default
cols_per_row = 2
for i in range(0, len(cards), cols_per_row):
    cols = st.columns(cols_per_row)
    for j, c in enumerate(cards[i:i+cols_per_row]):
        col = cols[j]
        with col:
            # Header: away @ home
            away_display = c["away_team"].replace("_", " ").title()
            home_display = c["home_team"].replace("_", " ").title()
            st.markdown(f"### {away_display}  @  **{home_display}**")
            # logos + score row
            logo_cols = st.columns([1,2,1])
            # away logo centered
            try:
                if c["away_logo"] is not None:
                    logo_cols[0].image(c["away_logo"], width=64)
                else:
                    logo_cols[0].write("")  # keep spacing
            except Exception:
                logo_cols[0].write("")
            # versus / score
            if c["completed"]:
                logo_cols[1].markdown(f"**Final:** {c['final_score_text']}  \n{c['model_correct_text']}")
            else:
                logo_cols[1].markdown(f"**Status:** {c['status'].title()}  \nHome Win Probability: **{c['prob_text']}**")
            # home logo
            try:
                if c["home_logo"] is not None:
                    logo_cols[2].image(c["home_logo"], width=64)
                else:
                    logo_cols[2].write("")
            except Exception:
                logo_cols[2].write("")
            # Details block
            with st.expander("Model & Market details", expanded=True):
                st.write(f"- **Model Probability:** {c['prob_text'] if c['prob'] is not None else 'N/A'}")
                st.write(f"- **Market (from spread) Prob:** {f'{c['market_prob']*100:.1f}%' if (c['market_prob'] is not None and not np.isnan(c['market_prob'])) else 'N/A'}")
                st.write(f"- **Blended Prob:** {f'{c['blended']*100:.1f}%' if (c['blended'] is not None and not np.isnan(c['blended'])) else 'N/A'}")
                st.write(f"- **Edge:** {f'+{c['edge_pp']:.1f} pp' if (c['edge_pp'] is not None and not np.isnan(c['edge_pp'])) else 'N/A'}")
                st.write(f"- **Vegas Spread:** {round_or_na(c['spread'])}")
                st.write(f"- **Over/Under:** {round_or_na(c['over_under'])}")
                st.write(f"- **Recommendation:** {c['rec']}")
            st.markdown("---")

# Footer / quick tips
st.caption("Notes: Covers is primary for schedule & odds. ESPN is used only to fetch final scores when available. Drop historical files into /data to override local archive.")