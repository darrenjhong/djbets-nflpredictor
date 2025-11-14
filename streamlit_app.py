<<<<<<< HEAD
﻿# streamlit_app.py
import os
import io
import time
from datetime import datetime
import traceback

=======
# streamlit_app.py
# DJBets — Streamlit NFL Predictor (full file)
# Replace your existing streamlit_app.py with this file.

import streamlit as st
>>>>>>> 83e4cd8c405192af3350849f65cd9e3058c42b44
import pandas as pd
import numpy as np
import traceback

<<<<<<< HEAD
from data_loader import (
    load_historical,
    load_local_schedule,
    fetch_espn_week,
    fetch_espn_season,
    load_or_fetch_schedule,
)
from covers_odds import fetch_covers_for_week
from team_logo_map import canonical_team_name
from model import train_or_load_model, predict_row, has_trained_model
from utils import (
    get_logo_path,
    compute_simple_elo,
    safe_request_json,
    compute_roi,
)

# --- Page config ---
st.set_page_config(page_title="DJBets — NFL Predictor", layout="wide")

# Constants
CURRENT_SEASON = datetime.now().year
MAX_WEEKS = 18

DATA_DIR = "data"
LOGOS_DIR = "public/logos"

# Ensure data dir exists
os.makedirs(DATA_DIR, exist_ok=True)


# ----------------------
# Load data (cached)
# ----------------------
@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_historical_cached(path=os.path.join(DATA_DIR, "nfl_archive_10Y.json")):
    return load_historical(path)


@st.cache_data(ttl=60 * 10, show_spinner=False)
def load_schedule_cached(season=CURRENT_SEASON):
    return load_or_fetch_schedule(season)


# ----------------------
# UI - Sidebar
# ----------------------
with st.sidebar:
    st.markdown("## 🏈 DJBets NFL Predictor")
    # week selector at top (dropdown)
    schedule_df = load_schedule_cached(CURRENT_SEASON)
    if not schedule_df.empty and "week" in schedule_df.columns:
        available_weeks = sorted(int(w) for w in schedule_df["week"].dropna().unique().tolist())
        week = st.selectbox("📅 Week", options=available_weeks, index=0)
    else:
        week = st.selectbox("📅 Week", options=list(range(1, MAX_WEEKS + 1)), index=0)

    st.markdown("### ⚙️ Model Controls")
    market_weight = st.slider("Market weight (blend model <> market)", 0.0, 1.0, 0.0, 0.05)
    bet_threshold = st.slider("Bet threshold (edge pts)", 0.0, 20.0, 8.0, 0.5)

    st.markdown("### 📊 Model Record")
    hist = load_historical_cached()
    model_trained = has_trained_model()
    if model_trained:
        st.success("Trained model available")
    else:
        st.info("No trained model available — Elo fallback active.")

    st.markdown("---")
    st.caption("Drop files into /data to override sources (schedule.csv, nfl_archive_10Y.json)")

# ----------------------
# Main
# ----------------------
st.header(f"DJBets — NFL Predictor — Season {CURRENT_SEASON} — Week {week}")

# load history & schedule
hist_df = load_historical_cached()
schedule_df = load_schedule_cached(CURRENT_SEASON)

# prepare week schedule: prefer ESPN schedule rows for the chosen week
def prepare_week_schedule(schedule_df, wk):
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame()

    df = schedule_df.copy()
    # Normalize columns
    for c in ["home_team", "away_team", "week", "season"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    # ensure numeric week
    if "week" in df.columns:
        try:
            df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int)
        except Exception:
            pass

    week_df = df[df["week"] == int(wk)].copy()
    if week_df.empty:
        return pd.DataFrame()

    # canonicalize team names for logo lookup and matching
    for col in ["home_team", "away_team"]:
        if col in week_df.columns:
            week_df[col] = week_df[col].astype(str).apply(lambda s: canonical_team_name(s.lower()))
    return week_df

week_sched = prepare_week_schedule(schedule_df, week)

# If no games for the week from ESPN/local, attempt to fetch via ESPN API for that week (one-shot)
if week_sched.empty:
    st.warning("No schedule entries for this week found in schedule sources. Attempting ESPN fetch for week...")
    try:
        espn_w = fetch_espn_week(CURRENT_SEASON, int(week))
        if not espn_w.empty:
            # canonicalize
            espn_w["home_team"] = espn_w["home_team"].apply(lambda s: canonical_team_name(str(s).lower()))
            espn_w["away_team"] = espn_w["away_team"].apply(lambda s: canonical_team_name(str(s).lower()))
            week_sched = espn_w
            st.success(f"Loaded {len(week_sched)} games from ESPN for week {week}")
        else:
            st.warning("ESPN returned no games for that week.")
    except Exception as e:
        st.error("ESPN fetch failed (network or API).")
        st.write(e)

# If still empty, try Covers to build simple matchups
if week_sched.empty:
    st.info("Trying Covers matchup scraping to build schedule...")
    try:
        covers_df = fetch_covers_for_week(CURRENT_SEASON, int(week))
        if not covers_df.empty:
            covers_df["home_team"] = covers_df["home"].apply(lambda s: canonical_team_name(str(s).lower()))
            covers_df["away_team"] = covers_df["away"].apply(lambda s: canonical_team_name(str(s).lower()))
            # create minimal week_sched
            week_sched = pd.DataFrame({
                "home_team": covers_df["home_team"],
                "away_team": covers_df["away_team"],
                "season": CURRENT_SEASON,
                "week": int(week),
                "spread": covers_df.get("spread"),
                "over_under": covers_df.get("over_under"),
            })
            st.success(f"Constructed schedule with {len(week_sched)} games from Covers")
        else:
            st.warning("Covers did not return matchups either.")
    except Exception as e:
        st.error("Covers scraping failed.")
        st.write(e)

# final guard
if week_sched.empty:
    st.error("No games found for this week. Ensure schedule.csv or data is present, or ESPN/Covers are reachable.")
    st.stop()

# ------------------------------------------------------------
# Enrich week dataframe: logos, spreads/OU from Covers if missing,
# compute simple ELO if history present, then predict
# ------------------------------------------------------------
# add logo paths
def ensure_logo_cols(df):
    df = df.copy()
    df["home_logo"] = df["home_team"].apply(lambda s: get_logo_path(s))
    df["away_logo"] = df["away_team"].apply(lambda s: get_logo_path(s))
    return df

week_sched = ensure_logo_cols(week_sched)

# fill spreads/over_under if missing via Covers (already attempted above)
if "spread" not in week_sched.columns or week_sched["spread"].isna().all():
    try:
        cov = fetch_covers_for_week(CURRENT_SEASON, int(week))
        if not cov.empty:
            # map cov rows into week_sched by team pairs
            for i, r in cov.iterrows():
                h = canonical_team_name(str(r["home"]).lower())
                a = canonical_team_name(str(r["away"]).lower())
                mask = (week_sched["home_team"] == h) & (week_sched["away_team"] == a)
                if mask.any():
                    idx = week_sched[mask].index[0]
                    week_sched.at[idx, "spread"] = r.get("spread")
                    week_sched.at[idx, "over_under"] = r.get("over_under")
    except Exception:
        pass

# compute Elo columns using history (simple team Elo from historical results)
if hist_df is not None and not hist_df.empty:
    try:
        elo_map = compute_simple_elo(hist_df)
        # attach elo_home / elo_away
        week_sched["elo_home"] = week_sched["home_team"].map(lambda t: elo_map.get(t, 1500))
        week_sched["elo_away"] = week_sched["away_team"].map(lambda t: elo_map.get(t, 1500))
        week_sched["elo_diff"] = week_sched["elo_home"] - week_sched["elo_away"]
    except Exception:
        week_sched["elo_home"] = 1500
        week_sched["elo_away"] = 1500
        week_sched["elo_diff"] = 0
else:
    week_sched["elo_home"] = 1500
    week_sched["elo_away"] = 1500
    week_sched["elo_diff"] = 0

# ensure numeric columns
for c in ["spread", "over_under"]:
    if c in week_sched.columns:
        week_sched[c] = pd.to_numeric(week_sched[c], errors="coerce")
    else:
        week_sched[c] = np.nan

# ----------------------
# Train or load model
# ----------------------
with st.spinner("Training/loading model..."):
    model, features = train_or_load_model(hist_df)

# ----------------------
# Predict per game
# ----------------------
pred_rows = []
for idx, row in week_sched.reset_index().iterrows():
    # prepare feature vector; use available features and fallbacks
    Xrow = {}
    # features we expect: elo_diff, spread, over_under
    Xrow["elo_diff"] = float(row.get("elo_diff", 0))
    Xrow["spread"] = float(row.get("spread")) if pd.notna(row.get("spread")) else np.nan
    Xrow["over_under"] = float(row.get("over_under")) if pd.notna(row.get("over_under")) else np.nan

    # predict
    try:
        prob, pred_home_pts, pred_away_pts = predict_row(model, Xrow)
    except Exception:
        prob = None
        pred_home_pts = None
        pred_away_pts = None

    # compute market prob (from spread -> implied prob) if spread available
    market_prob = None
    if pd.notna(row.get("spread")):
        # naive conversion: smaller spread -> closer to 50%; map spread to prob via logistic
        s = float(row["spread"])
        market_prob = 1 / (1 + np.exp(-(-s) / 3.0))  # heuristic

    # blended probability
    if prob is None:
        blended = market_prob
    elif market_prob is None:
        blended = prob
    else:
        blended = (1 - market_weight) * prob + market_weight * market_prob

    # edge: model - market (in percentage points)
    edge_pp = None
    if prob is not None and market_prob is not None:
        edge_pp = (prob - market_prob) * 100

    # recommendation
    rec = "🚫 No Bet"
    if edge_pp is not None and abs(edge_pp) >= bet_threshold:
        if edge_pp > 0:
            rec = "🛫 Bet Home (spread)"
        else:
            rec = "🛫 Bet Away (spread)"

    pred_rows.append(
        {
            "idx": idx,
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_logo": row["home_logo"],
            "away_logo": row["away_logo"],
            "prob_home": prob,
            "market_prob": market_prob,
            "blended": blended,
            "edge_pp": edge_pp,
            "recommendation": rec,
            "pred_home_pts": pred_home_pts,
            "pred_away_pts": pred_away_pts,
            "spread": row.get("spread"),
            "over_under": row.get("over_under"),
        }
    )

pred_df = pd.DataFrame(pred_rows)

# ----------------------
# Sidebar: show simple ROI/model record (light)
# ----------------------
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🎯 Model Snapshot")
    # compute quick model record/ROI if history exists and model trained
    try:
        if hist_df is not None and not hist_df.empty and model is not None:
            pnl, bets_made, roi = compute_roi(hist_df, model)
            st.metric("ROI", f"{roi:.2f}%")
            st.metric("Bets Made", f"{int(bets_made)}")
        else:
            st.write("Historical record unavailable")
    except Exception:
        st.write("Model performance unavailable")

# ----------------------
# Main UI: list games
# ----------------------
cols = st.columns(1)
for i, r in pred_df.iterrows():
    card = st.container()
    with card:
        st.markdown("---")
        c1, c2, c3 = st.columns([1, 5, 2])
        with c1:
            # away logo left, home logo right
            try:
                if r["away_logo"]:
                    st.image(r["away_logo"], width=56)
            except Exception:
                pass
        with c2:
            # matchup line: "away @ home" (home right)
            st.markdown(f"**{r['away_team'].replace('_',' ').title()}  @  {r['home_team'].replace('_',' ').title()}**")
            # probability
            prob_txt = "N/A"
            if r["prob_home"] is not None:
                prob_txt = f"{r['prob_home']*100:.1f}%"
            st.write(f"Home Win Probability: **{prob_txt}**")

            # predicted score if available
            if r["pred_home_pts"] is not None and r["pred_away_pts"] is not None:
                st.write(f"Predicted score: **{r['home_team'].split('_')[-1].title()} {r['pred_home_pts']:.1f} - {r['away_team'].split('_')[-1].title()} {r['pred_away_pts']:.1f}**")

            # spread / ou
            s_txt = "N/A" if pd.isna(r["spread"]) else f"{r['spread']:+.1f}"
            ou_txt = "N/A" if pd.isna(r["over_under"]) else f"{r['over_under']:.1f}"
            st.write(f"Spread (vegas): **{s_txt}** | O/U: **{ou_txt}**")

            # edge / recommendation
            edge_txt = "N/A" if r["edge_pp"] is None else f"{r['edge_pp']:+.1f} pp"
            st.write(f"Edge vs market: **{edge_txt}**")
            st.write(f"Recommendation: **{r['recommendation']}**")

        with c3:
            try:
                if r["home_logo"]:
                    st.image(r["home_logo"], width=64)
            except Exception:
                pass

# footer
st.caption(f"Data sources: ESPN (schedule) + Covers (odds). Local historical archive used for training. Last update {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
=======
from data_loader import load_or_fetch_schedule, load_historical
from covers_odds import fetch_covers_for_week
from team_logo_map import canonical_team_name
from team_logo_map import canonical_from_string
from utils import get_logo_path, compute_simple_elo, compute_roi
from model import train_model, predict
# ML model
from sklearn.linear_model import LogisticRegression
from datetime import datetime

# ---------- Config ----------
CURRENT_SEASON = datetime.now().year
MAX_WEEKS = 18
LOGOS_DIR = "public/logos"  # ensure this is where your canonical logos live
HIST_PATH = "data/nfl_archive_10Y.json"
LOCAL_SCHEDULE = "data/schedule.csv"

st.set_page_config(page_title="DJBets — NFL Predictor", layout="wide")

# ---------- Utility helpers ----------
def safe_df_concat(dfs):
    dfs = [d for d in dfs if isinstance(d, pd.DataFrame) and not d.empty]
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def canonical_name_for_display(s: str) -> str:
    if not s or pd.isna(s):
        return ""
    return canonical_from_string(s)

def try_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def elo_to_prob(elo_diff: float) -> float:
    """
    Convert Elo diff (home - away) to probability home wins.
    Use logistic-like conversion consistent with Elo expectation:
    P(home) = 1 / (1 + 10^(-diff/400))
    """
    try:
        return 1.0 / (1.0 + 10 ** (-(elo_diff) / 400.0))
    except Exception:
        return 0.5

def spread_to_market_prob(spread: float) -> float:
    """
    Convert a Vegas spread (positive means home by X) to a market win probability.
    This is heuristic: use logistic with scale ~4 points.
    If spread is None/NaN, return np.nan.
    """
    if spread is None or pd.isna(spread):
        return np.nan
    # Negative spread -> home underdog (market_prob < 0.5)
    # We'll convert such that +/-7 points ~ ~0.8/0.2, +/-3 points ~ ~0.6/0.4
    scale = 4.0
    return 1.0 / (1.0 + np.exp(-spread / scale))

# ---------- Load data ----------
st.markdown("# 🏈 DJBets — NFL Predictor")
st.caption("Local + ESPN schedule. Logos from public/logos/. Minimal, dark-ready layout.")

@st.cache_data(ttl=60*60)
def load_data(season: int):
    # historical dataframe
    hist = load_historical(HIST_PATH)
    # schedule: prefer ESPN but fallback to local CSV
    sched = load_or_fetch_schedule(season)
    # also load local schedule in case
    local_sched = load_local_schedule(LOCAL_SCHEDULE)
    return hist, sched, local_sched

hist_df, sched_df, local_sched_df = load_data(CURRENT_SEASON)

# ---------- Compute Elo (simple) ----------
with st.spinner("Computing Elo ratings from history..."):
    elos = compute_simple_elo(hist_df) if (hist_df is not None and not hist_df.empty) else {}

# ---------- Train a small model (elo_diff -> home win) ----------
@st.cache_data(ttl=60*60)
def train_model_from_history(history: pd.DataFrame):
    """
    Train a simple LogisticRegression using 'elo_diff' feature where possible.
    If not enough labeled rows, return None indicating fallback.
    """
    if history is None or history.empty:
        return None, "no_history"

    # Prepare training rows: require both scores to exist
    req_cols = {"home_team", "away_team", "home_score", "away_score"}
    if not req_cols.issubset(set(history.columns)):
        return None, "missing_cols"

    # Compute team-level end elos as proxies (quick and robust)
    team_elos = compute_simple_elo(history)
    if not team_elos:
        return None, "no_elos"

    rows = []
    for _, r in history.iterrows():
        h = r.get("home_team")
        a = r.get("away_team")
        if pd.isna(h) or pd.isna(a):
            continue
        eh = team_elos.get(h, 1500)
        ea = team_elos.get(a, 1500)
        elo_diff = eh - ea
        try:
            home_score = float(r.get("home_score"))
            away_score = float(r.get("away_score"))
        except Exception:
            continue
        label = 1 if home_score > away_score else 0 if away_score > home_score else 0.5
        # we only use decisive games (no ties) for training
        if label in (0, 1):
            rows.append({"elo_diff": elo_diff, "label": int(label)})
    if len(rows) < 30:
        return None, f"too_few_rows:{len(rows)}"
    df = pd.DataFrame(rows)
    X = df[["elo_diff"]].values
    y = df["label"].values
    model = LogisticRegression()
    model.fit(X, y)
    return model, f"trained:{len(rows)}"

model, model_status = train_model_from_history(hist_df)

# ---------- Sidebar ----------
# ======================================
#          SIDEBAR (RESTORED STYLE)
# ======================================
with st.sidebar:

    # --- HEADER ---
    st.markdown(
        """
        <h2 style='text-align:center; margin-bottom:5px;'>🏈 DJBets NFL</h2>
        <p style='text-align:center; font-size:13px; color:#bbb;'>Predictions • Edges • Odds</p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # --- WEEK SELECTOR DROPDOWN (RESTORED) ---
    available_weeks = sorted(schedule_df["week"].dropna().unique().tolist())
    if not available_weeks:
        available_weeks = list(range(1, 19))

    week = st.selectbox(
        "📅 Select Week",
        available_weeks,
        index=available_weeks.index(max(available_weeks)) if available_weeks else 0,
    )

    st.markdown("---")

    # --- MODEL CONTROLS (with icons restored) ---
    st.markdown("### 🎯 Model Controls")

    market_weight = st.slider(
        "Market weight (blend model <> market)",
        0.0, 1.0, 0.5, 0.01,
        help="0 = pure model, 1 = pure market vegas lines"
    )

    bet_threshold = st.slider(
        "Bet threshold (edge pts)",
        0.0, 15.0, 2.5, 0.1,
        help="Minimum projected edge required to recommend a bet"
    )

    st.markdown("---")

    # --- MODEL RECORD CARD ---
    st.markdown("### 📊 Model Record")

    if model_training_info.get("error"):
        st.markdown(
            f"""
            <div style='background:#330000; padding:10px; border-radius:8px; color:#ff7777;'>
                {model_training_info["error"]}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        if model_training_info.get("mode") == "elo":
            st.markdown(
                """
                <div style='background:#222; padding:10px; border-radius:8px;'>
                    Elo fallback model active — limited historical data.
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style='background:#111; padding:10px; border-radius:8px;'>
                    ROI: <b>{model_training_info['roi']}%</b>  
                    Record: <b>{model_training_info['wins']}-{model_training_info['losses']}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

# ---------- Prepare schedule for the chosen week ----------
def prepare_week_schedule(season: int, week_num: int) -> pd.DataFrame:
    # prefer sched_df (ESPN) then local_sched
    df = pd.DataFrame()
    if sched_df is not None and not sched_df.empty:
        df = sched_df[sched_df["week"] == week_num].copy()
    if df.empty and local_sched_df is not None and not local_sched_df.empty:
        df = local_sched_df[local_sched_df["week"] == week_num].copy()
    if df.empty:
        # try ESPN single-week fetch (defensive)
        try:
            one = fetch_espn_week(season, week_num)
            if one is not None and not one.empty:
                df = one
        except Exception:
            df = pd.DataFrame()
    # normalize columns
    if not df.empty:
        for c in ["home_team", "away_team"]:
            if c in df.columns:
                df[c] = df[c].astype(str).map(lambda x: canonical_name_for_display(x))
            else:
                df[c] = ""
        # ensure status/score cols exist
        for c in ["home_score", "away_score", "status", "kickoff_et", "spread", "over_under"]:
            if c not in df.columns:
                df[c] = np.nan
    return df

week_sched = prepare_week_schedule(CURRENT_SEASON, int(current_week))

# Fetch covers odds (best-effort); will be empty DataFrame if fails
st.markdown(f"## Week {current_week}")

week_sched = schedule[schedule["week"] == current_week]

# fetch covers odds
covers = fetch_covers_for_week(CURRENT_YEAR, current_week)

# merge
merged = week_sched.merge(
    covers,
    how="left",
    left_on=["home_team", "away_team"],
    right_on=["home", "away"]
)

for idx, row in merged.iterrows():
    with st.container(border=True):
        col1, col2, col3 = st.columns([1, 1, 2])

        # away
        col1.image(get_logo_path(row["away_team"]), width=70)
        col1.markdown(f"### {row['away_team'].replace('_',' ').title()}")

        col2.markdown("# @")

        # home
        col2.image(get_logo_path(row["home_team"]), width=70)
        col2.markdown(f"### {row['home_team'].replace('_',' ').title()}")

        # predictions
        pred = predict(model, row)
        st.markdown(f"**Home Win %:** {pred['prob']:.1f}%")
        st.markdown(f"**Vegas Spread:** {row.get('spread','N/A')}")
        st.markdown(f"**Vegas O/U:** {row.get('over_under','N/A')}")

        if pred["edge"] >= threshold:
            st.success(f"Recommended: {pred['bet_side']} (edge {pred['edge']:.1f})")
        else:
            st.info("No bet")

# ---------- Build display rows with model / market / logos ----------
display_rows = []
for idx, r in week_sched.iterrows():
    home = r.get("home_team", "") or ""
    away = r.get("away_team", "") or ""

    # canonical names (already canonicalized in loader)
    home_canon = canonical_name_for_display(home)
    away_canon = canonical_name_for_display(away)

    eh = elos.get(home_canon, elos.get(home, 1500))
    ea = elos.get(away_canon, elos.get(away, 1500))
    elo_diff = try_float(eh) - try_float(ea)

    # model probability
    if model is not None:
        try:
            prob_model = float(model.predict_proba(np.array([[elo_diff]]))[:, 1][0])
        except Exception:
            prob_model = elo_to_prob(elo_diff)
    else:
        prob_model = elo_to_prob(elo_diff)

    # try find market spread from covers_df (match by canonical names)
    spread_val = np.nan
    ou_val = np.nan
    if not covers_df.empty:
        # match row where canonical away/home equal
        match = covers_df[
            (covers_df["home_canon"] == home_canon) & (covers_df["away_canon"] == away_canon)
        ]
        if match.empty:
            # try reverse (sometimes swapped)
            match = covers_df[
                (covers_df["home_canon"] == away_canon) & (covers_df["away_canon"] == home_canon)
            ]
        if not match.empty:
            m0 = match.iloc[0]
            spread_val = try_float(m0.get("spread", np.nan))
            ou_val = try_float(m0.get("over_under", np.nan))

    # market implied prob from spread (heuristic)
    market_prob = np.nan
    if not (pd.isna(spread_val) or spread_val is None):
        market_prob = spread_to_market_prob(spread_val)

    # blended probability
    blended_prob = prob_model * (1.0 - market_weight) + (market_prob if not pd.isna(market_prob) else prob_model) * market_weight

    # edge in percentage points (model - market)
    edge_pp = np.nan
    if not pd.isna(market_prob):
        edge_pp = (prob_model - market_prob) * 100.0

    # Recommendation logic
    recommendation = "🚫 No Bet"
    if not pd.isna(edge_pp) and abs(edge_pp) >= bet_threshold:
        # recommend whichever side has positive edge
        if edge_pp > 0:
            recommendation = "🟢 Bet Home (spread)"
        else:
            recommendation = "🔴 Bet Away (spread)"

    # logos
    home_logo = get_logo_path(home_canon, LOGOS_DIR) or get_logo_path(home, LOGOS_DIR)
    away_logo = get_logo_path(away_canon, LOGOS_DIR) or get_logo_path(away, LOGOS_DIR)

    # status / kickoff
    status = r.get("status", "unknown")
    kickoff = r.get("kickoff_et", "")

    # final score display if available
    home_score = r.get("home_score", "")
    away_score = r.get("away_score", "")

    display_rows.append({
        "home": home,
        "away": away,
        "home_logo": home_logo,
        "away_logo": away_logo,
        "prob_model": prob_model,
        "market_prob": market_prob,
        "blended_prob": blended_prob,
        "edge_pp": edge_pp,
        "recommendation": recommendation,
        "spread": spread_val,
        "over_under": ou_val,
        "status": status,
        "kickoff": kickoff,
        "home_score": home_score,
        "away_score": away_score,
        "elo_diff": elo_diff
    })

# ---------- UI: show games ----------
# -----------------------------
# GAME CARDS — AUTO EXPANDED, TWO COLUMN
# -----------------------------
st.markdown(f"## Week {current_week} Games")

if week_sched.empty:
    st.warning(f"No games found for Week {current_week}.")
else:
    cols = st.columns(2)  # two‐column layout

    for i, (_, game) in enumerate(week_sched.iterrows()):
        col = cols[i % 2]  # alternate between left/right
        with col:

            # TEAM LOGOS + NAMES
            home_logo = lookup_logo(game["home_team"])
            away_logo = lookup_logo(game["away_team"])

            st.markdown("### " + 
                        f"{game['away_team']} @ {game['home_team']}")

            c1, c2, c3 = st.columns([1, 1, 1])

            with c1:
                st.image(away_logo, width=65)
                st.markdown(f"**{game['away_team']}**")

            with c3:
                st.image(home_logo, width=65)
                st.markdown(f"**{game['home_team']}**")

            st.markdown("---")

            # MODEL PREDICTIONS
            st.markdown(f"**Home Win Probability:** {game['home_win_prob']:.1f}%")
            st.markdown(f"**Predicted Winner:** {game['predicted_winner']}")

            # ODDS
            spread = game.get("spread")
            ou = game.get("over_under")

            st.markdown(f"**Spread (Vegas):** {spread if pd.notna(spread) else 'N/A'}")
            st.markdown(f"**O/U:** {ou if pd.notna(ou) else 'N/A'}")

            # EDGE + RECOMMENDATION
            edge = game.get("edge")
            st.markdown(f"**Edge:** {edge if pd.notna(edge) else 'N/A'}")

            rec = game.get("recommended_bet", None)
            if rec:
                st.success(f"**Recommended Bet:** {rec}")
            else:
                st.info("🚫 No Bet")
st.caption("Tip: put your canonical logos in `public/logos/` using lowercase underscore names (e.g. chicago_bears.png).")

# End of file
>>>>>>> 83e4cd8c405192af3350849f65cd9e3058c42b44
