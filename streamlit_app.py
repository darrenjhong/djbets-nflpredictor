# streamlit_app.py
# DJBets — Streamlit NFL Predictor (full file)
# Replace your existing streamlit_app.py with this file.

import streamlit as st
import pandas as pd
import numpy as np
import traceback

from data_loader import load_or_fetch_schedule, load_historical
from covers_odds import fetch_covers_for_week
from team_logo_map import canonical_team_name
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
# -----------------------------
# DJBETS SIDEBAR — FINAL LOCKED VERSION YOU REQUESTED
# -----------------------------
with st.sidebar:

    # Title with bullseye icon
    st.markdown("## 🎯 DJBets NFL Predictor")

    # --- WEEK SELECTOR (dropdown) ---
    st.markdown("### 📅 Select Week")

    current_week = st.selectbox(
        "Week",
        list(range(1, 19)),
        index=(current_week - 1) if "current_week" in globals() else 0
    )

    st.markdown("---")

    # --- MODEL CONTROLS ---
    st.markdown("### ⚙️ Model Controls")

    market_weight = st.slider(
        "Market weight (blend model <> market)",
        0.0, 1.0,
        value=market_weight if "market_weight" in globals() else 0.5,
        step=0.05
    )

    bet_threshold = st.slider(
        "Bet threshold (edge pts)",
        0.0, 15.0,
        value=bet_threshold if "bet_threshold" in globals() else 4.0,
        step=0.5,
    )

    st.markdown("---")

    # --- MODEL RECORD ---
    st.markdown("### 📊 Model Record")

    if "model_record" in globals() and model_record:
        roi     = model_record.get("roi", 0.0)
        wins    = model_record.get("wins", 0)
        losses  = model_record.get("losses", 0)
        pushes  = model_record.get("pushes", 0)

        st.markdown(
            f"""
            **ROI:** {'🟢' if roi >= 0 else '🔴'} {roi:.1f}%  
            **Record:** {wins}-{losses}-{pushes}
            """
        )
    else:
        st.info("No trained model available — Elo fallback active.")

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

week_sched = prepare_week_schedule(CURRENT_SEASON, int(week))

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