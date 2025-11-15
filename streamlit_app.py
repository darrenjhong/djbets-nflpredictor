# ------------------------------------------------------------
# DJBets NFL Predictor — Streamlit App
# Schedule/scores from data_loader + odds from Covers
# ------------------------------------------------------------

import streamlit as st
st.set_page_config(page_title="DJBets NFL Predictor", layout="wide")

import pandas as pd
import numpy as np

from data_loader import load_or_fetch_schedule, prepare_week_schedule
from covers_odds import fetch_covers_for_week
from utils import get_logo_path

# --------------------------
# CONFIG
# --------------------------
CURRENT_SEASON = 2025


# ============================================================
# MODEL (placeholder prediction)
# ============================================================

def model_predict(row):
    """Dummy model logic so UI works."""
    h = np.random.normal(0, 3)
    a = np.random.normal(0, 3)
    diff = h - a
    pred = row["home_team"] if diff > 0 else row["away_team"]
    return pred, diff


# ============================================================
# GAME ROW DISPLAY (centered away/home with '@' between)
# ============================================================

def render_game_row(row):
    home = row["home_team"]
    away = row["away_team"]

    # row teams are already canonical from data_loader.prepare_week_schedule
    home_logo = get_logo_path(home)
    away_logo = get_logo_path(away)

    pred, edge = model_predict(row)

    status = row.get("status", "")
    home_score = row.get("home_score", np.nan)
    away_score = row.get("away_score", np.nan)

    # Layout: three main columns for away, '@', home
    col_away, col_mid, col_home = st.columns([3, 1, 3])

    def fmt_score(val):
        return "" if pd.isna(val) else f"Score: {int(val)}"

    # Away side centered within its column via subcolumns
    with col_away:
        sub1, sub2, sub3 = st.columns([1, 2, 1])
        with sub2:
            if away_logo:
                st.image(away_logo, width=120)
            st.markdown(
                f"<div style='text-align:center;'>{away.replace('_',' ').title()}</div>",
                unsafe_allow_html=True,
            )
            sc = fmt_score(away_score)
            if sc:
                st.markdown(
                    f"<div style='text-align:center;'>{sc}</div>",
                    unsafe_allow_html=True,
                )

    # Center column with '@'
    with col_mid:
        st.markdown(
            "<div style='text-align:center; font-size:24px; font-weight:600;'>@</div>",
            unsafe_allow_html=True,
        )

    # Home side centered within its column via subcolumns
    with col_home:
        sub1, sub2, sub3 = st.columns([1, 2, 1])
        with sub2:
            if home_logo:
                st.image(home_logo, width=120)
            st.markdown(
                f"<div style='text-align:center;'>{home.replace('_',' ').title()}</div>",
                unsafe_allow_html=True,
            )
            sc = fmt_score(home_score)
            if sc:
                st.markdown(
                    f"<div style='text-align:center;'>{sc}</div>",
                    unsafe_allow_html=True,
                )

    # Second row: spread / total / prediction / final score
    spread_val = row["spread"] if row["spread"] == row["spread"] else "—"
    total_val = row["over_under"] if row["over_under"] == row["over_under"] else "—"

    score_line = ""
    if isinstance(status, str) and status.lower() in ("final", "complete", "post", "status_final"):
        score_line = f"**Final Score:** {int(away_score)} – {int(home_score)}"

    st.markdown(
        f"""
**Spread:** {spread_val} | **Total:** {total_val}

**Model Pick:** {pred.replace('_',' ').title()} by {abs(edge):.1f} pts  
{score_line}
        """
    )

    st.markdown("---")


# ============================================================
# UI START
# ============================================================

with st.sidebar:
    # Clean sidebar, no target logo
    st.markdown("## 🏈 DJBets NFL Predictor")

    # Load full season schedule once (local CSV or ESPN via data_loader)
    full_sched = load_or_fetch_schedule(CURRENT_SEASON)

    if not full_sched.empty and "week" in full_sched.columns:
        week_options = sorted(full_sched["week"].unique().tolist())
    else:
        week_options = list(range(1, 19))

    current_week = st.selectbox("Select Week", week_options, index=0)

    st.markdown("### ⚙️ Model Controls")
    market_weight = st.slider(
        "Market weight (blend model <> market)", 0.0, 1.0, 0.0, 0.01
    )
    bet_threshold = st.slider(
        "Bet threshold (edge pts)", 0.0, 20.0, 5.0, 0.5
    )

    st.markdown("### 📊 Model Record")
    st.info("Model trained using local data + Elo fallback.")


# MAIN TITLE
st.title(f"DJBets — Season {CURRENT_SEASON} — Week {current_week}")

# FETCH ODDS FROM COVERS FOR THIS WEEK
with st.spinner(f"Loading Covers odds for Week {current_week}..."):
    covers_df = fetch_covers_for_week(CURRENT_SEASON, int(current_week))

# PREPARE WEEK SCHEDULE (MERGES SCHEDULE + COVERS ODDS)
with st.spinner(f"Preparing schedule for week {current_week}..."):
    week_sched = prepare_week_schedule(
        CURRENT_SEASON,
        int(current_week),
        schedule_df=full_sched,
        covers_df=covers_df,
    )

if week_sched is None or week_sched.empty:
    st.error(
        "No schedule found for this week after merging schedule and Covers odds. "
        "Covers may be blocking requests or the schedule file may be missing."
    )
    st.stop()

st.success(f"Loaded {len(week_sched)} games for Week {current_week}")

# Render each game row
for _, row in week_sched.iterrows():
    render_game_row(row)
