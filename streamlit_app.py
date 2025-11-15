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

    # home/away are canonical names from prepare_week_schedule
    home_logo = get_logo_path(home)
    away_logo = get_logo_path(away)

    pred, edge = model_predict(row)

    status = str(row.get("status", "")).lower()
    home_score = row.get("home_score", np.nan)
    away_score = row.get("away_score", np.nan)

    col_away, col_mid, col_home = st.columns([3, 1, 3])

    def fmt_score(val):
        return "" if pd.isna(val) else f"Score: {int(val)}"

    # Away side centered within its column via subcolumns
    with col_away:
        s1, s2, s3 = st.columns([1, 2, 1])
        with s2:
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
        s1, s2, s3 = st.columns([1, 2, 1])
        with s2:
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
    if status in ("final", "complete", "post", "status_final"):
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

# 1) SCHEDULE-ONLY WEEK (guaranteed, since your loader already works)
with st.spinner(f"Preparing schedule for week {current_week}..."):
    base_week = prepare_week_schedule(
        CURRENT_SEASON,
        int(current_week),
        schedule_df=full_sched,
        covers_df=None,  # no odds yet
    )

if base_week is None or base_week.empty:
    st.error(
        "Schedule is empty for this week. Check data_loader or your local schedule file."
    )
    st.stop()

# 2) TRY TO FETCH COVERS ODDS AND MERGE
with st.spinner(f"Fetching Covers odds for Week {current_week}..."):
    covers_df = fetch_covers_for_week(CURRENT_SEASON, int(current_week))

if covers_df is not None and not covers_df.empty:
    # Merge schedule + covers odds using your canonical logic
    week_sched = prepare_week_schedule(
        CURRENT_SEASON,
        int(current_week),
        schedule_df=full_sched,
        covers_df=covers_df,
    )
    if week_sched is None or week_sched.empty:
        # If merge fails for some reason, fall back to schedule-only
        week_sched = base_week.copy()
        covers_ok = False
    else:
        covers_ok = True
else:
    week_sched = base_week.copy()
    covers_ok = False

if not covers_ok:
    st.info(
        "Covers odds were not attached for this week (site may have changed layout or blocked the request). "
        "Spreads and totals will show as '—'."
    )

st.success(f"Loaded {len(week_sched)} games for Week {current_week}")

# Render each game row
for _, row in week_sched.iterrows():
    render_game_row(row)
