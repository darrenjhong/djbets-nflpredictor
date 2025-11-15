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
from team_logo_map import canonical_team_name
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

    # 1) Load full season schedule exactly as before
    full_sched = load_or_fetch_schedule(CURRENT_SEASON)

    if not full_sched.empty and "week" in full_sched.columns:
        week_options = sorted(full_sched["week"].astype(int).unique().tolist())
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

# 2) Base schedule for the week (no Covers involved)
with st.spinner(f"Preparing schedule for week {current_week}..."):
    base_week = prepare_week_schedule(
        CURRENT_SEASON,
        int(current_week),
        schedule_df=full_sched,
        covers_df=None,  # do NOT pass Covers here; schedule stays as before
    )

if base_week is None or base_week.empty:
    st.error(
        "Schedule is empty for this week. This indicates an issue with data_loader "
        "or your local schedule file, not with Covers."
    )
    st.stop()

# 3) Fetch Covers odds separately and overlay onto base_week
with st.spinner(f"Fetching Covers odds for Week {current_week}..."):
    covers_df = fetch_covers_for_week(CURRENT_SEASON, int(current_week))

week_sched = base_week.copy()

if covers_df is not None and not covers_df.empty:
    cov = covers_df.copy()

    # Canonicalize Covers home/away using the same logic as data_loader
    def canonical_display(x):
        try:
            # normalize some punctuation/spacing for Covers strings first
            n = str(x).lower().strip().replace(".", "")
            return canonical_team_name(n)
        except Exception:
            return str(x).lower().replace(" ", "_")

    cov["home_canon"] = cov["home"].astype(str).apply(canonical_display)
    cov["away_canon"] = cov["away"].astype(str).apply(canonical_display)

    week_sched["home_team_canon"] = week_sched["home_team"].astype(str)
    week_sched["away_team_canon"] = week_sched["away_team"].astype(str)

    merged = pd.merge(
        week_sched,
        cov[["home_canon", "away_canon", "spread", "over_under"]],
        left_on=["home_team_canon", "away_team_canon"],
        right_on=["home_canon", "away_canon"],
        how="left",
    )

    # Prefer Covers values where available
    merged["spread"] = merged["spread_y"].combine_first(merged.get("spread_x"))
    merged["over_under"] = merged["over_under_y"].combine_first(
        merged.get("over_under_x")
    )

    # Clean up helper columns
    keep_cols = [
        c
        for c in merged.columns
        if not c.endswith("_x")
        and not c.endswith("_y")
        and not c.endswith("_canon")
    ]
    week_sched = merged[keep_cols]

    # If still no odds at all, keep schedule but warn
    if week_sched["spread"].isna().all() and week_sched["over_under"].isna().all():
        st.info(
            "Covers odds fetched but could not be matched to schedule team names. "
            "Check canonical mappings in team_logo_map.py / canonical_team_name."
        )
else:
    st.info(
        "Covers odds not available for this week (empty or blocked). "
        "Spreads and totals will show as '—'."
    )

st.success(f"Loaded {len(week_sched)} games for Week {current_week}")

# Render each game row
for _, row in week_sched.iterrows():
    render_game_row(row)
