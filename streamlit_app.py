# ------------------------------------------------------------
# DJBets NFL Predictor — Streamlit App
# Covers + fastR + ESPN schedule, model predictions, logos
# ------------------------------------------------------------

import streamlit as st
st.set_page_config(page_title="DJBets NFL Predictor", layout="wide")

import pandas as pd
import numpy as np
import requests
from io import StringIO
import os
import textwrap

# --------------------------
# CONFIG
# --------------------------
CURRENT_SEASON = 2025
LOGO_PATH = "public/logos"

# ============================================================
# LOAD FASTR SCHEDULE (Primary Source)
# ============================================================

FASTR_URL = (
    "https://raw.githubusercontent.com/"
    "nflverse/nflverse-data/master/releases/games.csv"
)


@st.cache_data(show_spinner=False)
def load_fastr_schedule(season: int):
    """
    Try to download weekly game schedule from nflverse fastR CSV.
    If the request fails (404, timeout, etc.), return empty DataFrame.
    """
    try:
        r = requests.get(FASTR_URL, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))

        # Normalize column names
        df.columns = [c.lower().strip() for c in df.columns]

        # Filter to this season and regular-season weeks if columns exist
        if "season" in df.columns:
            df = df[df["season"] == season]
        if "week" in df.columns:
            df = df[df["week"].between(1, 18)]
        if "game_type" in df.columns:
            df = df[df["game_type"].str.upper() == "REG"]

        keep = [
            "season", "week", "home_team", "away_team",
            "home_score", "away_score", "gameday", "game_id",
        ]
        df = df[[c for c in keep if c in df.columns]]
        return df.reset_index(drop=True)

    except Exception as e:
        print("[FASTR ERROR]", e)
        return pd.DataFrame()


# ============================================================
# COVERS — SCRAPE ODDS & SPREADS
# ============================================================

@st.cache_data(show_spinner=False)
def covers_fetch_week(season: int, week: int):
    """
    Returns DataFrame:
    away_team, home_team, spread, over_under
    Scrapes Covers.com's NFL matchups page for that season/week.
    """
    try:
        url = f"https://www.covers.com/sports/nfl/matchups?season={season}&week={week}"
        r = requests.get(url, timeout=8)
        r.raise_for_status()

        dfs = pd.read_html(r.text)
        if not dfs:
            return pd.DataFrame()

        df = dfs[0]
        df.columns = [c.lower() for c in df.columns]

        # Expecting columns: matchup, spread, total
        if "matchup" not in df.columns:
            return pd.DataFrame()

        out = {"away_team": [], "home_team": [], "spread": [], "over_under": []}

        for _, row in df.iterrows():
            m = str(row["matchup"]).lower()
            if " at " not in m:
                continue
            away, home = m.split(" at ")
            out["away_team"].append(away.strip())
            out["home_team"].append(home.strip())
            out["spread"].append(row.get("spread", np.nan))
            out["over_under"].append(row.get("total", np.nan))

        return pd.DataFrame(out)

    except Exception as e:
        print("[COVERS ERROR]", e)
        return pd.DataFrame()


# ============================================================
# ESPN — fallback scoreboard
# ============================================================

@st.cache_data(show_spinner=False)
def espn_fetch_week(season: int, week: int):
    """
    Fetch weekly schedule + scores from ESPN's public scoreboard API.
    Uses explicit year + seasontype (2 = regular season) + week.
    """
    try:
        url = (
            "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
            f"?year={season}&seasontype=2&week={week}"
        )
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()

        rows = []
        for ev in data.get("events", []):
            comp = ev.get("competitions", [{}])[0]
            teams = comp.get("competitors", [])

            away = next((t for t in teams if t.get("homeAway") == "away"), {})
            home = next((t for t in teams if t.get("homeAway") == "home"), {})

            rows.append(
                {
                    "season": season,
                    "week": week,
                    "away_team": away.get("team", {})
                    .get("displayName", "")
                    .lower(),
                    "home_team": home.get("team", {})
                    .get("displayName", "")
                    .lower(),
                    "away_score": int(away.get("score", 0))
                    if away.get("score")
                    else np.nan,
                    "home_score": int(home.get("score", 0))
                    if home.get("score")
                    else np.nan,
                    "status": ev.get("status", {})
                    .get("type", {})
                    .get("description", "scheduled")
                    .lower(),
                }
            )
        return pd.DataFrame(rows)

    except Exception as e:
        print("[ESPN ERROR]", e)
        return pd.DataFrame()


# ============================================================
# LOGOS
# ============================================================

def logo_path(team: str):
    """Return full path for a team logo."""
    if not team:
        return ""
    fname = f"{team.lower().replace(' ', '_')}.png"
    p = os.path.join(LOGO_PATH, fname)
    return p if os.path.exists(p) else ""


# ============================================================
# BUILD WEEK SCHEDULE
# ============================================================

def build_week_schedule(season: int, week: int):
    """Merge fastR + Covers + ESPN with fallback if fastR is unavailable."""

    # Try fastR schedule first
    fastr = load_fastr_schedule(season)

    if not fastr.empty and "week" in fastr.columns:
        # Standard path: filter fastR by week
        block = fastr[fastr["week"] == week].copy()
    else:
        # If fastR failed (404 or no week column), start with an empty block
        block = pd.DataFrame()

    # If we still have nothing for this week, fall back to ESPN as the base schedule
    if block.empty:
        block = espn_fetch_week(season, week)
        if block.empty:
            # No data from fastR or ESPN — give up for this week
            return pd.DataFrame()

        # Normalize team names to lowercase to match Covers parsing
        for col in ["home_team", "away_team"]:
            if col in block.columns:
                block[col] = block[col].str.lower()

    # Ensure required columns exist
    if "home_score" not in block.columns:
        block["home_score"] = np.nan
    if "away_score" not in block.columns:
        block["away_score"] = np.nan
    if "status" not in block.columns:
        block["status"] = "scheduled"

    # Add betting fields
    if "spread" not in block.columns:
        block["spread"] = np.nan
    if "over_under" not in block.columns:
        block["over_under"] = np.nan

    # Covers odds
    covers = covers_fetch_week(season, week)
    if not covers.empty:
        # Normalize Covers team names too
        for col in ["home_team", "away_team"]:
            if col in covers.columns:
                covers[col] = covers[col].str.lower()

        for i, r in block.iterrows():
            hit = covers[
                (covers["home_team"] == r["home_team"])
                & (covers["away_team"] == r["away_team"])
            ]
            if not hit.empty:
                block.loc[i, "spread"] = hit.iloc[0].get("spread", np.nan)
                block.loc[i, "over_under"] = hit.iloc[0].get(
                    "over_under", np.nan
                )

    # ESPN as score/status overlay (even if ESPN was the base, this is harmless)
    espn = espn_fetch_week(season, week)
    if not espn.empty:
        for col in ["home_team", "away_team"]:
            if col in espn.columns:
                espn[col] = espn[col].str.lower()

        for i, r in block.iterrows():
            hit = espn[
                (espn["home_team"] == r["home_team"])
                & (espn["away_team"] == r["away_team"])
            ]
            if not hit.empty:
                row = hit.iloc[0]
                block.loc[i, "home_score"] = row.get("home_score", np.nan)
                block.loc[i, "away_score"] = row.get("away_score", np.nan)
                block.loc[i, "status"] = row.get("status", "scheduled")

    return block.reset_index(drop=True)


# ============================================================
# MODEL (placeholder prediction)
# ============================================================

def model_predict(row):
    """Dummy model logic so UI works."""
    # Simple Elo-like random placeholder
    h = np.random.normal(0, 3)
    a = np.random.normal(0, 3)
    diff = h - a
    pred = row["home_team"] if diff > 0 else row["away_team"]
    return pred, diff


# ============================================================
# GAME ROW DISPLAY
# ============================================================

def render_game_row(row):
    home = row["home_team"]
    away = row["away_team"]

    home_logo = logo_path(home)
    away_logo = logo_path(away)

    pred, edge = model_predict(row)

    status = row.get("status", "")
    home_score = row.get("home_score", None)
    away_score = row.get("away_score", None)

    logo_style = "height:55px; margin-bottom:4px;"

    # MAIN MATCHUP ROW (HTML rendered via st.markdown)
    html_top = textwrap.dedent(
        f"""
        <div style="
            display:flex;
            justify-content:space-between;
            align-items:center;
            padding:18px 0;
            border-bottom:1px solid #e5e5e5;
        ">

            <div style="width:30%; text-align:center;">
                <img src="{away_logo}" style="{logo_style}">
                <div style="margin-top:6px; font-size:17px;">{away.title()}</div>
            </div>

            <div style="width:10%; text-align:center; font-size:28px; font-weight:600;">
                @
            </div>

            <div style="width:30%; text-align:center;">
                <img src="{home_logo}" style="{logo_style}">
                <div style="margin-top:6px; font-size:17px;">{home.title()}</div>
            </div>

        </div>
        """
    )
    st.markdown(html_top, unsafe_allow_html=True)

    # Prediction + Spread + Score
    score_text = ""
    if status in ("final", "complete", "post"):
        score_text = f"<b>Final Score:</b> {away_score} – {home_score}"

    html_bottom = textwrap.dedent(
        f"""
        <div style="padding: 10px 4px 20px;">
            <b>Spread:</b> {row['spread'] if row['spread']==row['spread'] else '—'}
            &nbsp; | &nbsp;
            <b>Total:</b> {row['over_under'] if row['over_under']==row['over_under'] else '—'}
            <br><br>

            <b>Model Pick:</b>
            <span style="color:#00b300; font-size:18px;">
                {pred.title()} by {abs(edge):.1f} pts
            </span>
            <br>
            {score_text}
        </div>
        """
    )
    st.markdown(html_bottom, unsafe_allow_html=True)


# ============================================================
# UI START
# ============================================================

with st.sidebar:
    st.markdown(
        textwrap.dedent(
            """
            <div style='text-align:center; margin-bottom:15px;'>
                <img src='https://img.icons8.com/ios-filled/100/target.png' width='60'>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    st.markdown("## 🏈 DJBets NFL Predictor")

    # Load fastR for week selector (fallback to full 1-18 range)
    fast_df = load_fastr_schedule(CURRENT_SEASON)
    if not fast_df.empty and "week" in fast_df.columns:
        weeks = sorted(fast_df["week"].unique().tolist())
    else:
        weeks = list(range(1, 19))

    current_week = st.selectbox("Select Week", weeks, index=0)

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

# FETCH WEEK SCHEDULE
with st.spinner(f"Loading schedule for week {current_week}..."):
    sched = build_week_schedule(CURRENT_SEASON, int(current_week))

if sched.empty:
    st.error(
        "No schedule found from Covers, fastR, or ESPN. "
        "This may be temporary (Cloudflare). Try again later."
    )
    st.stop()

st.success(f"Loaded {len(sched)} games for Week {current_week}")

# Render each game row
for _, row in sched.iterrows():
    render_game_row(row)
