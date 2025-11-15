# ------------------------------------------------------------
# DJBets NFL Predictor — Streamlit App
# ESPN schedule + odds, model predictions, logos
# ------------------------------------------------------------

import streamlit as st
st.set_page_config(page_title="DJBets NFL Predictor", layout="wide")

import pandas as pd
import numpy as np
import requests
from io import StringIO
import os

# --------------------------
# CONFIG
# --------------------------
CURRENT_SEASON = 2025
LOGO_PATH = "public/logos"

# ============================================================
# LOAD FASTR SCHEDULE (Primary Source, if available)
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
# (Optional) COVERS — OLD ODDS SCRAPER (kept but no longer required)
# ============================================================

@st.cache_data(show_spinner=False)
def covers_fetch_week(season: int, week: int):
    """
    Legacy odds scraper from Covers matchups page.

    Kept for reference; ESPN odds are now the primary source.
    If Covers changes its HTML or blocks scraping (Cloudflare),
    this may return an empty DataFrame.
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
# ESPN — schedule + scores + odds (spread & total)
# ============================================================

@st.cache_data(show_spinner=False)
def espn_fetch_week(season: int, week: int):
    """
    Fetch weekly schedule + scores + betting odds from ESPN's scoreboard API.

    Uses explicit year + seasontype (2 = regular season) + week.
    For odds, uses competitions[0]["odds"][0] with keys:
      - spread       (point spread relative to favorite)
      - overUnder    (game total points)
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

            # Default odds
            spread = np.nan
            over_under = np.nan

            odds_list = comp.get("odds", [])
            if odds_list:
                odds = odds_list[0]  # take first provider
                spread = odds.get("spread", np.nan)
                over_under = odds.get("overUnder", np.nan)

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
                    "spread": spread,
                    "over_under": over_under,
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
    """
    Merge fastR + ESPN (and optionally Covers) into one weekly schedule.

    Priority:
      1. fastR for base schedule if available.
      2. ESPN for base schedule if fastR empty.
      3. ESPN odds (spread, over/under) and scores always overlaid.
      4. Covers odds can optionally override ESPN if desired.
    """

    # Try fastR schedule first
    fastr = load_fastr_schedule(season)

    if not fastr.empty and "week" in fastr.columns:
        # Standard path: filter fastR by week
        block = fastr[fastr["week"] == week].copy()
    else:
        block = pd.DataFrame()

    # ESPN data for this week (used for base or overlay)
    espn = espn_fetch_week(season, week)
    if not espn.empty:
        for col in ["home_team", "away_team"]:
            if col in espn.columns:
                espn[col] = espn[col].str.lower()

    # If we still have nothing for this week, use ESPN as base schedule
    if block.empty and not espn.empty:
        block = espn.copy()

    if block.empty:
        # No schedule from fastR or ESPN
        return pd.DataFrame()

    # Normalize base schedule team names
    for col in ["home_team", "away_team"]:
        if col in block.columns:
            block[col] = block[col].str.lower()

    # Ensure required columns exist
    for col in ["home_score", "away_score", "status", "spread", "over_under"]:
        if col not in block.columns:
            if col in ["home_score", "away_score", "spread", "over_under"]:
                block[col] = np.nan
            else:
                block[col] = "scheduled"

    # Overlay ESPN scores + odds onto base
    if not espn.empty:
        for i, r in block.iterrows():
            hit = espn[
                (espn["home_team"] == r["home_team"])
                & (espn["away_team"] == r["away_team"])
            ]
            if not hit.empty:
                row_e = hit.iloc[0]
                # Scores / status
                block.loc[i, "home_score"] = row_e.get("home_score", np.nan)
                block.loc[i, "away_score"] = row_e.get("away_score", np.nan)
                block.loc[i, "status"] = row_e.get("status", "scheduled")
                # Odds
                if "spread" in row_e and not pd.isna(row_e["spread"]):
                    block.loc[i, "spread"] = row_e["spread"]
                if "over_under" in row_e and not pd.isna(row_e["over_under"]):
                    block.loc[i, "over_under"] = row_e["over_under"]

    # (Optional) Covers override — if you want Covers odds instead of ESPN, uncomment
    # covers = covers_fetch_week(season, week)
    # if not covers.empty:
    #     for col in ["home_team", "away_team"]:
    #         if col in covers.columns:
    #             covers[col] = covers[col].str.lower()
    #     for i, r in block.iterrows():
    #         hit = covers[
    #             (covers["home_team"] == r["home_team"])
    #             & (covers["away_team"] == r["away_team"])
    #         ]
    #         if not hit.empty:
    #             row_c = hit.iloc[0]
    #             block.loc[i, "spread"] = row_c.get("spread", np.nan)
    #             block.loc[i, "over_under"] = row_c.get("over_under", np.nan)

    return block.reset_index(drop=True)


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

    home_logo = logo_path(home)
    away_logo = logo_path(away)

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
                f"<div style='text-align:center;'>{away.title()}</div>",
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
                f"<div style='text-align:center;'>{home.title()}</div>",
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
    if status in ("final", "complete", "post"):
        score_line = f"**Final Score:** {int(away_score)} – {int(home_score)}"

    st.markdown(
        f"""
**Spread:** {spread_val} | **Total:** {total_val}

**Model Pick:** {pred.title()} by {abs(edge):.1f} pts  
{score_line}
        """
    )

    st.markdown("---")


# ============================================================
# UI START
# ============================================================

with st.sidebar:
    # Target logo removed; sidebar starts directly with title and controls
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
        "No schedule found from fastR or ESPN. "
        "This may be temporary. Try again later."
    )
    st.stop()

st.success(f"Loaded {len(sched)} games for Week {current_week}")

# Render each game row
for _, row in sched.iterrows():
    render_game_row(row)
