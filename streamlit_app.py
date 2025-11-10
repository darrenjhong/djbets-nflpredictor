# DJBets NFL Predictor v11.1
# Adds: explanations for sliders, ROI fix, model performance summary

import os
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
import streamlit as st
from datetime import datetime
from market_baseline import spread_to_home_prob, blend_probs

# --------------------------------------------------------------
# ⚙️ Configuration
# --------------------------------------------------------------
st.set_page_config(page_title="DJBets NFL Predictor", page_icon="🏈", layout="wide")

DATA_DIR = "data"
MODEL_FILE = os.path.join(DATA_DIR, "model.json")
SCHEDULE_FILE = os.path.join(DATA_DIR, "schedule.csv")
os.makedirs(DATA_DIR, exist_ok=True)
MAX_WEEKS = 18
MODEL_FEATURES = ["elo_diff", "temp_c", "wind_kph", "precip_prob"]

# --------------------------------------------------------------
# 🧮 Simulated Feature Generator
# --------------------------------------------------------------
def simulate_features(df: pd.DataFrame, week: int):
    """
    Simulate missing numeric model features for demonstration.
    When no DB data (ELO, injuries, weather) exist, this fills random values.
    """
    np.random.seed(week)  # week-based reproducibility

    # Simulate ELO difference — home minus away
    df["elo_home"] = np.random.normal(1550, 100, len(df))
    df["elo_away"] = np.random.normal(1500, 100, len(df))
    df["elo_diff"] = df["elo_home"] - df["elo_away"]

    # Simulate injuries — positive favors home, negative hurts them
    df["inj_diff"] = np.random.normal(0, 5, len(df))

    # Simulate weather (°C, km/h, precip %)
    df["temp_c"] = np.random.normal(10, 8, len(df))
    df["wind_kph"] = np.random.uniform(0, 30, len(df))
    df["precip_prob"] = np.random.uniform(0, 1, len(df))

    # Model uses those fields for prediction later
    return df



# --------------------------------------------------------------
# 🧠 Model Loader
# --------------------------------------------------------------
@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_FILE):
        model = xgb.XGBClassifier()
        model.load_model(MODEL_FILE)
        return model

    np.random.seed(42)
    df = pd.DataFrame({
        "elo_diff": np.random.normal(0, 100, 600),
        "temp_c": np.random.uniform(-5, 25, 600),
        "wind_kph": np.random.uniform(0, 25, 600),
        "precip_prob": np.random.uniform(0, 1, 600),
    })
    logits = 0.015*df["elo_diff"] - 0.04*(df["precip_prob"] - 0.4) - 0.02*(df["wind_kph"] - 10) + 0.01*(df["temp_c"] - 10)
    p = 1 / (1 + np.exp(-logits))
    y = (np.random.uniform(0, 1, 600) < p).astype(int)

    model = xgb.XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.08)
    model.fit(df[MODEL_FEATURES].values, y.values)
    model.save_model(MODEL_FILE)
    return model

# --------------------------------------------------------------
# 🏈 ESPN Schedule Scraper
# --------------------------------------------------------------
@st.cache_data(ttl=604800)
def fetch_schedule(season: int):
    """Fetch the NFL schedule + betting data from ESPN."""
    games = []
    for week in range(1, MAX_WEEKS + 1):
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?year={season}&seasontype=2&week={week}"
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception:
            continue

        for ev in data.get("events", []):
            comp = (ev.get("competitions") or [{}])[0]
            if not comp.get("competitors"):
                continue

            home, away, home_logo, away_logo = "TBD", "TBD", "", ""
            home_score, away_score = np.nan, np.nan
            state = ev.get("status", {}).get("type", {}).get("state", "pre")
            short_detail = ev.get("status", {}).get("type", {}).get("shortDetail", "")

            for team in comp["competitors"]:
                t = team.get("team", {})
                abbr = t.get("abbreviation", "")
                logo = t.get("logo") or (t.get("logos", [{}])[0].get("href", ""))
                score = team.get("score")
                if team.get("homeAway") == "home":
                    home, home_logo, home_score = abbr, logo, score
                else:
                    away, away_logo, away_score = abbr, logo, score

            # Defensive odds extraction
            odds_data = (comp.get("odds") or [{}])[0]
            spread = odds_data.get("details", "").strip() or "Even"
            over_under = odds_data.get("overUnder", np.nan)
            kickoff = comp.get("date", None)

            # Clean up spread text
            if spread in ("", "N/A", "NA", None):
                spread = "Even"
            if "Pick" in spread or "even" in spread.lower():
                spread = "Even"

            games.append({
                "season": season,
                "week": week,
                "home_team": home,
                "away_team": away,
                "home_logo": home_logo,
                "away_logo": away_logo,
                "kickoff_et": kickoff,
                "spread": spread,
                "over_under": over_under,
                "home_score": pd.to_numeric(home_score, errors="coerce"),
                "away_score": pd.to_numeric(away_score, errors="coerce"),
                "state": state,
                "status_text": short_detail,
            })

    df = pd.DataFrame(games)
    df.to_csv(SCHEDULE_FILE, index=False)
    return df


# --------------------------------------------------------------
# 🎛️ Sidebar Controls
# --------------------------------------------------------------
st.sidebar.header("🏈 DJBets NFL Predictor")

season = st.sidebar.selectbox("Season", [2026, 2025, 2024], index=1)
week = st.sidebar.selectbox("Week", list(range(1, MAX_WEEKS + 1)), index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Settings")

ALPHA = st.sidebar.slider("Market Weight (α)", 0.0, 1.0, 0.6, 0.05,
                          help="Controls how much weight to give to **market (Vegas)** probabilities vs the model. "
                               "α = 1 means fully trust the market; α = 0 means trust only the model.")
edge_thresh = st.sidebar.slider("Bet Threshold (pp)", 0.0, 10.0, 3.0, 0.5,
                                help="Minimum edge (percentage points) vs market required to trigger a recommended bet. "
                                     "Higher = fewer but stronger bets; lower = more frequent but riskier bets.")

if st.sidebar.button("♻️ Refresh Data"):
    fetch_schedule.clear()
    st.rerun()

# --------------------------------------------------------------
# 💰 ROI / Profit & Loss Calculation
# --------------------------------------------------------------
import numpy as np

def compute_roi(df: pd.DataFrame, bet_unit: float = 100.0):
    """
    Simulate a basic betting ROI strategy.
    For each completed game:
      - Bet on the team with the model's recommendation ("Bet Home" or "Bet Away")
      - Uses American odds derived from market probability (approximation)
      - Calculates total PnL and ROI
    """

    pnl = 0.0
    bets = 0

    for _, row in df.iterrows():
        if row.get("state") != "post":  # only include completed games
            continue

        # skip invalid or missing edges
        edge = row.get("edge_pp")
        if edge is None or np.isnan(edge):
            continue

        # require meaningful advantage to place a "bet"
        if abs(edge) < 3:
            continue

        rec = row.get("recommendation", "🚫 No Bet")
        model_home_win = row.get("home_win_prob_model", 0.5)

        # approximate fair American odds
        def implied_odds(p):
            if p <= 0 or np.isnan(p): 
                return 0
            if p > 0.5:
                return -round(100 * p / (1 - p))
            else:
                return round(100 * (1 - p) / p)

        # approximate market odds based on market probability
        market_prob = row.get("market_prob_home", np.nan)
        odds = implied_odds(market_prob if not np.isnan(market_prob) else model_home_win)

        # check winner
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            continue
        home_won = row["home_score"] > row["away_score"]

        # determine bet and outcome
        if "Home" in rec:
            bets += 1
            pnl += bet_unit * (100 / abs(odds)) if home_won else -bet_unit
        elif "Away" in rec:
            bets += 1
            pnl += bet_unit * (100 / abs(odds)) if not home_won else -bet_unit

    roi = (pnl / (bets * bet_unit)) * 100 if bets > 0 else 0.0
    return round(pnl, 2), bets, round(roi, 2)



# --------------------------------------------------------------
# 📊 Load Data
# --------------------------------------------------------------
model = load_or_train_model()
sched = fetch_schedule(season)
sched["kickoff_et"] = pd.to_datetime(sched["kickoff_et"], errors="coerce")

week_df = sched.query("week == @week").copy()
if week_df.empty:
    st.warning("No games found for this week.")
    st.stop()

week_df = simulate_features(week_df, week)
X = week_df[MODEL_FEATURES].astype(float)
week_df["home_win_prob_model"] = model.predict_proba(X)[:, 1]
week_df["market_prob_home"] = week_df["spread"].apply(spread_to_home_prob)

# Fill missing market probs with model probs (so they aren't NaN)
week_df["market_prob_home"] = week_df["market_prob_home"].fillna(week_df["home_win_prob_model"])

week_df["blended_prob_home"] = [
    blend_probs(m, mk, ALPHA)
    for m, mk in zip(week_df["home_win_prob_model"], week_df["market_prob_home"])
]

week_df["edge_pp"] = (week_df["blended_prob_home"] - week_df["market_prob_home"]) * 100

# Safely merge into full schedule for ROI computation
if "edge_pp" not in sched.columns and "edge_pp" in week_df.columns:
    sched = sched.merge(week_df[["season", "week", "home_team", "away_team", "edge_pp"]],
                        on=["season", "week", "home_team", "away_team"], how="left")

pnl, bets, roi = compute_roi(sched)

# --------------------------------------------------------------
# 🧾 Sidebar Metrics
# --------------------------------------------------------------
st.sidebar.markdown("### 📈 Performance")
st.sidebar.metric("💵 ROI", f"{roi*100:.1f}%", f"{pnl:+.2f} units")
st.sidebar.metric("🎯 Bets Made", f"{bets}")
st.sidebar.caption("ROI simulated using -110 odds (win pays +0.91 units)")

st.sidebar.markdown("---")
st.sidebar.markdown("🟩 = Home favored 🟥 = Away favored 🟨 = Even match")
st.sidebar.caption("Bars represent **blended home win probability**")

# --------------------------------------------------------------
# 🎯 Main Display
# --------------------------------------------------------------
st.title(f"🏈 DJBets NFL Predictor — Week {week} ({season})")

def safe_prob(val):
    """Ensure probability is a float between 0 and 1."""
    try:
        v = float(val)
        if np.isnan(v):
            return 0.5   # fallback neutral
        return min(max(v, 0.0), 1.0)
    except Exception:
        return 0.5


for _, row in week_df.iterrows():
    prob = row["blended_prob_home"]
    color = "🟩 Home Favored" if prob > 0.55 else ("🟥 Away Favored" if prob < 0.45 else "🟨 Even")
    edge = row["edge_pp"]
    edge_txt = f"Edge: {edge:+.2f} pp" if not np.isnan(edge) else "No market edge"

    state = row.get("state", "pre")
    status = {"pre": "⏳ Upcoming", "in": "🟢 Live", "post": "🔵 Final"}.get(state, "⚪ Unknown")

    st.markdown(f"### {row['away_team']} @ {row['home_team']} ({status})")
    st.markdown(f"**{color}** | {edge_txt}")

    kickoff = row["kickoff_et"].strftime("%a %b %d, %I:%M %p") if pd.notna(row["kickoff_et"]) else "TBD"
    st.caption(f"Kickoff: {kickoff}")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.image(row["away_logo"], width=60)
        st.markdown(f"**{row['away_team']}**")

    with col3:
        st.image(row["home_logo"], width=60)
        st.markdown(f"**{row['home_team']}**")

    with col2:
        prob_clamped = safe_prob(prob)
        st.progress(prob_clamped, text=f"Home Win Probability: {prob_clamped*100:.1f}%")

        st.markdown(f"Spread: {row['spread']} | O/U: {row['over_under']}")

        if state == "post" and not np.isnan(row["home_score"]) and not np.isnan(row["away_score"]):
            home_win = row["home_score"] > row["away_score"]
            pred_win = prob >= 0.5
            result = "✅ Correct" if home_win == pred_win else "❌ Missed"
            st.markdown(f"**Final:** {row['away_score']} - {row['home_score']} ({result})")
        elif state == "in":
            st.markdown(f"**Live:** {row['status_text']}")
        else:
            st.markdown("⏳ Not started yet")

        with st.expander("📊 Betting Breakdown"):
            st.markdown(f"**Model Probability:** {row['home_win_prob_model']*100:.1f}%")
            st.markdown(f"**Market Probability:** {row['market_prob_home']*100:.1f}%")
            st.markdown(f"**Blended Probability:** {row['blended_prob_home']*100:.1f}%")
            st.markdown(f"**Edge:** {edge:+.2f} pp")
            st.markdown(
                "**Recommendation:** " +
                ("🏠 Bet Home" if edge > edge_thresh else "🛫 Bet Away" if edge < -edge_thresh else "🚫 No Bet")
            )

st.markdown("---")
st.caption("🏈 DJBets NFL Predictor v11.1 — Market blending, ROI tracking, and smart bet thresholding.")
