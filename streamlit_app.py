# DJBets NFL Predictor v10.9
# Fixes UnhashableParamError by avoiding caching on unhashable model objects.
# Adds minor record calculation improvements and stable sidebar.

import os
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta
import hashlib

# --------------------------------------------------------------
# ⚙️ Config
# --------------------------------------------------------------
st.set_page_config(page_title="DJBets NFL Predictor", page_icon="🏈", layout="wide")
DATA_DIR = "data"
MODEL_FILE = os.path.join(DATA_DIR, "model.json")
SCHEDULE_FILE = os.path.join(DATA_DIR, "schedule.csv")
os.makedirs(DATA_DIR, exist_ok=True)
MAX_WEEKS = 18
MODEL_FEATURES = ["elo_diff", "temp_c", "wind_kph", "precip_prob"]

TEAMS = [
    "BUF", "MIA", "NE", "NYJ", "BAL", "CIN", "CLE", "PIT",
    "HOU", "IND", "JAX", "TEN", "DEN", "KC", "LV", "LAC",
    "DAL", "NYG", "PHI", "WAS", "CHI", "DET", "GB", "MIN",
    "ATL", "CAR", "NO", "TB", "ARI", "LAR", "SF", "SEA"
]

# --------------------------------------------------------------
# 🧠 Model
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

    model = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.08)
    model.fit(df[MODEL_FEATURES].values, y.values)
    model.save_model(MODEL_FILE)
    return model

# --------------------------------------------------------------
# 🏈 ESPN Scraper
# --------------------------------------------------------------
@st.cache_data(ttl=604800)
def fetch_schedule(season: int):
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

            odds = (comp.get("odds") or [{}])[0]
            spread = odds.get("details", "N/A")
            over_under = odds.get("overUnder", np.nan)
            kickoff = comp.get("date", None)

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
                "status_text": short_detail
            })

    df = pd.DataFrame(games)
    if df.empty:
        df = generate_mock_schedule(season)
    df.to_csv(SCHEDULE_FILE, index=False)
    return df

def generate_mock_schedule(season: int):
    np.random.seed(season)
    rows = []
    for week in range(1, MAX_WEEKS + 1):
        np.random.shuffle(TEAMS)
        for i in range(0, len(TEAMS), 2):
            home, away = TEAMS[i], TEAMS[i+1]
            rows.append({
                "season": season,
                "week": week,
                "home_team": home,
                "away_team": away,
                "home_logo": f"https://a.espncdn.com/i/teamlogos/nfl/500/{home.lower()}.png",
                "away_logo": f"https://a.espncdn.com/i/teamlogos/nfl/500/{away.lower()}.png",
                "kickoff_et": (datetime.now() + timedelta(days=(week-1)*7)).isoformat(),
                "spread": f"-{np.random.randint(1,8)}",
                "over_under": np.random.randint(38, 55),
                "home_score": np.nan,
                "away_score": np.nan,
                "state": "pre",
                "status_text": "Scheduled"
            })
    return pd.DataFrame(rows)

# --------------------------------------------------------------
# 🧮 Helpers
# --------------------------------------------------------------
def parse_spread(value):
    if not isinstance(value, str) or value in ["N/A", "", None]:
        return np.nan
    try:
        num = ''.join(ch for ch in value if ch in "+-.0123456789")
        if num == "" or num == ".":
            return np.nan
        return float(num)
    except Exception:
        return np.nan


def simulate_features(df, week=1):
    np.random.seed(week * 123)
    df["elo_diff"] = np.random.normal(0, 100, len(df))
    df["temp_c"] = np.random.uniform(-5, 25, len(df))
    df["wind_kph"] = np.random.uniform(0, 25, len(df))
    df["precip_prob"] = np.random.uniform(0, 1, len(df))
    df["over_under"] = df["over_under"].fillna(
        np.clip(45 + (df["elo_diff"]/100)*2 + np.random.normal(0, 2, len(df)), 37, 56)
    ).round(1)
    return df

# --------------------------------------------------------------
# 🧾 Compute model record (safe caching)
# --------------------------------------------------------------
@st.cache_data(ttl=1800)
def compute_model_record(schedule_hash: str, sched_df: pd.DataFrame):
    df = sched_df.query("state == 'post' and home_score.notna() and away_score.notna()").copy()
    if df.empty:
        return (0, 0, 0.0)
    df = simulate_features(df)
    X = df[MODEL_FEATURES].astype(float)
    model = load_or_train_model()
    df["home_win_prob"] = model.predict_proba(X)[:,1]
    df["predicted_home_win"] = (df["home_win_prob"] >= 0.5)
    df["actual_home_win"] = df["home_score"] > df["away_score"]
    correct = (df["predicted_home_win"] == df["actual_home_win"]).sum()
    total = len(df)
    pct = (correct / total) * 100 if total > 0 else 0
    return (correct, total - correct, pct)

# --------------------------------------------------------------
# 🔢 Sidebar
# --------------------------------------------------------------
st.sidebar.header("🏈 DJBets NFL Predictor")
season = st.sidebar.selectbox("Season", [2026, 2025, 2024], index=1)
week = st.sidebar.selectbox("Week", list(range(1, MAX_WEEKS+1)), index=0)
if st.sidebar.button("♻️ Refresh Schedule"):
    fetch_schedule.clear()
    st.rerun()

# --------------------------------------------------------------
# 📊 Load data
# --------------------------------------------------------------
model = load_or_train_model()
sched = fetch_schedule(season)
sched["kickoff_et"] = pd.to_datetime(sched["kickoff_et"], errors="coerce")

week_df = sched.query("week == @week").copy()
if week_df.empty:
    st.warning("No games found for this week.")
    st.stop()

# Generate a short hash of schedule for caching
sched_hash = hashlib.sha1(pd.util.hash_pandas_object(sched, index=False).values).hexdigest()

# Compute model record safely
correct, incorrect, pct = compute_model_record(sched_hash, sched)
st.sidebar.markdown("### 📈 Model Record")
st.sidebar.markdown(f"**Record:** {correct}-{incorrect} ({pct:.1f}%)")
st.sidebar.caption("Updated from completed ESPN games")
st.sidebar.markdown("---")
st.sidebar.markdown("🟩 = Home favored\n🟥 = Away favored")
st.sidebar.caption("Bar represents predicted home win probability")

# --------------------------------------------------------------
# 🎯 Predict
# --------------------------------------------------------------
week_df = simulate_features(week_df, week)
X = week_df[MODEL_FEATURES].astype(float)
week_df["home_win_prob"] = model.predict_proba(X)[:,1]
week_df["predicted_spread"] = np.round(-7 * (week_df["home_win_prob"] - 0.5), 1)
week_df["predicted_total"] = np.round(week_df["over_under"] + np.random.normal(0, 2, len(week_df)), 1)

# --------------------------------------------------------------
# 🎨 Display
# --------------------------------------------------------------
st.title(f"🏈 DJBets NFL Predictor — Week {week} ({season})")

for _, row in week_df.iterrows():
    vegas_spread = parse_spread(row["spread"])
    state = row.get("state", "pre")
    color = {"pre": "🟡 Upcoming", "in": "🟢 Live", "post": "🔵 Final"}.get(state, "⚪ Unknown")
    bg = "rgba(0,255,0,0.08)" if row["home_win_prob"] > 0.55 else "rgba(255,0,0,0.08)"

    st.markdown(f'<div style="background-color:{bg}; padding: 1rem; border-radius: 1rem;">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 3, 1])

    with c1:
        st.image(row["away_logo"], width=60)
        st.markdown(f"**{row['away_team']}**")

    with c3:
        st.image(row["home_logo"], width=60)
        st.markdown(f"**{row['home_team']}**")

    with c2:
        kickoff = row["kickoff_et"].strftime("%a %b %d, %I:%M %p") if pd.notna(row["kickoff_et"]) else "TBD"
        st.markdown(f"**{color}** — {row['status_text']}")
        st.markdown(f"**Kickoff:** {kickoff}")
        st.markdown(f"**Spread:** {row['spread']} | **Model Spread:** {row['predicted_spread']:+.1f}")
        st.markdown(f"**Over/Under:** {row['over_under']:.1f} | **Model Total:** {row['predicted_total']:.1f}")
        st.progress(row["home_win_prob"], text=f"🏠 Home Win Prob: {row['home_win_prob']*100:.1f}%")

        if state == "post" and not np.isnan(row["home_score"]) and not np.isnan(row["away_score"]):
            actual_winner = "home" if row["home_score"] > row["away_score"] else "away"
            predicted_winner = "home" if row["home_win_prob"] >= 0.5 else "away"
            correct = (actual_winner == predicted_winner)
            result = "✅ Correct" if correct else "❌ Incorrect"
            st.markdown(f"**Final:** {row['away_score']} - {row['home_score']} ({result})")
        elif state == "in":
            st.markdown(f"**Live:** {row['status_text']}")
        else:
            st.markdown("⏳ **Not started yet**")

        with st.expander("📊 Detailed Betting Analysis"):
            if not np.isnan(vegas_spread):
                spread_diff = row["predicted_spread"] - vegas_spread
                spread_rec = (
                    f"Bet **{row['home_team']}** (model stronger)" if spread_diff < 0 else f"Bet **{row['away_team']}** (+points)"
                )
            else:
                spread_rec = "No Vegas spread data"
            ou_diff = row["predicted_total"] - row["over_under"]
            ou_rec = "Bet **Over**" if ou_diff > 0 else "Bet **Under**"

            st.markdown(f"**Spread Recommendation:** {spread_rec}")
            st.markdown(f"**Over/Under Recommendation:** {ou_rec}")

            feats = {k: row[k] for k in MODEL_FEATURES}
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.bar(feats.keys(), feats.values(), color="skyblue")
            ax.set_title("Model Feature Inputs")
            st.pyplot(fig)

    st.markdown("</div><br>", unsafe_allow_html=True)

st.markdown("---")
st.caption("🏈 DJBets NFL Predictor v10.9 — fixed caching, sidebar record, clear color meaning.")
