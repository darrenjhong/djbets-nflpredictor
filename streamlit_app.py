# streamlit_app.py — DJBets NFL Predictor v10.6
# Fixes: Tracker persistence, logo pathing, spread recommendations, robust CSV

import os
import io
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from PIL import Image

# --------------------------------------------------------------
# ⚙️ Setup
st.set_page_config(page_title="DJBets NFL Predictor", page_icon="🏈", layout="wide")

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
LOGO_DIR = ROOT_DIR / "logos"
MODEL_FILE = DATA_DIR / "model.json"
DATA_DIR.mkdir(exist_ok=True)
MAX_WEEKS = 18
MODEL_FEATURES = ["elo_diff", "inj_diff", "temp_c", "wind_kph", "precip_prob"]

# --------------------------------------------------------------
# 🖼️ Logo Loader (Works in Streamlit Cloud)
def load_logo(team_abbr):
    for path in [
        LOGO_DIR / f"{team_abbr}.png",
        ROOT_DIR / f"logos/{team_abbr}.png",
        Path.cwd() / f"logos/{team_abbr}.png",
    ]:
        if path.exists():
            try:
                return Image.open(path)
            except Exception:
                continue
    return None

# --------------------------------------------------------------
# 🧠 Model Training
def train_fresh_model():
    np.random.seed(42)
    df = pd.DataFrame({
        "elo_diff": np.random.normal(0, 100, 1000),
        "inj_diff": np.random.normal(0, 10, 1000),
        "temp_c": np.random.uniform(-5, 25, 1000),
        "wind_kph": np.random.uniform(0, 25, 1000),
        "precip_prob": np.random.uniform(0, 1, 1000),
    })
    logits = 0.02 * df["elo_diff"] + 0.01 * df["inj_diff"] - 0.03 * df["precip_prob"]
    p = 1 / (1 + np.exp(-logits))
    y = (np.random.uniform(0, 1, 1000) < p).astype(int)
    model = xgb.XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.07)
    model.fit(df[MODEL_FEATURES], y)
    model.save_model(str(MODEL_FILE))
    return model

@st.cache_resource
def load_or_train_model():
    if not MODEL_FILE.exists():
        return train_fresh_model()
    try:
        model = xgb.XGBClassifier()
        model.load_model(str(MODEL_FILE))
        return model
    except Exception:
        return train_fresh_model()

# --------------------------------------------------------------
# 🏈 ESPN Schedule Fetch
@st.cache_data(ttl=3600)
def fetch_schedule_espn(season, week):
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?year={season}&week={week}"
    r = requests.get(url)
    if r.status_code != 200:
        return pd.DataFrame()
    data = r.json()
    games = []
    for ev in data.get("events", []):
        comp = ev.get("competitions", [{}])[0]
        home = next((c for c in comp["competitors"] if c["homeAway"] == "home"), None)
        away = next((c for c in comp["competitors"] if c["homeAway"] == "away"), None)
        if not home or not away:
            continue
        state = comp.get("status", {}).get("type", {}).get("name", "").lower()
        odds = comp.get("odds", [{}])[0] if comp.get("odds") else {}
        spread = odds.get("spread", np.nan)
        ou = odds.get("overUnder", np.nan)
        games.append({
            "season": season,
            "week": week,
            "home_team": home["team"]["displayName"],
            "away_team": away["team"]["displayName"],
            "home_abbr": home["team"]["abbreviation"],
            "away_abbr": away["team"]["abbreviation"],
            "home_score": int(home.get("score", 0)),
            "away_score": int(away.get("score", 0)),
            "kickoff_et": comp.get("date"),
            "state": state,
            "spread": spread,
            "over_under": ou,
        })
    df = pd.DataFrame(games)
    df["kickoff_et"] = pd.to_datetime(df["kickoff_et"], errors="coerce")
    return df

# --------------------------------------------------------------
# 🧮 Feature simulation
def simulate_features(df, week):
    np.random.seed(week)
    df["elo_home"] = np.random.normal(1550, 100, len(df))
    df["elo_away"] = np.random.normal(1500, 100, len(df))
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    df["inj_diff"] = np.random.normal(0, 5, len(df))
    df["temp_c"] = np.random.normal(10, 8, len(df))
    df["wind_kph"] = np.random.uniform(0, 30, len(df))
    df["precip_prob"] = np.random.uniform(0, 1, len(df))
    return df

# --------------------------------------------------------------
# 🎛️ Sidebar
st.sidebar.markdown("## 🏈 DJBets NFL Predictor")
tab = st.sidebar.radio("View Mode", ["Predictor", "Model Tracker"])
season = st.sidebar.selectbox("Season", [2026, 2025, 2024], index=1)
week = st.sidebar.selectbox("Week", list(range(1, MAX_WEEKS + 1)), index=0)
alpha = st.sidebar.slider("Market Weight (α)", 0.0, 1.0, 0.6, 0.05)
edge_thresh = st.sidebar.slider("Bet Threshold (pp)", 0.0, 10.0, 3.0, 0.5)

# --------------------------------------------------------------
# 🧠 Model + Data
model = load_or_train_model()
sched = fetch_schedule_espn(season, week)
sched = simulate_features(sched, week)

if sched.empty:
    st.warning("No games found for this week.")
    st.stop()

X = sched[MODEL_FEATURES].fillna(0).astype(float)
sched["home_win_prob_model"] = model.predict_proba(X)[:, 1]
sched["market_prob_home"] = 1 / (1 + np.exp(-0.2 * sched["elo_diff"]))
sched["blended_prob_home"] = (1 - alpha) * sched["home_win_prob_model"] + alpha * sched["market_prob_home"]
sched["edge_pp"] = (sched["blended_prob_home"] - sched["market_prob_home"]) * 100

# --------------------------------------------------------------
# 🧾 Predictor Tab
if tab == "Predictor":
    st.title(f"🏈 DJBets NFL Predictor — {season} Week {week}")

    history_records = []
    for _, row in sched.iterrows():
        kickoff = row["kickoff_et"].strftime("%a %b %d %I:%M %p") if pd.notna(row["kickoff_et"]) else "TBD"
        prob = row["blended_prob_home"]

        if "final" in row["state"]:
            home_win = row["home_score"] > row["away_score"]
            model_home = row["blended_prob_home"] > 0.5
            result_tag = "✅ Correct" if home_win == model_home else "❌ Wrong"
            correct = int(home_win == model_home)
            pnl = 1 if correct else -1
        else:
            result_tag, correct, pnl = "⏳ Pending", np.nan, 0

        home_logo = load_logo(row["home_abbr"])
        away_logo = load_logo(row["away_abbr"])

        cols = st.columns([1, 3, 1])
        with cols[0]:
            if away_logo: st.image(away_logo, width=70)
            else: st.write(row["away_abbr"])
        with cols[1]:
            st.markdown(f"### {row['away_team']} @ {row['home_team']} ({result_tag})")
            st.caption(f"Kickoff: {kickoff}")
        with cols[2]:
            if home_logo: st.image(home_logo, width=70)
            else: st.write(row["home_abbr"])

        if "final" in row["state"]:
            st.markdown(f"**Final Score:** {row['away_score']} - {row['home_score']}")
        else:
            st.markdown("⏳ **Not Started**")

        conf = "🟢 High" if abs(row["edge_pp"]) > 5 else "🟡 Medium" if abs(row["edge_pp"]) > 2 else "🔴 Low"
        st.progress(min(max(prob, 0), 1), text=f"Home Win Probability: {prob*100:.1f}% ({conf})")

        # Spread recommendation
        spread = row.get("spread", np.nan)
        try:
            spread_val = float(str(spread).replace("+", "").replace("−", "-"))
        except:
            spread_val = 0.0

        if row["edge_pp"] > edge_thresh:
            rec = f"🏠 Bet Home ({spread_val:+.1f})"
        elif row["edge_pp"] < -edge_thresh:
            rec = f"🛫 Bet Away ({-spread_val:+.1f})"
        else:
            rec = "🚫 No Bet"

        st.markdown(f"**Edge:** {row['edge_pp']:+.2f} pp | **Spread:** {spread} | **Recommendation:** {rec}")

        history_records.append({
            "season": season,
            "week": week,
            "home": row["home_team"],
            "away": row["away_team"],
            "pred_prob_home": prob,
            "result": result_tag,
            "correct": correct,
            "pnl": pnl,
        })

    # Save predictions
    hist_path = DATA_DIR / "predictions_history.csv"
    hist_df = pd.DataFrame(history_records)
    if hist_path.exists():
        old = pd.read_csv(hist_path)
        hist_df = pd.concat([old, hist_df]).drop_duplicates(subset=["season", "week", "home", "away"], keep="last")
    hist_df.to_csv(hist_path, index=False)
    st.caption("v10.6 — Persistent tracker + logos + spread-aware recs")

# --------------------------------------------------------------
# 📊 Tracker Tab
else:
    st.title("📈 Model Tracker — Performance Overview")
    hist_path = DATA_DIR / "predictions_history.csv"
    if not hist_path.exists():
        st.info("No history yet. Run predictions first.")
        st.stop()

    hist = pd.read_csv(hist_path)
    hist = hist.dropna(subset=["week"])
    st.dataframe(hist, use_container_width=True)
    st.download_button("⬇️ Export CSV", hist.to_csv(index=False).encode(), "model_tracker.csv", "text/csv")

    acc_week = hist.dropna(subset=["correct"]).groupby("week")["correct"].mean() * 100
    pnl_week = hist.groupby("week")["pnl"].sum()

    st.markdown("### 🎯 Weekly Accuracy and ROI")
    fig, ax1 = plt.subplots(figsize=(6, 3))
    ax1.bar(acc_week.index, acc_week.values, color="#4CAF50", label="Accuracy (%)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xlabel("Week")
    ax2 = ax1.twinx()
    ax2.plot(pnl_week.index, pnl_week.values, color="#FFC107", marker="o", label="P&L")
    ax2.set_ylabel("Profit/Loss (units)")
    ax1.set_title("Weekly Model Performance")
    st.pyplot(fig)

    st.markdown("### 🏈 ROI by Team")
    team_roi = hist.groupby("home")["pnl"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3))
    team_roi.plot(kind="bar", ax=ax)
    ax.set_ylabel("P&L (units)")
    ax.set_title("Cumulative ROI by Team")
    st.pyplot(fig)
