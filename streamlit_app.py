import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import xgboost as xgb

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
st.set_page_config(page_title="DJBets NFL Predictor", page_icon="🏈", layout="wide")

DATA_DIR = Path("data")
HIST_PATTERN = "historical*.csv"
SCHED_PATTERN = "schedule_*.csv"

# ---------------------------------------------
# LOAD DATA (ALWAYS PICK LATEST)
# ---------------------------------------------
@st.cache_data(show_spinner=False)
def load_latest_schedule():
    files = sorted(DATA_DIR.glob(SCHED_PATTERN))
    if not files:
        st.error("No schedule file found in /data.")
        st.stop()
    latest = files[-1]
    df = pd.read_csv(latest)
    df["kickoff_et"] = pd.to_datetime(df["kickoff_et"])
    st.session_state["active_schedule_file"] = latest.name
    st.session_state["season"] = int(latest.stem.split("_")[-1])
    return df

@st.cache_data(show_spinner=False)
def load_latest_history():
    files = sorted(DATA_DIR.glob(HIST_PATTERN))
    if not files:
        st.error("No historical data file found in /data.")
        st.stop()
    latest = files[-1]
    df = pd.read_csv(latest)
    st.session_state["active_historical_file"] = latest.name
    return df

# ---------------------------------------------
# MODEL TRAINING
# ---------------------------------------------
@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df[["elo_diff", "temp_c", "wind_kph", "precip_prob"]]
    y = df["home_win"]
    model = xgb.XGBClassifier(
        n_estimators=350, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.85, reg_lambda=1.0,
        eval_metric="logloss", tree_method="hist"
    )
    model.fit(X, y)
    return model

# ---------------------------------------------
# FEATURE SIMULATION (for demo)
# ---------------------------------------------
CLIMATE = {
    "ARI":"warm","ATL":"warm","BAL":"cold","BUF":"cold","CAR":"mild","CHI":"cold","CIN":"cold","CLE":"cold",
    "DAL":"warm","DEN":"cold","DET":"dome","GB":"cold","HOU":"dome","IND":"dome","JAX":"warm","KC":"cold",
    "LV":"dome","LAC":"mild","LAR":"mild","MIA":"warm","MIN":"dome","NE":"cold","NO":"dome","NYG":"cold",
    "NYJ":"cold","PHI":"cold","PIT":"cold","SF":"mild","SEA":"mild","TB":"warm","TEN":"mild","WAS":"cold"
}

def simulate_features(df):
    out = []
    rng = np.random.default_rng(100 + int(df["week"].iloc[0]))
    for _, r in df.iterrows():
        clim = CLIMATE.get(r["home_team"], "mild")
        month = r["kickoff_et"].month
        if clim == "dome":
            temp_c, wind, precip = 21, 0, 0
        else:
            base = {"warm":26,"mild":17,"cold":7}.get("cold" if month in [12,1] else clim, 17)
            temp_c = int(np.clip(rng.normal(base,4), -5, 30))
            wind = int(np.clip(rng.normal(10,4), 0, 35))
            precip = int(np.clip(rng.normal(25,15), 0, 100))
        elo_diff = int(np.clip(rng.normal(0, 100), -200, 200))
        out.append({**r.to_dict(), "elo_diff":elo_diff, "temp_c":temp_c, "wind_kph":wind, "precip_prob":precip})
    return pd.DataFrame(out)

def nice_kick(dt: datetime)->str:
    try:
        return dt.strftime("%a, %b %-d • %-I:%M %p ET")
    except:
        return dt.strftime("%a, %b %d • %I:%M %p ET")

# ---------------------------------------------
# APP LAYOUT
# ---------------------------------------------
st.title("🏈 DJBets NFL Predictor — Live Auto Git Mode")

sched = load_latest_schedule()
hist = load_latest_history()
season = st.session_state["season"]
weeks = sorted(sched["week"].unique())
wk = st.sidebar.selectbox("Week", weeks, index=min(len(weeks)-1, 0))
slots = st.sidebar.multiselect("Slots", ["TNF","SUN-1","SUN-2","SNF","MNF"], default=[])

df_week = sched[(sched["season"]==season) & (sched["week"]==wk)].copy()
if slots:
    df_week = df_week[df_week["network_slot"].isin(slots)]

if df_week.empty:
    st.warning("No games found for this week.")
    st.stop()

model = train_model(hist)
feats = simulate_features(df_week)
probs = model.predict_proba(feats[["elo_diff","temp_c","wind_kph","precip_prob"]])[:,1]
feats["home_win_prob"] = probs
feats["predicted_winner"] = np.where(probs>=0.5, feats["home_team"], feats["away_team"])

# ---------------------------------------------
# DISPLAY
# ---------------------------------------------
st.markdown(f"### Season {season} • Week {wk}")
st.progress(float(feats["home_win_prob"].mean()))

for _, row in feats.iterrows():
    st.markdown(f"**{row['away_team']} @ {row['home_team']}** — {nice_kick(row['kickoff_et'])}")
    st.caption(f"Win %: {row['home_win_prob']*100:.1f}% | Predicted: {row['predicted_winner']}")
    st.divider()

st.caption(f"🔄 Data: {st.session_state['active_schedule_file']} | History: {st.session_state['active_historical_file']} | Updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
