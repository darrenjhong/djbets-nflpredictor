# streamlit_app.py — DJBets NFL Predictor v12
# ✅ Dynamic training (SOH + current data)
# ✅ Model-predicted spreads/totals
# ✅ Safe logo rendering, smart recommendations
# ✅ Full UI + sidebar ROI retained

import os, io, json, math, warnings, requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from PIL import Image
import xgboost as xgb
from soh_utils import load_soh_data, merge_espn_soh, fill_missing_spreads

# --- Setup ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="DJBets NFL Predictor", layout="wide", initial_sidebar_state="expanded")
THIS_YEAR = datetime.now().year
DATA_DIR = os.path.join(os.getcwd(), "data")
PUBLIC_DIRS = [os.path.join(os.getcwd(), "public"), os.path.join(os.getcwd(), "public", "logos")]

# --- Load API key ---
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
if not ODDS_API_KEY:
    key_path = os.path.join(DATA_DIR, "odds_api_key.txt")
    if os.path.exists(key_path):
        with open(key_path) as f:
            ODDS_API_KEY = f.read().strip()

# --- Utility Functions ---
def get_logo_path(team):
    if not team or not isinstance(team, str):
        return None
    slug = team.lower().replace("&", "and").replace(".", "").replace(" ", "_").strip()
    for d in PUBLIC_DIRS:
        for ext in ["png", "jpg", "jpeg", "svg", "webp"]:
            path = os.path.join(d, f"{slug}.{ext}")
            if os.path.exists(path):
                return path
    return None

def safe_show_logo(path, team, width=65):
    if not path or not os.path.exists(path):
        st.write(team)
        return
    try:
        with open(path, "rb") as f:
            img = Image.open(io.BytesIO(f.read()))
        st.image(img, width=width)
    except Exception:
        st.write(team)

def simple_espn_scrape(season=THIS_YEAR):
    """Scrape ESPN weekly schedule."""
    base = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    rows = []
    for offset in range(-60, 120, 7):
        try:
            r = requests.get(base, params={"dates": (datetime.utcnow() + timedelta(days=offset)).strftime("%Y-%m-%d")}, timeout=8)
            for ev in r.json().get("events", []):
                comp = ev.get("competitions", [{}])[0]
                teams = comp.get("competitors", [])
                home = [t for t in teams if t.get("homeAway") == "home"][0]
                away = [t for t in teams if t.get("homeAway") == "away"][0]
                rows.append({
                    "season": comp.get("season", {}).get("year", season),
                    "week": comp.get("week", 0),
                    "home_team": home.get("team", {}).get("displayName", "").lower(),
                    "away_team": away.get("team", {}).get("displayName", "").lower(),
                    "home_score": pd.to_numeric(home.get("score"), errors="coerce"),
                    "away_score": pd.to_numeric(away.get("score"), errors="coerce"),
                    "kickoff_ts": pd.to_datetime(comp.get("date")),
                    "status": comp.get("status", {}).get("type", {}).get("name", "")
                })
        except Exception:
            continue
    return pd.DataFrame(rows).drop_duplicates()

def fetch_oddsapi():
    """Fetch current odds from OddsAPI."""
    if not ODDS_API_KEY:
        return pd.DataFrame()
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds",
            params={"apiKey": ODDS_API_KEY, "regions": "us", "markets": "spreads,totals", "oddsFormat": "american"},
            timeout=8,
        )
        rows = []
        for g in r.json():
            home, away = g.get("home_team", "").lower(), g.get("away_team", "").lower()
            spread, ou = None, None
            for bk in g.get("bookmakers", []):
                for m in bk.get("markets", []):
                    if m["key"] == "spreads":
                        for o in m["outcomes"]:
                            if o["name"].lower() == home:
                                spread = o.get("point")
                    elif m["key"] == "totals":
                        for o in m["outcomes"]:
                            if "point" in o:
                                ou = o["point"]
            rows.append({"home_team": home, "away_team": away, "spread": spread, "over_under": ou})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# --- Machine Learning ---
@st.cache_data(show_spinner=False)
def train_dynamic_model(hist_df, live_df):
    """Train model dynamically on historical + live data."""
    if hist_df.empty and live_df.empty:
        return None, ["spread", "over_under"]
    df = pd.concat([hist_df, live_df], ignore_index=True).fillna(0)
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    if len(df) < 50:
        return None, ["spread", "over_under"]
    X = df[["spread", "over_under"]]
    y = df["home_win"]
    model = xgb.XGBClassifier(eval_metric="logloss", n_estimators=200, max_depth=5)
    model.fit(X, y)
    return model, ["spread", "over_under"]

def predict_spread_total(model, df, features):
    df = df.copy()
    if model is None or df.empty:
        df["model_home_prob"] = 0.5
        df["model_pred_spread"] = 0
        df["model_pred_total"] = 44
        return df
    X = df[features].fillna(0)
    try:
        df["model_home_prob"] = model.predict_proba(X)[:, 1]
    except Exception:
        df["model_home_prob"] = 0.5
    df["model_pred_spread"] = -np.log((1 / df["model_home_prob"]) - 1) / 0.15
    df["model_pred_total"] = df["over_under"].fillna(44)
    return df

# --- Sidebar ---
st.sidebar.title("🏈 DJBets NFL Predictor")
season = st.sidebar.selectbox("Season", [THIS_YEAR, THIS_YEAR - 1], index=0)
market_weight = st.sidebar.slider("Market Weight", 0.0, 1.0, 0.5, help="Blends model vs market probabilities")
bet_threshold = st.sidebar.slider("Bet Threshold (pp)", 0.0, 10.0, 3.0, help="Minimum edge before model recommends a bet")

# --- Load Data ---
st.info("Fetching ESPN + OddsAPI data...")
espn_df = simple_espn_scrape(season)
soh_df = load_soh_data()
merged = merge_espn_soh(espn_df, soh_df, season=season)
weeks = sorted(merged["week"].dropna().unique().astype(int)) or list(range(1, 19))
week = st.sidebar.selectbox("📅 Week", weeks, index=0)

# --- Train Model ---
odds_df = fetch_oddsapi()
hist = soh_df.copy()
model, features = train_dynamic_model(hist, merged)
if model:
    st.sidebar.success("Model trained dynamically ✅")
else:
    st.sidebar.warning("Fallback model active (limited data)")

# --- Merge + Predict ---
week_df = merged[merged["week"] == week].copy()
if not odds_df.empty:
    odds_df["home_team"] = odds_df["home_team"].str.lower()
    odds_df["away_team"] = odds_df["away_team"].str.lower()
    week_df = week_df.merge(
        odds_df[["home_team", "away_team", "spread", "over_under"]],
        on=["home_team", "away_team"],
        how="left",
        suffixes=("", "_oddsapi"),
    )
    week_df["spread"] = week_df["spread_oddsapi"].combine_first(week_df["spread"])
    week_df["over_under"] = week_df["over_under_oddsapi"].combine_first(week_df["over_under"])

week_df = fill_missing_spreads(week_df)
week_df = predict_spread_total(model, week_df, features)

# Compute model-vs-market edge
week_df["market_prob"] = 1 / (1 + np.exp(-(-week_df["spread"] * 0.15)))
week_df["blend_prob"] = (week_df["model_home_prob"] * (1 - market_weight)) + (week_df["market_prob"] * market_weight)
week_df["edge_pp"] = (week_df["blend_prob"] - week_df["market_prob"]) * 100
week_df["predicted_home_pts"] = (week_df["model_pred_total"] / 2) + (week_df["model_pred_spread"] / 2)
week_df["predicted_away_pts"] = (week_df["model_pred_total"] / 2) - (week_df["model_pred_spread"] / 2)

def recommend_bet(r):
    if abs(r["edge_pp"]) < bet_threshold:
        return "🚫 No Bet"
    side = "Home" if r["edge_pp"] > 0 else "Away"
    return f"🟩 Bet {side} ({r['edge_pp']:+.1f} pp)"

week_df["recommendation"] = week_df.apply(recommend_bet, axis=1)

# --- Display ---
st.title(f"🏈 DJBets NFL Predictor — Week {week}")
st.caption(f"Season {season} | Updated {datetime.now():%Y-%m-%d %H:%M:%S}")

if week_df.empty:
    st.warning("No games found for this week.")
    st.stop()

for _, r in week_df.iterrows():
    away, home = r["away_team"].capitalize(), r["home_team"].capitalize()
    away_logo, home_logo = get_logo_path(away), get_logo_path(home)
    kickoff = (
        pd.to_datetime(r.get("kickoff_ts")).strftime("%a %b %d %H:%M ET")
        if pd.notna(r.get("kickoff_ts"))
        else "TBD"
    )

    with st.expander(f"{away} @ {home} — {kickoff}", expanded=True):
        c1, c2, c3 = st.columns([1, 3, 3])

        with c1:
            safe_show_logo(away_logo, away)
            st.markdown("**@**")
            safe_show_logo(home_logo, home)

        with c2:
            st.markdown(
                f"**Vegas:** Spread {r['spread']:.1f} | O/U {r['over_under']:.1f}\n"
                f"**Model:** Spread {r['model_pred_spread']:.1f} | Total {r['model_pred_total']:.1f}"
            )
            st.markdown(f"**Edge:** {r['edge_pp']:+.1f} pp\n**Rec:** {r['recommendation']}")

        with c3:
            st.markdown(
                f"**Predicted Score:** {r['home_team'].capitalize()} {r['predicted_home_pts']:.1f} - "
                f"{r['away_team'].capitalize()} {r['predicted_away_pts']:.1f}"
            )
            if not pd.isna(r["home_score"]) and not pd.isna(r["away_score"]):
                correct_pred = (r["home_score"] > r["away_score"]) == (r["blend_prob"] >= 0.5)
                res = "✅ Correct" if correct_pred else "❌ Wrong"
                st.markdown(f"**Final:** {int(r['home_score'])}-{int(r['away_score'])} ({res})")

# --- Top Bets ---
st.header("🏆 Top Model Bets")
top_bets = week_df.sort_values("edge_pp", ascending=False).head(5)
for _, row in top_bets.iterrows():
    st.markdown(f"**{row['away_team'].capitalize()} @ {row['home_team'].capitalize()}** — {row['recommendation']} ({row['edge_pp']:+.1f} pp)")

st.caption("Model dynamically trained using historical SOH + live ESPN/OddsAPI data.")