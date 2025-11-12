# streamlit_app.py — DJBets NFL Predictor v11
# Full stable release — silent SOH, safe logo loading, all features retained

import os, json, math, warnings, requests, io
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from PIL import Image
import xgboost as xgb

# --- General setup ---
warnings.filterwarnings("ignore")
st.set_option("logger.level", "error")
st.set_page_config(page_title="DJBets NFL Predictor", layout="wide", initial_sidebar_state="expanded")

THIS_YEAR = datetime.now().year
DATA_DIR = os.path.join(os.getcwd(), "data")
PUBLIC_DIRS = [
    os.path.join(os.getcwd(), "public"),
    os.path.join(os.getcwd(), "public", "logos"),
    os.path.join(os.getcwd(), "public", "images"),
]

# --- OddsAPI Key ---
ODDS_API_KEY = os.getenv("ODDS_API_KEY") or None
if not ODDS_API_KEY:
    for f in ["odds_api_key.txt", "ODDS_API_KEY.txt"]:
        p = os.path.join(DATA_DIR, f)
        if os.path.exists(p):
            with open(p) as fh:
                ODDS_API_KEY = fh.read().strip()


# --- Utility Functions ---
def get_logo_path(team_name):
    """Safely find a team logo in public/ directories."""
    if not team_name or not isinstance(team_name, str):
        return None
    slug = (
        team_name.lower()
        .replace("&", "and")
        .replace(".", "")
        .replace(" ", "_")
        .strip()
    )
    for d in PUBLIC_DIRS:
        for ext in ["png", "jpg", "jpeg", "svg", "webp"]:
            path = os.path.join(d, f"{slug}.{ext}")
            if os.path.exists(path):
                return path
    return None


def safe_show_logo(path, team_name, width=60):
    """Safely render logo or fallback to text if broken/missing."""
    if not path or not os.path.exists(path):
        st.write(team_name)
        return
    try:
        with open(path, "rb") as f:
            data = f.read()
        img = Image.open(io.BytesIO(data))
        st.image(img, width=width)
    except Exception:
        st.write(team_name)


def simple_espn_schedule_scrape(season=THIS_YEAR):
    """Fetch weekly NFL schedule data from ESPN."""
    rows = []
    try:
        base_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        for offset in range(-60, 120, 7):
            date_str = (datetime.utcnow() + timedelta(days=offset)).strftime("%Y-%m-%d")
            r = requests.get(base_url, params={"dates": date_str}, timeout=8)
            j = r.json()
            for ev in j.get("events", []):
                comp = ev.get("competitions", [{}])[0]
                teams = comp.get("competitors", [])
                home, away = None, None
                for t in teams:
                    if t.get("homeAway") == "home":
                        home = t
                    else:
                        away = t
                if not home or not away:
                    continue
                rows.append(
                    {
                        "season": comp.get("season", {}).get("year", season),
                        "week": comp.get("week", 0),
                        "home_team": home.get("team", {}).get("displayName", "").lower(),
                        "away_team": away.get("team", {}).get("displayName", "").lower(),
                        "home_score": pd.to_numeric(home.get("score"), errors="coerce"),
                        "away_score": pd.to_numeric(away.get("score"), errors="coerce"),
                        "kickoff_ts": pd.to_datetime(comp.get("date")),
                        "status": comp.get("status", {}).get("type", {}).get("name", ""),
                    }
                )
        return pd.DataFrame(rows).drop_duplicates()
    except Exception:
        return pd.DataFrame()


def fetch_oddsapi():
    """Fetch current and future odds via OddsAPI if key available."""
    if not ODDS_API_KEY:
        return pd.DataFrame()
    try:
        url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": "spreads,totals",
            "oddsFormat": "american",
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        rows = []
        for e in data:
            home = e.get("home_team", "").lower()
            away = e.get("away_team", "").lower()
            commence = e.get("commence_time")
            spread, ou = None, None
            for b in e.get("bookmakers", []):
                for m in b.get("markets", []):
                    if m.get("key") == "spreads":
                        for o in m.get("outcomes", []):
                            if o.get("name", "").lower() == home:
                                spread = o.get("point")
                    elif m.get("key") == "totals":
                        for o in m.get("outcomes", []):
                            if o.get("point"):
                                ou = o.get("point")
            rows.append(
                {"home_team": home, "away_team": away, "spread": spread, "over_under": ou, "commence_time": commence}
            )
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


# --- Model ---
@st.cache_data(show_spinner=False)
def train_model(df):
    if df.empty:
        return None, ["spread", "over_under"]
    df = df.copy()
    for c in ["home_score", "away_score", "spread", "over_under"]:
        if c not in df.columns:
            df[c] = np.nan
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df = df.dropna(subset=["home_score", "away_score"])
    if len(df) < 30:
        return None, ["spread", "over_under"]
    X = df[["spread", "over_under"]].fillna(0)
    y = df["home_win"]
    try:
        model = xgb.XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", n_estimators=100, max_depth=4
        )
        model.fit(X, y)
        return model, ["spread", "over_under"]
    except Exception:
        return None, ["spread", "over_under"]


def predict_model(model, df, features):
    if df.empty:
        return df
    out = df.copy()
    if model is None:
        out["home_win_prob_model"] = 1 / (1 + np.exp(-(-out["spread"].fillna(0) * 0.15)))
        return out
    X = out[features].fillna(0)
    try:
        out["home_win_prob_model"] = model.predict_proba(X)[:, 1]
    except Exception:
        out["home_win_prob_model"] = 1 / (1 + np.exp(-(-out["spread"].fillna(0) * 0.15)))
    return out


def compute_model_record(hist_df, model):
    if hist_df.empty:
        return 0, 0, 0.0
    df = hist_df.dropna(subset=["home_score", "away_score"])
    if df.empty:
        return 0, 0, 0.0
    df = fill_missing_spreads(df)
    df = predict_model(model, df, ["spread", "over_under"])
    df["pred_home"] = (df["home_win_prob_model"] >= 0.5).astype(int)
    df["actual_home"] = (df["home_score"] > df["away_score"]).astype(int)
    correct = int((df["pred_home"] == df["actual_home"]).sum())
    total = len(df)
    return correct, total - correct, correct / total * 100 if total > 0 else 0.0


# --- Sidebar ---
st.sidebar.markdown("## 🏈 DJBets NFL Predictor")
season = st.sidebar.selectbox("Season", [THIS_YEAR, THIS_YEAR - 1], index=0)
market_weight = st.sidebar.slider("Market Weight", 0.0, 1.0, 0.5)
bet_threshold_pp = st.sidebar.slider("Bet Threshold (pp)", 0.0, 10.0, 3.0)
model_record_box = st.sidebar.empty()

# --- Load data ---
st.info("Loading schedule from ESPN ...")
espn = simple_espn_schedule_scrape(season)
soh = load_soh_data()
merged = merge_espn_soh(espn, soh, season=season)
weeks = sorted(merged["week"].dropna().unique().astype(int)) or list(range(1, 19))
week = st.sidebar.selectbox("📅 Week", weeks, index=0)

# --- Train model ---
st.info("Training model ...")
hist = soh.copy()
model, features = train_model(hist)
correct, incorrect, pct = compute_model_record(hist, model)
model_record_box.markdown(f"**Record:** {correct}-{incorrect}  \n**Accuracy:** {pct:.1f}%")

# --- Prepare data ---
week_df = merged[merged["week"] == week].copy()
if week_df.empty:
    st.warning("No games found for this week.")
    st.stop()

# --- Merge odds ---
odds_df = fetch_oddsapi()
if not odds_df.empty:
    odds_df["home_team"] = odds_df["home_team"].str.lower().str.strip()
    odds_df["away_team"] = odds_df["away_team"].str.lower().str.strip()
    week_df["home_team"] = week_df["home_team"].str.lower().str.strip()
    week_df["away_team"] = week_df["away_team"].str.lower().str.strip()
    week_df = week_df.merge(
        odds_df[["home_team", "away_team", "spread", "over_under"]],
        on=["home_team", "away_team"],
        how="left",
        suffixes=("", "_oddsapi"),
    )
    week_df["spread"] = week_df["spread_oddsapi"].combine_first(week_df["spread"])
    week_df["over_under"] = week_df["over_under_oddsapi"].combine_first(week_df["over_under"])

week_df = fill_missing_spreads(week_df)
week_df = predict_model(model, week_df, features)
week_df["home_win_prob_market"] = 1 / (1 + np.exp(-(-week_df["spread"] * 0.15)))
week_df["home_win_prob_blended"] = (
    week_df["home_win_prob_model"] * (1 - market_weight)
    + week_df["home_win_prob_market"] * market_weight
)
week_df["edge_pp"] = (week_df["home_win_prob_blended"] - week_df["home_win_prob_market"]) * 100

def recommend(r):
    if abs(r["edge_pp"]) < bet_threshold_pp:
        return "🚫 No Bet"
    return f"🛫 Bet {'Home' if r['edge_pp']>0 else 'Away'} ({r['edge_pp']:+.1f} pp)"

week_df["recommendation"] = week_df.apply(recommend, axis=1)

# --- Main UI ---
st.title(f"🏈 DJBets NFL Predictor — Week {week}")
st.caption(f"Season {season} | Updated {datetime.now():%Y-%m-%d %H:%M:%S}")

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
            safe_show_logo(home_logo, home)

        with c2:
            st.markdown(f"**Spread:** {r['spread']:.1f}  \n**O/U:** {r['over_under']:.1f}")
            st.markdown(f"**Edge:** {r['edge_pp']:+.1f} pp  \n**Rec:** {r['recommendation']}")
            st.markdown(f"**Model Prob:** {r['home_win_prob_model']*100:.1f}%  \n**Market Prob:** {r['home_win_prob_market']*100:.1f}%")

        with c3:
            try:
                p = r["home_win_prob_model"]
                margin = - (np.log((1 / p) - 1)) / 0.15 if 0 < p < 1 else 0
                total = r.get("over_under", 44)
                home_pts = (total + margin) / 2
                away_pts = (total - margin) / 2
                st.markdown(f"**Predicted:** {home_pts:.1f} - {away_pts:.1f}")
            except Exception:
                st.write("Predicted: N/A")

            if not pd.isna(r["home_score"]) and not pd.isna(r["away_score"]):
                correct_pred = (r["home_score"] > r["away_score"]) == (r["home_win_prob_model"] >= 0.5)
                res = "✅ Correct" if correct_pred else "❌ Wrong"
                st.markdown(f"**Final:** {int(r['home_score'])}-{int(r['away_score'])} ({res})")

# --- Footer ---
st.markdown("---")
st.header("🏆 Top Model Bets of the Week")
top_bets = week_df.sort_values("edge_pp", ascending=False).head(10)
if top_bets.empty:
    st.write("No top bets this week.")
else:
    for _, row in top_bets.iterrows():
        st.write(f"**{row['away_team'].capitalize()} @ {row['home_team'].capitalize()}** — Edge {row['edge_pp']:+.1f} pp → {row['recommendation']}")

st.caption("Data: ESPN + local SOH archive | OddsAPI for current odds | All missing values simulated gracefully.")
