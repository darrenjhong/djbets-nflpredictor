# streamlit_app.py
"""
Full Streamlit app for DJBets NFL Predictor (v9.5-S -> extended).
- Uses ESPN scraping for schedule (simple HTTP parse).
- Uses SportsOddsHistory/local NFL archive (via soh_utils.load_soh_data) for historical spreads.
- Uses OddsAPI (if key available) **only for upcoming/current** games (not historical).
- Auto-trains on first launch (uses XGBoost fallback if not enough data).
- Sidebar contains logos, sliders for config, ROI and model record.
- Game cards are auto-expanded and show model + market info, prediction, quick analysis.

Drop this file and soh_utils.py into your repo root. Logos should be in ./public/ or ./public/logos/
"""

import os, json, time, math
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import streamlit as st
from soh_utils import load_soh_data, merge_espn_soh, fill_missing_spreads
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

# ---------- Config ----------
THIS_YEAR = int(os.getenv("SEASON_YEAR", datetime.now().year))
MAX_WEEKS = 18
DATA_DIR = os.path.join(os.getcwd(), "data")
PUBLIC_DIRS = [os.path.join(os.getcwd(),"public","logos"), os.path.join(os.getcwd(),"public"), os.path.join(os.getcwd(),"public","images")]

# Try to discover Odds API key: env or /data/odds_api_key.txt or /data/ODDS_API_KEY
ODDS_API_KEY = os.getenv("ODDS_API_KEY") or os.getenv("ODDSAPI_KEY") or None
if not ODDS_API_KEY:
    # try common files
    for fname in ["odds_api_key.txt","odds_api_key","ODDS_API_KEY.txt","ODDS_API_KEY"]:
        p = os.path.join(DATA_DIR, fname)
        if os.path.exists(p):
            try:
                with open(p,"r") as f:
                    ODDS_API_KEY = f.read().strip()
                    break
            except Exception:
                continue

# ---------- Helpers ----------
def log(msg):
    st.sidebar.text(msg)

def get_logo_path(team_name):
    """
    Team_name expected in various forms; user mentioned they have files like 'bears.png'.
    We try:
      public/logos/{slug}.png
      public/{slug}.png
      public/{slug}.svg
    """
    if team_name is None:
        return None
    slug = str(team_name).lower().strip()
    slug = "".join([c for c in slug.replace("&","and").replace(".","").replace(" ","_") if c.isalnum() or c=="_"])
    candidates = []
    for d in PUBLIC_DIRS:
        for ext in ["png","svg","jpg","jpeg","webp"]:
            candidates.append(os.path.join(d,f"{slug}.{ext}"))
            candidates.append(os.path.join(d,f"{slug.upper()}.{ext}"))
    for p in candidates:
        if os.path.exists(p):
            return p
    # last attempt: exact file in public root named team_name
    for d in PUBLIC_DIRS:
        p = os.path.join(d, team_name)
        if os.path.exists(p):
            return p
    return None

def simple_espn_schedule_scrape(season=THIS_YEAR):
    """
    Lightweight ESPN schedule fetcher (non-API). It uses ESPN's scoreboard/teams pages
    to create a schedule-like DataFrame. This is intentionally defensive: if ESPN structure changes,
    we fallback to returning an empty DataFrame.
    NOTE: scraping can be rate-limited; keep it small.
    """
    try:
        # ESPN full-season schedule isn't exposed in a simple stable JSON without API keys.
        # We'll try ESPN scoreboard for the current week range (0..18) to gather matchups.
        rows = []
        # try next 18 weeks around now (search by dates)
        base_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        # We will fetch for a few dates (today +/- 60 days) to gather many games
        for offset in range(-60, 120, 7):
            dt = (datetime.utcnow() + timedelta(days=offset)).strftime("%Y-%m-%d")
            try:
                r = requests.get(base_url, params={"dates": dt}, timeout=8)
                j = r.json()
                events = j.get("events", [])
                for ev in events:
                    try:
                        competitions = ev.get("competitions", [])
                        if not competitions: continue
                        comp = competitions[0]
                        status = comp.get("status", {}).get("type", {}).get("name","")
                        season = comp.get("season", {}).get("year", THIS_YEAR)
                        # week is sometimes in "week" field
                        week = comp.get("week") or comp.get("season",{}).get("number") or 0
                        home = None; away = None
                        for team in comp.get("competitors", []):
                            if team.get("homeAway") == "home":
                                home = team
                            else:
                                away = team
                        kickoff = comp.get("date")
                        home_team = home.get("team",{}).get("displayName") if home else None
                        away_team = away.get("team",{}).get("displayName") if away else None
                        home_score = home.get("score") if home else None
                        away_score = away.get("score") if away else None
                        rows.append({
                            "season": int(season),
                            "week": int(week) if week is not None else 0,
                            "home_team": str(home_team).lower() if home_team else "",
                            "away_team": str(away_team).lower() if away_team else "",
                            "home_score": pd.to_numeric(home_score, errors="coerce"),
                            "away_score": pd.to_numeric(away_score, errors="coerce"),
                            "kickoff_ts": pd.to_datetime(kickoff) if kickoff else pd.NaT,
                            "status": status
                        })
                    except Exception:
                        continue
            except Exception:
                continue
        if not rows:
            return pd.DataFrame()
        sdf = pd.DataFrame(rows).drop_duplicates(subset=["season","week","home_team","away_team","kickoff_ts"])
        # keep only this season's rows for relevance
        sdf["season"] = sdf["season"].fillna(THIS_YEAR).astype(int)
        return sdf
    except Exception:
        return pd.DataFrame()

def read_local_historical():
    try:
        soh = load_soh_data()
        if soh.empty:
            return soh
        # normalize team names
        soh["home_team"] = soh["home_team"].astype(str).str.lower().str.replace(r"[^a-z0-9 ]","",regex=True).str.strip()
        soh["away_team"] = soh["away_team"].astype(str).str.lower().str.replace(r"[^a-z0-9 ]","",regex=True).str.strip()
        return soh
    except Exception:
        return pd.DataFrame()

def fetch_oddsapi_for_gamepair(dt_from, dt_to, sport_key="americanfootball_nfl"):
    """
    Uses OddsAPI to fetch odds between two datetimes (ISO strings).
    Only used for upcoming/current games; requires ODDS_API_KEY.
    Returns DataFrame with columns: ['home_team','away_team','commence_time','book','spread','over_under']
    """
    if not ODDS_API_KEY:
        return pd.DataFrame()
    url = "https://api.the-odds-api.com/v4/sports/{}/odds-history/".format(sport_key)
    # Note: OddsAPI endpoint structure may differ; in free docs, we might use /v4/sports/{sport}/odds to get current odds.
    # We'll use a conservative call to odds endpoint. If account doesn't support history, we'll gracefully fallback.
    try:
        # use current odds endpoint - will return current lines for upcoming events
        odds_url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
        params = {"apiKey": ODDS_API_KEY, "regions":"us", "markets":"spreads,totals", "oddsFormat":"american"}
        r = requests.get(odds_url, params=params, timeout=8)
        data = r.json()
        rows = []
        for event in data:
            commence = event.get("commence_time")
            comps = event.get("bookmakers", [])
            # we'll get consensus/bookmaker spreads
            if not event.get("home_team") or not event.get("away_team"):
                continue
            home = event.get("home_team").lower()
            away = event.get("away_team").lower()
            best_spread = None
            best_ou = None
            for bk in comps:
                markets = bk.get("markets", [])
                for m in markets:
                    if m.get("key") == "spreads":
                        # get outcomes
                        outcomes = m.get("outcomes", [])
                        # find home spread outcome
                        for out in outcomes:
                            # outcome structure: {name: 'Home Team', point: -3, price: ...}
                            if out.get("name") and out.get("point") is not None:
                                # this point is the spread (positive means ??? depends on API; we'll try to interpret)
                                # Odds API gives point with sign; we assume point applies to the named team
                                if out.get("name").lower() == home:
                                    # this point is how many points favorite (neg favored?) Keep as-is
                                    best_spread = float(out.get("point"))
                    if m.get("key") == "totals":
                        outcomes = m.get("outcomes", [])
                        for out in outcomes:
                            # totals often come as two outcomes, we want "total" point
                            if out.get("point") is not None:
                                best_ou = float(out.get("point"))
                if best_spread is not None or best_ou is not None:
                    break
            rows.append({
                "home_team": home,
                "away_team": away,
                "commence_time": commence,
                "spread": best_spread,
                "over_under": best_ou
            })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# ---------------- Model utilities ----------------
@st.cache_data(show_spinner=False)
def train_model_from_history(hist_df):
    """
    Train a simple model using historical data (from SOH or your local archive).
    This function is defensive: it will create a fallback simulator model if training fails.
    Model features:
      - spread (home)
      - over_under
      - simple engineered features (if present)
    Label: home_win (home_score > away_score)
    Returns: trained xgboost model (sklearn wrapper) and feature list
    """
    if hist_df is None or hist_df.empty:
        # fallback: simple default model that returns 0.52 for home (dummy)
        model = None
        return None, ["spread","over_under"]
    df = hist_df.copy()
    # ensure numeric columns exist
    for c in ["home_score","away_score","spread","over_under"]:
        if c not in df.columns:
            df[c] = np.nan
    # label
    df["home_win"] = (pd.to_numeric(df["home_score"], errors="coerce") > pd.to_numeric(df["away_score"], errors="coerce")).astype(int)
    # drop games without scores (can't train on future games)
    train_df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    if train_df.empty:
        return None, ["spread","over_under"]
    # features
    features = []
    # use spread and over_under if present
    for cand in ["spread","over_under"]:
        if cand in train_df.columns:
            features.append(cand)
            train_df[cand] = pd.to_numeric(train_df[cand], errors="coerce").fillna(0)
    # add simple home/away season normalized feature if possible (not required)
    # trim to avoid weird huge datasets
    X = train_df[features].fillna(0)
    y = train_df["home_win"].astype(int)
    # If not many unique samples, fallback
    if len(X) < 30 or len(y.unique()) < 2:
        # not enough history -> return None to indicate fallback
        return None, features
    # basic train/test
    try:
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=100, max_depth=4, verbosity=0)
        model.fit(X.values, y.values)
        return model, features
    except Exception:
        return None, features

def predict_on_df(model, features, df):
    """
    Model prediction: returns array of home win probability, and attaches it to df
    If model is None, return fallback probabilities based on spread (logistic-ish).
    """
    out = df.copy()
    if model is None or not features:
        # fallback: convert spread to a home-win probability using logistic transform
        if "spread" not in out.columns:
            out["spread"] = 0.0
        out["home_win_prob_model"] = (1 / (1 + np.exp(-(-out["spread"]*0.15))))  # small coefficient so -7 spread -> ~0.75
        out["home_win_prob_model"] = out["home_win_prob_model"].fillna(0.5)
        return out
    # prepare X
    X = df.copy()
    for f in features:
        if f not in X.columns:
            X[f] = 0
    Xf = X[features].fillna(0)
    try:
        probs = model.predict_proba(Xf.values)[:,1]
        out["home_win_prob_model"] = probs
    except Exception:
        # fallback to simple rule
        out["home_win_prob_model"] = (1 / (1 + np.exp(-(-out["spread"]*0.15))))
    return out

# ---------- UI & App Flow ----------
st.set_page_config(page_title="DJBets NFL Predictor", layout="wide", initial_sidebar_state="expanded")

st.sidebar.markdown("## 🏈 DJBets NFL Predictor — DJBets")
# Week selector at top
# We'll populate weeks after we have merged data; to ensure the selector is at top, we create a placeholder
week_placeholder = st.sidebar.empty()

# Sidebar controls: season
st.sidebar.markdown("### Season")
season = st.sidebar.selectbox("Season", options=[THIS_YEAR, THIS_YEAR-1, THIS_YEAR-2], index=0, key="season_select")

# Model tuning sliders and explanations
st.sidebar.markdown("### Model Controls")
market_weight = st.sidebar.slider("Market weight (blend model/market)", 0.0, 1.0, 0.5, 0.05, key="market_weight")
bet_threshold_pp = st.sidebar.slider("Bet threshold (pp)", 0.0, 10.0, 3.0, 0.5, key="bet_threshold_pp")
st.sidebar.caption("Market weight blends model probability with market probability. Bet threshold is minimum edge in percentage points to place a bet.")

# small area for logos and model record
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Record")
model_record_box = st.sidebar.empty()

# ---------- Load data ----------
st.info("Loading schedule from ESPN (this may take a few seconds)...")
espn_sched = simple_espn_schedule_scrape(season=season)
if espn_sched is None or espn_sched.empty:
    st.warning("Unable to fetch ESPN schedule. Please upload a schedule.csv to /data or check network.")
    espn_sched = pd.DataFrame(columns=["season","week","home_team","away_team","kickoff_ts","status"])

# Normalize espn team names
espn_sched["home_team"] = espn_sched["home_team"].astype(str).str.lower().str.replace(r"[^a-z0-9 ]","",regex=True).str.strip()
espn_sched["away_team"] = espn_sched["away_team"].astype(str).str.lower().str.replace(r"[^a-z0-9 ]","",regex=True).str.strip()

st.info("Loading local historical SOH (if present)...")
soh_hist = read_local_historical()

# merge espn + soh per week (we'll do per-week later)
# but prepare full merged to extract available weeks
merged_full = merge_espn_soh(espn_sched, soh_hist, season=season)

# determine week options
weeks_available = sorted([int(w) for w in merged_full["week"].dropna().unique()]) if not merged_full.empty else list(range(1, MAX_WEEKS+1))
if not weeks_available:
    weeks_available = list(range(1, MAX_WEEKS+1))

# put the week selector into placeholder so it's at top
with week_placeholder.container():
    week = st.selectbox("📅 Week", options=weeks_available, index=0, key="week")

# show quick dataset counts
st.sidebar.markdown(f"Schedule rows: {len(espn_sched)}  \nHistorical rows: {len(soh_hist)}")

# ---------- Historical training data & model ----------
st.info("Preparing historical data and training model (auto-train) ...")
hist = soh_hist.copy()
# ensure required columns exist for training
for c in ["home_score","away_score","spread","over_under","season","week","home_team","away_team","date"]:
    if c not in hist.columns:
        hist[c] = np.nan

# train
model, model_features = train_model_from_history(hist)
if model is None:
    st.warning("Not enough valid historical training examples - using fallback model (spread-based heuristic).")
else:
    st.success("Model trained successfully.")

# ---------- Prepare week-specific schedule + odds ----------
week_df = merged_full[ merged_full["week"] == int(week) ].copy()
if week_df.empty:
    st.warning(f"No games found for Week {week}. Check schedule source or try another week.")
else:
    # use OddsAPI for upcoming games (next 14 days) if available
    upcoming_odds = pd.DataFrame()
    if ODDS_API_KEY:
        try:
            odds_df = fetch_oddsapi_for_gamepair(None,None)
            if not odds_df.empty:
                # merge by home/away normalized
                odds_df["home_team"] = odds_df["home_team"].astype(str).str.lower().str.replace(r"[^a-z0-9 ]","",regex=True).str.strip()
                odds_df["away_team"] = odds_df["away_team"].astype(str).str.lower().str.replace(r"[^a-z0-9 ]","",regex=True).str.strip()
                upcoming_odds = odds_df
        except Exception:
            upcoming_odds = pd.DataFrame()

    # merge week_df with upcoming_odds if present
    if not upcoming_odds.empty:
        week_df["home_team_norm"] = week_df["home_team"].astype(str).str.lower().str.replace(r"[^a-z0-9 ]","",regex=True).str.strip()
        week_df["away_team_norm"] = week_df["away_team"].astype(str).str.lower().str.replace(r"[^a-z0-9 ]","",regex=True).str.strip()
        w = pd.merge(
            week_df,
            upcoming_odds[["home_team","away_team","spread","over_under"]],
            left_on=["home_team_norm","away_team_norm"],
            right_on=["home_team","away_team"],
            how="left",
            suffixes=("","_oddsapi")
        )
        # where oddsapi present, override soh spread
        w["spread"] = w["spread_oddsapi"].combine_first(w["spread"])
        w["over_under"] = w["over_under_oddsapi"].combine_first(w["over_under"])
        week_df = w

    # ensure numeric spreads and totals exist
    week_df = fill_missing_spreads(week_df)

    # predict probabilities
    week_df = predict_on_df(model, model_features, week_df)

    # compute market probability from spread: convert spread -> market home-win prob approx
    # naive conversion: home_win_prob_market = logistic(-spread*0.15)
    week_df["home_win_prob_market"] = 1/(1+np.exp(-(-week_df["spread"]*0.15)))
    # blended probability
    week_df["home_win_prob_blended"] = (week_df["home_win_prob_model"]*(1-market_weight) + week_df["home_win_prob_market"]*market_weight)

    # edge (pp)
    week_df["edge_pp"] = (week_df["home_win_prob_blended"] - week_df["home_win_prob_market"]) * 100

    # recommendation
    def recommend_row(r):
        try:
            if math.isnan(r["edge_pp"]): return "🚫 No Bet"
            if abs(r["edge_pp"]) < bet_threshold_pp:
                return "🚫 No Bet"
            # recommend side vs spread
            if r["edge_pp"] > 0:
                return f"🛫 Bet Home ({r['edge_pp']:.1f} pp)"
            else:
                return f"🛫 Bet Away ({r['edge_pp']:.1f} pp)"
        except Exception:
            return "🚫 No Bet"
    week_df["recommendation"] = week_df.apply(recommend_row, axis=1)

# ---------- Model record & ROI (sidebar) ----------
def compute_model_record(hist_df_local, model_local):
    """
    Compute cumulative record of model on historical games using train model or fallback.
    Returns correct, incorrect, pct.
    """
    if hist_df_local is None or hist_df_local.empty:
        return 0,0,0.0
    df = hist_df_local.copy()
    # only games with final scores
    df = df[df["home_score"].notna() & df["away_score"].notna()]
    if df.empty:
        return 0,0,0.0
    # ensure spreads present
    df = fill_missing_spreads(df)
    df = predict_on_df(model_local, model_features, df)
    df["pred_home"] = (df["home_win_prob_model"] >= 0.5).astype(int)
    df["actual_home"] = (pd.to_numeric(df["home_score"], errors="coerce") > pd.to_numeric(df["away_score"], errors="coerce")).astype(int)
    correct = int((df["pred_home"] == df["actual_home"]).sum())
    incorrect = int((df["pred_home"] != df["actual_home"]).sum())
    pct = (correct/(correct+incorrect))*100 if (correct+incorrect)>0 else 0.0
    return correct, incorrect, pct

correct, incorrect, pct = compute_model_record(hist, model)
model_record_box.markdown(f"**Record:** {correct} - {incorrect}  \n**Accuracy:** {pct:.1f}%")

# ---------- Main layout ----------
st.title("🏈 DJBets NFL Predictor")
st.caption(f"Season {season} — Week {week} — Updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# show top-of-page model summary and controls
col1, col2, col3 = st.columns([1,2,1])
with col1:
    st.metric("Model Accuracy (hist)", f"{pct:.1f}%")
with col2:
    st.metric("Model Bets Recommended", week_df["recommendation"].value_counts().get("🛫 Bet Home (",0) if not week_df.empty else 0)
with col3:
    st.metric("Data Source", "ESPN + SOH/local" if not soh_hist.empty else "ESPN only")

# -------- Display games ----------
if week_df.empty:
    st.warning("No games found for this week.")
else:
    # show each game as expander (auto-open)
    for i, r in week_df.reset_index(drop=True).iterrows():
        home = r["home_team"]
        away = r["away_team"]
        ks = r.get("kickoff_ts", None)
        kickoff_str = pd.to_datetime(ks).strftime("%a %b %d %H:%M ET") if pd.notna(ks) else "TBD"
        home_logo = get_logo_path(home)
        away_logo = get_logo_path(away)
        # card layout
        exp_label = f"{away.capitalize()} @ {home.capitalize()} — {kickoff_str}"
        with st.expander(exp_label, expanded=True):
            c1, c2, c3, c4 = st.columns([1,3,3,2])
            with c1:
                # logos aligned: away left, home right with centered names
                if away_logo:
                    try:
                        st.image(away_logo, width=60)
                    except Exception:
                        st.write(away.capitalize())
                else:
                    st.write(away.capitalize())
                st.write("")  # spacer
                if home_logo:
                    try:
                        st.image(home_logo, width=60)
                    except Exception:
                        st.write(home.capitalize())
                else:
                    st.write(home.capitalize())
            with c2:
                st.markdown(f"**Model**  \nHome Win Prob: **{r.get('home_win_prob_model',0)*100:.1f}%**  \nMarket Prob: **{r.get('home_win_prob_market',np.nan)*100 if not pd.isna(r.get('home_win_prob_market')) else np.nan:.1f}%**  \nBlended: **{r.get('home_win_prob_blended',0)*100:.1f}%**")
                st.write(f"Edge: **{r.get('edge_pp', np.nan):+.1f} pp**  \nRecommendation: **{r.get('recommendation','🚫 No Bet')}**")
            with c3:
                st.write(f"Spread: **{r.get('spread', 'N/A')}**  \nO/U: **{r.get('over_under', 'N/A')}**")
                # predicted score (simple point model): use blended prob to estimate point diff
                try:
                    p = r.get("home_win_prob_model", 0.5)
                    # naive predicted margin from probability (logit)
                    margin = - (np.log((1/p)-1) ) / 0.15 if p not in (0,1) else 0.0
                    total = r.get("over_under", 43.9)
                    if pd.isna(total):
                        total = 43.9
                    predicted_home = (total + margin) / 2
                    predicted_away = (total - margin) / 2
                    st.write(f"Predicted score: **{predicted_home:.1f} - {predicted_away:.1f}**  \nPredicted total: **{predicted_home+predicted_away:.1f}**")
                except Exception:
                    st.write("Predicted score: N/A")
            with c4:
                # show final score if complete
                if pd.notna(r.get("home_score")) and pd.notna(r.get("away_score")):
                    st.markdown(f"**Final:** {int(r['home_score'])} - {int(r['away_score'])}")
                    # correctness
                    pred_home = (r.get("home_win_prob_model", 0.5) >= 0.5)
                    actual_home = (r.get("home_score",0) > r.get("away_score",0))
                    res = "Correct ✅" if pred_home == actual_home else "Wrong ❌"
                    st.markdown(f"Model was: **{res}**")
                else:
                    st.markdown("Status: Not started / In progress")
            # deeper analysis: show some feature values
            st.markdown("**Analysis**")
            st.write({
                "Elo diff (if present)": r.get("elo_diff", "n/a"),
                "Injury diff (if present)": r.get("inj_diff", "n/a"),
                "Wind (kph)": r.get("wind_kph","n/a"),
                "Temp (C)": r.get("temp_c","n/a")
            })

# ---------- Footer: top model bets & ROI ----------
st.markdown("---")
st.header("🏆 Top Model Bets of the Week")
try:
    top_bets = week_df.sort_values("edge_pp", ascending=False).head(10)
    if not top_bets.empty:
        cols = st.columns(5)
        for idx, row in top_bets.iterrows():
            with st.container():
                st.write(f"**{row['away_team'].capitalize()} @ {row['home_team'].capitalize()}**")
                st.write(f"Edge: {row['edge_pp']:+.1f} pp")
                st.write(f"Rec: {row['recommendation']}")
    else:
        st.write("No recommended bets this week.")
except Exception:
    st.write("No recommended bets this week.")

st.markdown("---")
st.caption("Notes: Odds API used only for current/upcoming games if configured. Historical odds loaded from local archive when available.")
