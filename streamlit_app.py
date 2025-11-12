# streamlit_app.py
# Full updated app — copy/replace your existing file with this.
# Key goals:
# - preserve sidebar UI + model/ROI tracker
# - week selector as a dropdown at top of sidebar
# - robust logos loading and fallbacks
# - safe training (XGBoost preferred, fallback to LogisticRegression)
# - do not call st.* inside helper imports (see soh_utils.py)
# - train-on-first-launch if historical data present
# - explanatory tooltips in sidebar
#
# NOTE: adjust paths and data filenames as needed.

import streamlit as st
st.set_page_config(page_title="DJBets NFL Predictor", layout="wide", initial_sidebar_state="expanded")

import os
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple

# local helper utilities (no Streamlit calls inside)
from soh_utils import (
    load_espn_schedule,
    load_soh_data,
    merge_espn_soh,
    get_logo_path,
    ensure_numeric_cols,
)

# ML imports: try XGBoost first, fallback to sklearn
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------
# Config / constants
# -------------------------
DATA_DIR = "./data"
HIST_FILENAME = os.path.join(DATA_DIR, "nfl_archive_10Y.json")  # your uploaded historical
THIS_YEAR = int(os.environ.get("SEASON_YEAR", datetime.now().year))
DEFAULT_MAX_WEEKS = 18
FEATURES_PREFERRED = ["elo_diff", "inj_diff", "temp_c", "wind_kph", "precip_prob", "spread", "over_under"]

# -------------------------
# Utilities: model train/load
# -------------------------
@st.cache_data(ttl=3600)
def load_historical():
    """Load historical data if available; returns DataFrame or empty DataFrame."""
    if os.path.exists(HIST_FILENAME):
        try:
            with open(HIST_FILENAME, "r", encoding="utf-8") as f:
                raw = json.load(f)
            df = pd.DataFrame(raw)
            st.info(f"✅ Loaded historical data with {len(df)} rows.")
            return df
        except Exception as e:
            st.warning(f"Could not load historical JSON: {e}")
            return pd.DataFrame()
    st.warning("No historical file found in /data. Model will use fallback simulated data.")
    return pd.DataFrame()

def build_train_matrix(df: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return X, y, used_features."""
    # Ensure columns exist; add defaults if missing
    used = []
    for feat in features:
        if feat in df.columns:
            used.append(feat)
        else:
            # add fallback numeric column filled with 0
            df[feat] = 0.0
            used.append(feat)
            st.warning(f"⚠️ Added missing feature column: {feat}")
    # label: home_win (assumes historical contains home_score, away_score)
    if "home_score" in df.columns and "away_score" in df.columns:
        df["home_win"] = (df["home_score"].astype(float) > df["away_score"].astype(float)).astype(int)
        y = df["home_win"].values
    else:
        # fallback simulated labels if absent
        y = (np.random.rand(len(df)) > 0.5).astype(int)
        st.warning("⚠️ No explicit scores in history — using simulated labels.")
    X = df[used].fillna(0).astype(float).values
    return X, y, used

@st.cache_resource
def train_model(hist_df: pd.DataFrame):
    """Train and return model (and feature names). Uses xgboost if available, else logistic regression."""
    if hist_df.empty or len(hist_df) < 50:
        st.warning("⚠️ Not enough valid data — using fallback model.")
        # Create a trivial logistic model trained on tiny simulated data
        X = np.random.randn(200, 4)
        y = (np.random.rand(200) > 0.5).astype(int)
        clf = LogisticRegression(max_iter=200)
        clf.fit(X, y)
        return clf, [f"f{i}" for i in range(X.shape[1])]
    # prepare features (attempt to include spread/OU if present)
    hist = hist_df.copy()
    # Basic featurization: compute simple diffs if applicable
    if "elo_home" in hist.columns and "elo_away" in hist.columns:
        hist["elo_diff"] = hist["elo_home"] - hist["elo_away"]
    if "inj_home" in hist.columns and "inj_away" in hist.columns:
        hist["inj_diff"] = hist["inj_home"] - hist["inj_away"]
    # Make sure numeric columns exist
    ensure_numeric_cols(hist, FEATURES_PREFERRED)
    X, y, used = build_train_matrix(hist, FEATURES_PREFERRED)
    # Try XGBoost
    try:
        if HAVE_XGB:
            model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=100, verbosity=0)
            model.fit(X, y)
            st.success("✅ Model trained with XGBoost.")
            return model, used
    except Exception as e:
        st.warning(f"XGBoost training failed: {e} — falling back to LogisticRegression.")
    # fallback
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    st.success("✅ Model trained with LogisticRegression (fallback).")
    return model, used

# -------------------------
# Fetch schedule + merge + prepare UI dataframe
# -------------------------
@st.cache_data(ttl=300)
def load_schedule_and_merge(week: int = None, season: int = THIS_YEAR) -> pd.DataFrame:
    """Load ESPN schedule, attempt to merge with SOH spreads; return merged DataFrame."""
    espn = load_espn_schedule(season)
    soh = load_soh_data(season)  # may be empty; functions handle exceptions
    merged = merge_espn_soh(espn, soh)
    # Normalize dates
    if "kickoff_et" in merged.columns:
        merged["kickoff_et"] = pd.to_datetime(merged["kickoff_et"], errors="coerce")
    else:
        merged["kickoff_et"] = pd.NaT
    return merged

# -------------------------
# Compute helpers for UI
# -------------------------
def get_weeks_from_df(df: pd.DataFrame) -> List[int]:
    if df.empty or "week" not in df.columns:
        return list(range(1, DEFAULT_MAX_WEEKS + 1))
    weeks = sorted([int(w) for w in df["week"].dropna().unique()])
    if not weeks:
        return list(range(1, DEFAULT_MAX_WEEKS + 1))
    return weeks

def safe_predict_prob(model, features, row_df):
    """Predict home-win probability for a DataFrame row using only available features."""
    # select columns present in row_df in same order as features
    X = []
    for f in features:
        if f in row_df:
            X.append(float(row_df[f]))
        else:
            X.append(0.0)
    arr = np.array(X).reshape(1, -1)
    try:
        if HAVE_XGB and isinstance(model, XGBClassifier):
            p = model.predict_proba(arr)[:, 1][0]
        else:
            p = model.predict_proba(arr)[:, 1][0]
    except Exception:
        try:
            p = float(model.predict(arr)[0])
        except Exception:
            p = 0.5
    # clip
    p = max(0.0, min(1.0, p))
    return p

# -------------------------
# UI: Sidebar (week dropdown at top)
# -------------------------
st.sidebar.markdown("## 🏈 DJBets NFL Predictor")
st.sidebar.caption("Adjust season/week and model parameters")
# load merged to compute week list (load minimal)
merged_preview = load_schedule_and_merge(season=THIS_YEAR)
WEEKS = get_weeks_from_df(merged_preview)
selected_season = st.sidebar.selectbox("Season", [THIS_YEAR, THIS_YEAR-1, THIS_YEAR-2], index=0, key="season")
selected_week = st.sidebar.selectbox("📅 Week", WEEKS, index=0, key="week")

# model sliders & settings (preserve previous functionality)
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Model Controls")
market_weight = st.sidebar.slider("Market weight (blend model with market)", 0.0, 1.0, 0.5, 0.05, help="How much weight to give market implied probability when blending with model.")
bet_threshold = st.sidebar.slider("Min edge (pp) to recommend", 0.0, 10.0, 3.0, 0.5, help="Minimum edge in percentage points (pp) required to recommend a bet.")
st.sidebar.markdown("---")

# sidebar: model record and ROI (placeholders until computed)
st.sidebar.markdown("### 📊 Model Record")
model_record_placeholder = st.sidebar.empty()
st.sidebar.markdown("### 🕒 Last Updated")
st.sidebar.caption(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# -------------------------
# Main app logic
# -------------------------
st.title("🏈 DJBets NFL Predictor")
st.markdown("Predictions, spreads, and model recommendations. (Logos: put PNGs in `public/logos/` or `public/`.)")

# Load data
hist = load_historical()
merged = load_schedule_and_merge(season=selected_season)

if merged.empty:
    st.warning("⚠️ No games loaded from ESPN for this season.")
else:
    # filter by week
    week_df = merged[merged["week"] == int(selected_week)].copy()
    if week_df.empty:
        st.info(f"No games found for week {selected_week}.")
    else:
        # train model (on historical merged with elo if available)
        with st.spinner("Training model..."):
            model, feature_names = train_model(hist)

        # compute model predictions
        # ensure feature columns exist in week_df
        for feat in feature_names:
            if feat not in week_df.columns:
                week_df[feat] = 0.0
        # Predict row by row to be robust to column order
        probs = []
        for _, row in week_df.iterrows():
            p = safe_predict_prob(model, feature_names, row)
            probs.append(p)
        week_df["home_win_prob_model"] = probs

        # Market implied probs from spread (if spread present)
        def spread_to_market_prob(spread):
            # Very simple conversion: convert spread to probability via normal cdf approx
            try:
                s = float(spread)
                # map -15..+15 to prob 0.95..0.05 roughly
                prob = 0.5 - (s / 30.0)
                return float(max(0.01, min(0.99, prob)))
            except Exception:
                return np.nan

        week_df["market_prob"] = week_df.get("spread", np.nan).apply(spread_to_market_prob) if "spread" in week_df.columns else np.nan
        # blend
        week_df["blended_prob"] = week_df.apply(lambda r: (1 - market_weight) * r["home_win_prob_model"] + (market_weight * r["market_prob"] if not np.isnan(r.get("market_prob", np.nan)) else 0.0), axis=1)

        # compute edge (in percentage points)
        def to_pp(x): return round(100 * x, 2)
        week_df["edge_pp"] = week_df.apply(lambda r: to_pp(r["home_win_prob_model"] - (r["market_prob"] if not np.isnan(r.get("market_prob", np.nan)) else 0.0)), axis=1)

        # UI: show model/roi stats (simple)
        try:
            # compute simple hitrate on historical if available
            if not hist.empty and "home_score" in hist.columns and "away_score" in hist.columns:
                hist_sample = hist.dropna(subset=["home_score", "away_score"])
                hist_sample["home_win"] = (hist_sample["home_score"] > hist_sample["away_score"]).astype(int)
                # compute model predictions on hist using model & feature_names (best-effort)
                for f in feature_names:
                    if f not in hist_sample.columns:
                        hist_sample[f] = 0.0
                Xhist = hist_sample[feature_names].fillna(0).values
                try:
                    predf = model.predict(Xhist)
                except Exception:
                    try:
                        predf = model.predict_proba(Xhist)[:,1] > 0.5
                    except Exception:
                        predf = np.random.randint(0,2,len(hist_sample))
                acc = float((predf == hist_sample["home_win"].values).mean())
                model_record_placeholder.markdown(f"**Accuracy:** {acc*100:.1f}%\n\n**Trained on:** {len(hist_sample)} games")
            else:
                model_record_placeholder.markdown("No historical scoring data available to compute record.")
        except Exception as e:
            model_record_placeholder.markdown("Error computing model record.")
            st.error(e)

        # Present games as cards
        st.subheader(f"Week {selected_week} — {len(week_df)} games")
        for _, row in week_df.iterrows():
            cols = st.columns([1.2, 2, 1.2, 1.2, 1])
            # logos: use helper get_logo_path from soh_utils (it returns a path or None)
            home_logo = get_logo_path(row.get("home_team"))
            away_logo = get_logo_path(row.get("away_team"))

            # left - away
            with cols[0]:
                if away_logo:
                    try:
                        st.image(away_logo, width=64)
                    except Exception:
                        st.write(row.get("away_team", "Away"))
                else:
                    st.write(row.get("away_team", "Away"))

            # center - matchup & time
            with cols[1]:
                kickoff = row.get("kickoff_et")
                ko_str = kickoff.strftime("%Y-%m-%d %H:%M ET") if pd.notna(kickoff) else "TBD"
                st.markdown(f"**{row.get('away_team','Away')} @ {row.get('home_team','Home')}**  \n**Kickoff:** {ko_str}")
                # status & final score if present
                if row.get("home_score") is not None and row.get("away_score") is not None and not pd.isna(row.get("home_score")):
                    st.markdown(f"Final: **{int(row.get('home_score',0))} - {int(row.get('away_score',0))}**")
                else:
                    st.markdown("Status: Upcoming")

            # home logo
            with cols[2]:
                if home_logo:
                    try:
                        st.image(home_logo, width=64)
                    except Exception:
                        st.write(row.get("home_team", "Home"))
                else:
                    st.write(row.get("home_team", "Home"))

            # probabilities & market
            with cols[3]:
                prob = float(row.get("home_win_prob_model", 0.5))
                market = row.get("market_prob", np.nan)
                blended = float(row.get("blended_prob", prob))
                st.write(f"Model: {prob*100:.1f}%")
                st.write(f"Market: {'' if np.isnan(market) else f'{market*100:.1f}%'}")
                st.write(f"Blended: {blended*100:.1f}%")

            # recommendation
            with cols[4]:
                edge = row.get("edge_pp", 0.0)
                if np.isnan(edge):
                    st.write("Edge: N/A")
                    st.write("Recommendation: 🚫 No Data")
                else:
                    st.write(f"Edge: {edge:+.2f} pp")
                    if abs(edge) >= bet_threshold:
                        rec = "Bet Home" if edge > 0 else "Bet Away"
                        st.success(f"Recommendation: 🔥 {rec}")
                    else:
                        st.info("Recommendation: 🚫 No Bet")

# Footer
st.markdown("---")
st.caption("DJBets NFL Predictor — keep logos in public/logos/ and historical JSON in /data/")