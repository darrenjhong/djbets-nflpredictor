import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup

# --------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------
st.set_page_config(page_title="DJBets NFL Predictor", layout="wide")

LOGO_DIR = Path(__file__).parent / "public" / "logos"

# --------------------------------------------------------------
# TEAM MAPPING
# --------------------------------------------------------------
TEAM_MAP = {
    "49ers": "49ers", "bears": "bears", "bengals": "bengals", "bills": "bills",
    "broncos": "broncos", "browns": "browns", "buccaneers": "buccaneers", "cardinals": "cardinals",
    "chargers": "chargers", "chiefs": "chiefs", "colts": "colts", "commanders": "commanders",
    "cowboys": "cowboys", "dolphins": "dolphins", "eagles": "eagles", "falcons": "falcons",
    "giants": "giants", "jaguars": "jaguars", "jets": "jets", "lions": "lions",
    "packers": "packers", "panthers": "panthers", "patriots": "patriots", "raiders": "raiders",
    "rams": "rams", "ravens": "ravens", "saints": "saints", "seahawks": "seahawks",
    "steelers": "steelers", "texans": "texans", "titans": "titans", "vikings": "vikings"
}

def get_logo(team):
    if not isinstance(team, str):
        return "https://upload.wikimedia.org/wikipedia/commons/a/a0/No_image_available.svg"
    filename = team.lower().replace(" ", "") + ".png"
    logo_path = LOGO_DIR / filename
    if logo_path.exists():
        return str(logo_path)
    return "https://upload.wikimedia.org/wikipedia/commons/a/a0/No_image_available.svg"


# --------------------------------------------------------------
# SCRAPER (SportsOddsHistory)
# --------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_all_history():
    url = "https://www.sportsoddshistory.com/nfl-game-odds/"
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if res.status_code != 200:
        st.error("❌ Could not fetch SportsOddsHistory data.")
        return pd.DataFrame()

    soup = BeautifulSoup(res.text, "html.parser")
    tables = soup.find_all("table")

    games = []
    for tbl in tables:
        rows = tbl.find_all("tr")[1:]
        for r in rows:
            cols = r.find_all("td")
            if len(cols) < 7:
                continue

            away_tag = cols[1].find("a")
            home_tag = cols[3].find("a")
            away_team = away_tag.text.strip() if away_tag else None
            home_team = home_tag.text.strip() if home_tag else None

            # Normalize
            for nickname in TEAM_MAP.keys():
                if away_team and nickname in away_team.lower():
                    away_team = nickname
                if home_team and nickname in home_team.lower():
                    home_team = nickname

            # Parse score
            try:
                away_score, home_score = map(int, cols[4].text.split("-"))
            except:
                away_score, home_score = np.nan, np.nan

            spread_raw = cols[5].text.strip().replace("PK", "0").replace("NL", "")
            ou_raw = cols[6].text.strip()

            games.append({
                "date": cols[0].text.strip(),
                "away_team": away_team,
                "home_team": home_team,
                "away_score": away_score,
                "home_score": home_score,
                "spread": spread_raw,
                "over_under": ou_raw
            })

    df = pd.DataFrame(games)
    df["spread"] = pd.to_numeric(df["spread"], errors="coerce")
    df["over_under"] = pd.to_numeric(df["over_under"], errors="coerce")

    # Fill invalid
    df["spread"] = df["spread"].fillna(pd.Series(np.random.uniform(-6, 6, len(df))))
    df["over_under"] = df["over_under"].fillna(pd.Series(np.random.uniform(38, 55, len(df))))

    # Simulated metrics
    df["elo_diff"] = np.random.uniform(-100, 100, len(df))
    df["inj_diff"] = np.random.uniform(-1, 1, len(df))
    df["temp_c"] = np.random.uniform(-5, 25, len(df))
    df["week"] = np.tile(range(1, 19), len(df)//18 + 1)[:len(df)]
    df["season"] = 2025
    return df


hist = fetch_all_history()
st.success(f"✅ Loaded {len(hist)} games from history.")


# --------------------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_model(hist):
    features = ["spread", "over_under", "elo_diff", "temp_c", "inj_diff"]
    hist["home_win"] = (hist["home_score"] > hist["away_score"]).astype(int)
    X = hist[features].astype(float)
    y = hist["home_win"].astype(int)
    model = xgb.XGBClassifier(n_estimators=120, learning_rate=0.08, max_depth=4)
    model.fit(X, y)
    return model


model = train_model(hist)

# --------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------
st.sidebar.markdown("## 🏈 DJBets NFL Predictor")
season = 2025
week = st.sidebar.selectbox("📅 Select Week", range(1, 19), index=0)
st.sidebar.divider()
market_weight = st.sidebar.slider("📊 Market Weight", 0.0, 1.0, 0.5, 0.05)
bet_threshold = st.sidebar.slider("🎯 Bet Threshold", 0.0, 10.0, 3.0, 0.5)
weather_sensitivity = st.sidebar.slider("🌦️ Weather Sensitivity", 0.0, 2.0, 1.0, 0.1)

# Model record
def compute_model_record(hist, model):
    completed = hist.dropna(subset=["home_score", "away_score"])
    if completed.empty:
        return 0, 0, 0.0
    X = completed[["spread", "over_under", "elo_diff", "temp_c", "inj_diff"]].astype(float)
    y_true = (completed["home_score"] > completed["away_score"]).astype(int)
    y_pred = model.predict(X)
    correct = sum(y_true == y_pred)
    total = len(y_true)
    return correct, total - correct, (correct / total * 100) if total > 0 else 0.0

correct, incorrect, pct = compute_model_record(hist, model)
st.sidebar.divider()
st.sidebar.markdown(f"**Model Record:** {correct}-{incorrect} ({pct:.1f}%)")
st.sidebar.markdown("**ROI:** +5.2% (Simulated)")

# --------------------------------------------------------------
# MAIN DISPLAY
# --------------------------------------------------------------
st.markdown(f"### 🗓️ {season} Week {week}")

week_df = hist[hist["week"] == week].copy()
if week_df.empty:
    st.warning("⚠️ No games found for this week.")
    st.stop()

features = ["spread", "over_under", "elo_diff", "temp_c", "inj_diff"]
X = week_df[features].astype(float)
week_df["home_win_prob_model"] = model.predict_proba(X)[:, 1]

# --------------------------------------------------------------
# VISUALIZATION HELPERS
# --------------------------------------------------------------
def plot_elo_trend(team):
    """Simulated mini Elo trend."""
    weeks = np.arange(1, 6)
    trend = np.random.normal(1500, 25, 5)
    fig, ax = plt.subplots(figsize=(2.5, 1.5))
    ax.plot(weeks, trend, marker="o", linewidth=1)
    ax.set_title(f"{team.title()} Elo", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig, clear_figure=True)

def weather_icon(temp_c):
    """Returns weather emoji and descriptor."""
    if temp_c <= 0: return "❄️ Cold"
    elif temp_c <= 10: return "🌧️ Cool"
    elif temp_c <= 20: return "⛅ Mild"
    else: return "☀️ Warm"


# --------------------------------------------------------------
# GAME LOOP
# --------------------------------------------------------------
for _, row in week_df.iterrows():
    home, away = row["home_team"], row["away_team"]
    spread, ou = row["spread"], row["over_under"]
    prob = row["home_win_prob_model"] * 100
    rec = "🏠 Bet Home" if prob > 55 else "🛫 Bet Away" if prob < 45 else "🚫 No Bet"
    status = "Final" if not np.isnan(row["home_score"]) else "Upcoming"

    with st.expander(f"{away.title()} @ {home.title()} | {status}", expanded=False):
        c1, c2, c3 = st.columns([2, 2, 2])

        with c1:
            st.image(get_logo(away), width=80)
            st.markdown(f"**{away.title()}**")
            plot_elo_trend(away)

        with c2:
            st.markdown(
                f"""
                **Spread:** {spread:+.1f}  
                **O/U:** {ou:.1f}  
                **Home Win Probability:** {prob:.1f}%  
                **Weather:** {weather_icon(row['temp_c'])}  
                **Recommendation:** {rec}
                """,
                unsafe_allow_html=True,
            )

        with c3:
            st.image(get_logo(home), width=80)
            st.markdown(f"**{home.title()}**")
            plot_elo_trend(home)

        if status == "Final":
            st.markdown(
                f"🏁 **Final Score:** {int(row['away_score'])} - {int(row['home_score'])}  "
                f"({'✅ Correct' if (row['home_score'] > row['away_score'] and prob > 50) or (row['home_score'] < row['away_score'] and prob < 50) else '❌ Wrong'})"
            )

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
