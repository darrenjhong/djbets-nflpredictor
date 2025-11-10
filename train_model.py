import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = DATA_DIR / "model.json"

FEATURES = ["spread", "over_under", "elo_diff", "temp_c", "inj_diff"]

def train_model():
    hist_file = DATA_DIR / "historical_odds.csv"
    if not hist_file.exists():
        raise FileNotFoundError("⚠️ No historical_odds.csv found. Run data_fetcher.py first.")

    df = pd.read_csv(hist_file)
    df = df.dropna(subset=["spread", "home_score", "away_score"])
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["elo_diff"] = np.random.normal(50, 100, len(df))  # placeholder until we have ELO feed
    df["temp_c"] = np.random.uniform(-5, 25, len(df))
    df["inj_diff"] = np.random.normal(0, 5, len(df))

    X = df[FEATURES].fillna(0)
    y = df["home_win"]
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9
    )
    model.fit(X, y)
    model.save_model(str(MODEL_PATH))
    print(f"✅ Model trained and saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
