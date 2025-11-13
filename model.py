# model.py
import numpy as np
import pandas as pd
from typing import Tuple, List
import os
import joblib

MODEL_PATH = "data/model.joblib"

def _ensure_features(df: pd.DataFrame, required: List[str]) -> pd.DataFrame:
    for c in required:
        if c not in df.columns:
            df[c] = 0.0
    return df

def train_model(historical: pd.DataFrame, feature_cols=None, target_col="home_win"):
    """
    Train a simple classifier. historical should have columns: home_score, away_score to label wins.
    feature_cols: list of features to use (strings). If missing, defaults will be used.
    Returns trained model and list of feature columns.
    """
    if feature_cols is None:
        feature_cols = ["elo_diff","inj_diff","temp_c","spread","over_under"]

    # label y
    hist = historical.copy()
    if "home_score" in hist.columns and "away_score" in hist.columns:
        hist[target_col] = (hist["home_score"] > hist["away_score"]).astype(int)
    else:
        # fallback: cannot train; return None
        return None, feature_cols

    hist = hist.dropna(subset=feature_cols + [target_col])
    if hist.shape[0] < 50:
        # not enough training rows; fallback None
        return None, feature_cols

    X = hist[feature_cols].astype(float).fillna(0.0)
    y = hist[target_col].astype(int)

    # try xgboost
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", verbosity=0)
        model.fit(X, y)
    except Exception:
        # fallback to logistic regression
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

    # persist model
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump((model, feature_cols), MODEL_PATH)
    except Exception:
        pass

    return model, feature_cols

def load_model():
    import joblib
    if os.path.exists(MODEL_PATH):
        try:
            model, feature_cols = joblib.load(MODEL_PATH)
            return model, feature_cols
        except Exception:
            return None, None
    return None, None

def predict(model, feature_cols, df: pd.DataFrame):
    df2 = df.copy()
    df2 = df2.reindex(columns=feature_cols).astype(float).fillna(0.0)
    probs = model.predict_proba(df2)[:,1]
    return probs