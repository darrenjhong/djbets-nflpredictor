# model.py
<<<<<<< HEAD
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

MODEL_PATH = "data/model.pkl"

# Features we expect for training/prediction
DEFAULT_FEATURES = ["elo_diff", "spread", "over_under"]

def prepare_training_df(hist_df: pd.DataFrame):
    # Expect historical with home_team, away_team, home_score, away_score
    if hist_df is None or hist_df.empty:
        return pd.DataFrame(), []
    df = hist_df.copy()
    # crude feature engineering: compute elo diff if present else zero
    if "elo_home" not in df.columns or "elo_away" not in df.columns:
        df["elo_home"] = 1500
        df["elo_away"] = 1500
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    # ensure spread/over_under exist
    for c in ["spread", "over_under"]:
        if c not in df.columns:
            df[c] = np.nan
    # label: home win
    df["home_win"] = (pd.to_numeric(df.get("home_score", 0), errors="coerce") > pd.to_numeric(df.get("away_score", 0), errors="coerce")).astype(int)
    features = ["elo_diff", "spread", "over_under"]
    # keep rows that have non-null label
    df = df.dropna(subset=["home_win"])
    return df, features

def train_or_load_model(hist_df: pd.DataFrame):
    # If a model file exists, load it
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                mdl = pickle.load(f)
            return mdl, DEFAULT_FEATURES
        except Exception:
            pass

    # train if enough labeled historical rows exist
    df, features = prepare_training_df(hist_df)
    if df is None or df.empty or len(df) < 250:
        # fallback: create a trivial model (uses only elo_diff)
        mdl = LogisticRegression()
        # train on simulated small set with elo only
        X = np.random.normal(0, 100, size=(200, 1))
        y = (X[:, 0] > 0).astype(int)
        try:
            mdl.fit(X, y)
            # wrap predictor to expect features list
            class SimpleWrapper:
                def __init__(self, model):
                    self.model = model
                    self.features = ["elo_diff"]
                def predict_proba(self, X):
                    return self.model.predict_proba(X)
            wrapper = SimpleWrapper(mdl)
            # save wrapper minimally
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(wrapper, f)
            return wrapper, ["elo_diff"]
        except Exception:
            return None, ["elo_diff"]

    # proceed with training
    X = df[features].fillna(0).values
    y = df["home_win"].values
    mdl = LogisticRegression(max_iter=200)
    mdl.fit(X, y)
    # save
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(mdl, f)
    except Exception:
        pass
    return mdl, features

def has_trained_model():
    return os.path.exists(MODEL_PATH)

def predict_row(model, Xrow: dict):
    # model must provide predict_proba over features. We support sklearn LogisticRegression or wrapped fallback.
    if model is None:
        return None, None, None
    # prepare feature vector in the expected order; try common feature orders
    if hasattr(model, "features"):
        features = model.features
    else:
        # default features
        features = ["elo_diff", "spread", "over_under"]
    vals = []
    for f in features:
        v = Xrow.get(f, np.nan)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = 0.0
        vals.append(float(v))
    X = np.array(vals).reshape(1, -1)
    # if model expects 1 feature (simple fallback), slice
    try:
        probs = model.predict_proba(X)
        prob_home = float(probs[0][1])
    except Exception:
        # try with first column only
        try:
            probs = model.predict_proba(X[:, :1])
            prob_home = float(probs[0][1])
        except Exception:
            prob_home = None
    # simple score projection: use elo_diff to split 48 points baseline
    pred_home_pts = 24 + (vals[0] / 100.0) * 7 if len(vals) > 0 else 24
    pred_away_pts = 24 - (vals[0] / 100.0) * 7 if len(vals) > 0 else 24
    return prob_home, pred_home_pts, pred_away_pts
=======
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
>>>>>>> 83e4cd8c405192af3350849f65cd9e3058c42b44
