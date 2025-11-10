# trainer.py
# Walk-forward training & evaluation for DJBets NFL Predictor

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

def train_walkforward(df, feature_cols, label_col="home_win", time_cols=("season", "week")):
    """Walk-forward training by week."""
    df = df.sort_values(list(time_cols))
    preds, trues, weeks = [], [], []

    for (s, w) in sorted(df[list(time_cols)].drop_duplicates().itertuples(index=False, name=None)):
        train = df[(df["season"] < s) | ((df["season"] == s) & (df["week"] < w))]
        test  = df[(df["season"] == s) & (df["week"] == w)]
        if len(train) < 100 or len(test) == 0:
            continue

        Xtr, ytr = train[feature_cols].values, train[label_col].values
        Xte, yte = test[feature_cols].values,  test[label_col].values

        base = LogisticRegression(max_iter=250)
        clf = CalibratedClassifierCV(base, cv=5, method="sigmoid")
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]

        preds.append(pd.Series(p, index=test.index))
        trues.append(pd.Series(yte, index=test.index))
        weeks.extend([w] * len(test))

    if not preds:
        raise ValueError("Not enough training data")

    p_all = pd.concat(preds).sort_index()
    y_all = pd.concat(trues).sort_index()

    metrics = {
        "Brier": brier_score_loss(y_all, p_all),
        "LogLoss": log_loss(y_all, p_all),
        "Accuracy": accuracy_score(y_all, (p_all >= 0.5).astype(int))
    }
    return p_all, y_all, metrics
