# market_baseline.py
# Utilities for converting betting lines to probabilities and blending with model output

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# --- odds conversions ---
def american_to_prob(ml):
    """Convert American moneyline to implied probability."""
    try:
        ml = float(str(ml).replace("−", "-"))
        if ml > 0:
            return 100.0 / (ml + 100.0)
        else:
            return (-ml) / ((-ml) + 100.0)
    except Exception:
        return np.nan


def remove_vig(p_home, p_away):
    """Normalize implied probabilities to sum to 1."""
    s = p_home + p_away
    if s <= 0:
        return np.nan, np.nan
    return p_home / s, p_away / s


def spread_to_home_prob(spread):
    """Approximate home win probability from point spread."""
    try:
        s = float(str(spread).replace("−", "-"))
    except Exception:
        return np.nan
    k = 6.8  # logistic slope for NFL
    return 1.0 / (1.0 + np.exp(-(s) / k))


def blend_probs(model_p, market_p, alpha=0.6):
    """Blend model and market probabilities."""
    if np.isnan(model_p) and np.isnan(market_p):
        return np.nan
    if np.isnan(market_p):
        return model_p
    if np.isnan(model_p):
        return market_p
    return alpha * market_p + (1 - alpha) * model_p
