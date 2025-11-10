# market_baseline.py — market spread logic + blending

import numpy as np
import re

def spread_to_home_prob(spread):
    """
    Convert a point spread (e.g. '-3.5') into implied home win probability.
    Returns np.nan if parsing fails.
    """
    if spread is None or spread == "":
        return np.nan
    try:
        s = str(spread).strip()
        if "even" in s.lower() or "pick" in s.lower():
            return 0.5
        m = re.search(r"([-+−]?\d+\.?\d*)", s)
        if not m:
            return np.nan
        num = float(m.group(1).replace("−", "-"))
    except Exception:
        return np.nan

    # logistic conversion; more negative spread = higher home win chance
    prob = 1 / (1 + np.exp(-num / 7.5))
    return float(np.clip(prob, 0.05, 0.95))

def blend_probs(model_prob, market_prob, alpha=0.6):
    """Weighted blend of model and market probabilities."""
    if np.isnan(model_prob) or np.isnan(market_prob):
        return np.nan
    return float(alpha * market_prob + (1 - alpha) * model_prob)
