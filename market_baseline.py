# market_baseline.py — robust Vegas spread conversion and blending logic

import numpy as np
import re

def spread_to_home_prob(spread):
    """
    Convert a point spread (e.g. 'KC -3.5', 'BUF +2.5', 'Even', 'Pick') into
    the market-implied home win probability.
    Positive spread = home underdog, negative spread = home favorite.
    Returns np.nan if parsing fails.
    """
    if spread is None or (isinstance(spread, float) and np.isnan(spread)):
        return np.nan
    if not isinstance(spread, str):
        spread = str(spread)

    s = spread.strip().lower()

    # Handle cases like "Even", "Pick", "Pick'em"
    if "even" in s or "pick" in s:
        return 0.5

    # Extract numeric spread (may appear after a team name)
    m = re.search(r"([-+−]?\d+\.?\d*)", s)
    if not m:
        return np.nan

    try:
        num = m.group(1).replace("−", "-")  # normalize minus sign
        spread_val = float(num)
    except Exception:
        return np.nan

    # Convert spread to win probability using logistic model approximation
    # Empirical relationship: spread of -3 ≈ 0.60 home win prob
    prob = 1 / (1 + np.exp(-spread_val / 7.5))
    return float(np.clip(prob, 0.05, 0.95))


def blend_probs(model_prob, market_prob, alpha=0.6):
    """
    Blend model and market probabilities with a linear weight.
    alpha = 1.0 → fully market, alpha = 0.0 → fully model
    """
    if np.isnan(model_prob) or np.isnan(market_prob):
        return np.nan
    return float(alpha * market_prob + (1 - alpha) * model_prob)
