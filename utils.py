# utils.py
# Shared safe utilities for HTTP requests, cleaning, numerical conversions

import requests
import json
import time
import traceback
import streamlit as st


# -------------------------------------------------------------------
# Safe JSON request (returns None instead of crashing)
# -------------------------------------------------------------------
def safe_request_json(url, params=None, headers=None, retries=2, timeout=6):
    """
    Makes a safe HTTP GET request and returns parsed JSON or None on failure.
    """
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            else:
                st.write(f"⚠️ Request error {r.status_code}: {url}")
        except Exception as e:
            st.write(f"⚠️ Exception during request: {e}")
            traceback.print_exc()
        time.sleep(0.3)

    return None


# -------------------------------------------------------------------
# Convert American odds to implied probability
# -------------------------------------------------------------------
def american_to_prob(odds):
    """
    Convert American odds (+110, -145, etc.) to implied probability.
    Returns None if conversion fails.
    """
    try:
        odds = float(odds)
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    except:
        return None


# -------------------------------------------------------------------
# Compute expected margin or rating difference
# -------------------------------------------------------------------
def compute_edge(model_prob, market_prob):
    """
    Returns edge in percentage points.
    If either input missing, returns None.
    """
    try:
        if model_prob is None or market_prob is None:
            return None
        return (model_prob - market_prob) * 100
    except:
        return None


# -------------------------------------------------------------------
# Normalize team names across ESPN / Covers / internal archive
# -------------------------------------------------------------------
def normalize_team(name: str):
    if not isinstance(name, str):
        return None
    return name.lower().replace(".", "").replace("'", "").strip()