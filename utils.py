import os
import json
import requests

# ---------------------------------------------
# Safe JSON fetch (used for ESPN scoreboard)
# ---------------------------------------------
def safe_request_json(url: str, timeout: float = 10.0):
    """
    Fetch JSON safely. Returns {} instead of throwing.
    """
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


# ---------------------------------------------
# Team logo path resolution
# ---------------------------------------------
LOGO_DIR = os.path.join("public", "logos")

def get_logo_path(team: str):
    """
    Returns the path to the team's logo based on canonical name.
    Example: "san_francisco_49ers" -> "public/logos/san_francisco_49ers.png"
    """
    if not isinstance(team, str):
        return None

    fname = f"{team}.png"
    path = os.path.join(LOGO_DIR, fname)

    # If logo exists, return path
    if os.path.isfile(path):
        return path

    # fallback icon
    return "https://img.icons8.com/ios-filled/100/question-mark.png"