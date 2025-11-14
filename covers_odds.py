# covers_odds.py
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time

REQUEST_DELAY = 0.8

def parse_covers_game_row(tr):
    """
    Parse a single Covers matchup row/box into:
    { home, away, spread, over_under }
    Markup changes often — this uses multiple fallbacks.
    """
    try:
        txt = tr.get_text(" ", strip=True)

        # ---- TEAM NAMES ----
        teams = []

        # Preferred: spans with class team-name
        spans = tr.find_all("span", class_="team-name")
        if spans:
            teams = [s.get_text(strip=True) for s in spans]

        # Fallback: text split (e.g. "NY Giants at Dallas Cowboys")
        if len(teams) < 2:
            if " at " in txt.lower():
                parts = txt.split(" at ")
                if len(parts) == 2:
                    away = parts[0].strip()
                    home = parts[1].strip()
                    teams = [away, home]

        if len(teams) < 2:
            return None

        away, home = teams[0], teams[1]

        # ---- SPREAD + OVER/UNDER ----
        spread = None
        ou = None

        # Extract numbers from row text
        nums = re.findall(r"([+-]?\d+(?:\.\d+)?)", txt)

        # Heuristic:
        # spreads ~ < 20
        # totals ~ > 20
        for n in nums:
            try:
                v = float(n)
                if abs(v) > 20:
                    ou = v
                else:
                    if spread is None:
                        spread = v
            except:
                continue

        return {
            "home": home,
            "away": away,
            "spread": spread,
            "over_under": ou
        }

    except Exception:
        return None


def fetch_covers_for_week(year: int, week: int):
    """
    Fetch NFL matchups for a week using Covers.
    Covers has no stable API — we attempt multiple URLs and parse both
    table layouts and matchup-box layouts.
    """
    urls = [
        # Week-specific
        f"https://www.covers.com/sports/nfl/matchups?selectedDate={year}-W{week}",
        # Generic (they often show current/next week's games)
        "https://www.covers.com/sports/nfl/matchups",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; DJBetsBot/1.0)"
    }

    results = []

    for url in urls:
        try:
            time.sleep(REQUEST_DELAY)
            r = requests.get(url, headers=headers, timeout=8)
            r.raise_for_status()

            soup = BeautifulSoup(r.text, "html.parser")

            # Try table format
            rows = soup.select("table tbody tr")
            # Try card/box format
            if not rows:
                rows = soup.find_all("div", class_="cmg_matchup_game_box")

            for tr in rows:
                parsed = parse_covers_game_row(tr)
                if parsed:
                    results.append(parsed)

            if results:
                break

        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df