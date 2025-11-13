# covers_odds.py
import requests, time, re
from bs4 import BeautifulSoup
import pandas as pd
from utils import normalize_team_name
import logging

logger = logging.getLogger("djbets.covers")
REQUEST_DELAY = 0.8
HEADERS = {"User-Agent":"Mozilla/5.0 (compatible; DJBetsBot/1.0)"}

def parse_covers_game_row(tr):
    """
    Parse a Covers table row/div block. Best-effort.
    Returns dict {home, away, spread, over_under}
    """
    try:
        # try to find the matchup area
        text = tr.get_text(" ", strip=True)
        # attempt to find team names (heuristic)
        # Common pattern "AwayTeam @ HomeTeam" or "AwayTeam vs HomeTeam"
        m = re.search(r"([A-Za-z .']+)\s+[@vV][sS]?\s+([A-Za-z .']+)", text)
        teams = []
        if m:
            teams = [m.group(1).strip(), m.group(2).strip()]
        else:
            # fallback: collect span.team-name
            spans = tr.select("span.team-name")
            for s in spans:
                teams.append(s.get_text(strip=True))
        if len(teams) < 2:
            return None
        away_raw, home_raw = teams[0], teams[1]
        # find numbers
        nums = re.findall(r"([+-]?\d+(?:\.\d+)?)", text)
        spread = None
        ou = None
        # heuristics: totals > 20 are totals, small numbers are spreads
        for n in nums:
            val = float(n)
            if abs(val) > 20:
                if ou is None:
                    ou = val
            else:
                if spread is None:
                    spread = val
        # normalize team names
        home = normalize_team_name(home_raw) or home_raw
        away = normalize_team_name(away_raw) or away_raw
        return {"home": home, "away": away, "spread": spread, "over_under": ou}
    except Exception as e:
        logger.debug("parse error: %s", e)
        return None

def fetch_covers_for_week(year:int, week:int):
    """
    Try to fetch Covers matchups page and extract odds. Best-effort, non-API.
    """
    urls = [
        "https://www.covers.com/sports/nfl/matchups",
        f"https://www.covers.com/sports/nfl/matchups?selectedDate={year}-W{week}"
    ]
    results = []
    for url in urls:
        try:
            time.sleep(REQUEST_DELAY)
            r = requests.get(url, headers=HEADERS, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            # rows
            rows = soup.select("table.covers-matchups__table tbody tr")
            if not rows:
                rows = soup.select("div.cmg_matchup_game_box")
            for tr in rows:
                parsed = parse_covers_game_row(tr)
                if parsed:
                    results.append(parsed)
            if results:
                break
        except Exception as e:
            logger.debug("covers fetch failed %s", e)
            continue
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)