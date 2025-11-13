# covers_odds.py
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from datetime import datetime, timezone
import time

# You may want to throttle calls
REQUEST_DELAY = 0.8

def parse_covers_game_row(tr):
    """
    Parse a row from Covers schedule/odds table. Returns (home_abbr, away_abbr, spread, over_under)
    This is defensive — if covers markup changes this may fail; we return None on failure.
    """
    try:
        tds = tr.find_all("td")
        if not tds:
            return None
        # common layout: date/time | matchup | odds...
        matchup_td = tr.find("td", class_="matchup")
        teams = []
        if matchup_td:
            team_spans = matchup_td.find_all("span", class_="team-name")
            for s in team_spans:
                txt = s.get_text(strip=True)
                teams.append(txt)
        if len(teams) < 2:
            # fallback: try 'a' tags
            links = matchup_td.find_all("a") if matchup_td else []
            for a in links:
                teams.append(a.get_text(strip=True))
        if len(teams) < 2:
            return None
        away, home = teams[0], teams[1]

        # spreads and totals present in tds — search for numbers
        spread = None
        ou = None
        # find any "line" cells
        for td in tds:
            text = td.get_text(" ", strip=True)
            # look for numbers like -3.5 or 47.5
            m_spread = re.search(r"([+-]?\d+(?:\.\d+)?)\s*(?:pts|p|spread)?", text)
            if m_spread:
                val = float(m_spread.group(1))
                # Heuristic: totals tend to be > 20, spreads small
                if abs(val) > 20:
                    ou = val
                else:
                    spread = val
        return {"home": home, "away": away, "spread": spread, "over_under": ou}
    except Exception:
        return None

def fetch_covers_for_week(year:int, week:int):
    """
    Scrape the Covers schedule/odds page for a given week/year. Covers doesn't have a simple API,
    so this is best-effort. We'll attempt multiple common URLs and return DataFrame.
    """
    urls = [
        f"https://www.covers.com/sports/nfl/matchups?selectedDate={year}-W{week}",
        f"https://www.covers.com/sports/nfl/matchups",
    ]
    results = []
    headers = {"User-Agent":"Mozilla/5.0 (compatible; DJBetsBot/1.0)"}
    for u in urls:
        try:
            time.sleep(REQUEST_DELAY)
            r = requests.get(u, headers=headers, timeout=8)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            # find table rows for matchups
            rows = soup.select("table tbody tr")
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
    return pd.DataFrame(results)