# covers_odds.py
"""
Lightweight Covers.com scraper (best-effort) that returns a DataFrame with:
columns: home, away, spread, over_under
This is fragile (site may change). We keep it defensive and best-effort.
"""

import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time
import os

REQUEST_DELAY = 0.8
HEADERS = {"User-Agent":"Mozilla/5.0 (compatible; DJBetsBot/1.0)"}

def parse_covers_game_row(tr):
    try:
        text = tr.get_text(" ", strip=True)
        # find teams: look for ' at ' or ' vs ' or splits
        # prefer elements with class team-name
        teams=[]
        if hasattr(tr, "find_all"):
            spans = tr.find_all(class_=re.compile("team"))
            for s in spans:
                t = s.get_text(strip=True)
                if t:
                    teams.append(t)
        if len(teams) < 2:
            # fallback
            if " at " in text:
                parts = text.split(" at ")
                teams = [parts[0].strip(), parts[1].split()[0].strip()]
            elif " vs " in text:
                parts = text.split(" vs ")
                teams = [parts[0].strip(), parts[1].split()[0].strip()]

        if len(teams) < 2:
            return None
        away, home = teams[0], teams[1]

        # numeric heuristics for spread and total
        spread = None
        ou = None
        nums = re.findall(r"([+-]?\d+(?:\.\d+)?)", text)
        for n in nums:
            try:
                v = float(n)
                if abs(v) > 20:
                    ou = v
                else:
                    if spread is None:
                        spread = v
            except:
                pass
        return {"home": home, "away": away, "spread": spread, "over_under": ou}
    except Exception:
        return None

def fetch_covers_for_week(year:int, week:int):
    results=[]
    urls = [
        f"https://www.covers.com/sports/nfl/matchups?selectedDate={year}-W{week}",
        "https://www.covers.com/sports/nfl/matchups"
    ]
    for u in urls:
        try:
            time.sleep(REQUEST_DELAY)
            r = requests.get(u, headers=HEADERS, timeout=8)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            rows = soup.select("table tbody tr")
            if not rows:
                rows = soup.find_all("div", class_="cmg_matchup_game_box")
            for tr in rows:
                p = parse_covers_game_row(tr)
                if p:
                    results.append(p)
            if results:
                break
        except Exception:
            continue
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)