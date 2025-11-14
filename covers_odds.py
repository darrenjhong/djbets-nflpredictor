# covers_odds.py
import time
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd

REQUEST_DELAY = 0.6
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DJBetsBot/1.0)"}

def _parse_row_text_for_numbers(text: str):
    nums = re.findall(r"([+-]?\d+(?:\.\d+)?)", text)
    parsed = {"spread": None, "over_under": None}
    for n in nums:
        try:
            v = float(n)
            if abs(v) > 20:
                parsed["over_under"] = v
            else:
                if parsed["spread"] is None:
                    parsed["spread"] = v
        except:
            continue
    return parsed

def parse_covers_game_row(tr):
    try:
        # try to find team names
        matchup = tr.find(class_="cmg_teamName") if tr else None
        text = tr.get_text(" ", strip=True) if tr else ""
        teams = []
        if matchup:
            spans = matchup.find_all("span")
            for s in spans:
                t = s.get_text(strip=True)
                if t:
                    teams.append(t)
        # fallback: parse by " at " or " vs "
        if len(teams) < 2:
            parts = text.split(" at ")
            if len(parts) == 2:
                teams = [parts[0].strip(), parts[1].strip()]
            else:
                parts = text.split(" vs ")
                if len(parts) == 2:
                    teams = [parts[0].strip(), parts[1].strip()]
        if len(teams) < 2:
            return None
        away = teams[0]
        home = teams[1]
        parsed = _parse_row_text_for_numbers(text)
        return {"home": home, "away": away, "spread": parsed["spread"], "over_under": parsed["over_under"]}
    except Exception:
        return None

def fetch_covers_for_week(year: int, week: int) -> pd.DataFrame:
    urls = [
        f"https://www.covers.com/sports/nfl/matchups?selectedDate={year}-W{week}",
        "https://www.covers.com/sports/nfl/matchups",
    ]
    rows = []
    for url in urls:
        try:
            time.sleep(REQUEST_DELAY)
            r = requests.get(url, headers=HEADERS, timeout=8)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            # covers often uses table rows
            trs = soup.select("table.cmg_matchup_table tbody tr")
            if not trs:
                trs = soup.select("div.cmg_matchup_game_box")
            for tr in trs:
                parsed = parse_covers_game_row(tr)
                if parsed:
                    rows.append(parsed)
            if rows:
                break
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
