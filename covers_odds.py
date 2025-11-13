# covers_odds.py
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time

REQUEST_DELAY = 0.8

def parse_covers_game_row(tr):
    try:
        # covers markup varies. defensive parsing:
        text = tr.get_text(" ", strip=True)
        # attempt to extract two team names (heuristic)
        # prefer elements with class "team-name"
        matchup_td = tr.find("td", class_="matchup") if hasattr(tr, "find") else None
        teams = []
        if matchup_td:
            spans = matchup_td.find_all("span", class_="team-name")
            for s in spans:
                teams.append(s.get_text(strip=True))
        if len(teams) < 2:
            # fallback: naive split of line
            pieces = text.split(" at ")
            if len(pieces) == 2:
                away = pieces[0].strip()
                home = pieces[1].strip()
                teams = [away, home]
        if len(teams) < 2:
            return None
        away, home = teams[0], teams[1]

        # attempt to find spread and total (OU)
        spread = None
        ou = None
        # find numbers in the full row text
        nums = re.findall(r"([+-]?\d+(?:\.\d+)?)", text)
        # heuristic: totals > 20
        for n in nums:
            try:
                v = float(n)
                if abs(v) > 20:
                    ou = v
                else:
                    # prefer first small number as spread
                    if spread is None:
                        spread = v
            except:
                continue

        return {"home": home, "away": away, "spread": spread, "over_under": ou}
    except Exception:
        return None


def fetch_covers_for_week(year: int, week: int):
    urls = [
        f"https://www.covers.com/sports/nfl/matchups?selectedDate={year}-W{week}",
        "https://www.covers.com/sports/nfl/matchups",
    ]
    headers = {"User-Agent": "Mozilla/5.0 (compatible; DJBetsBot/1.0)"}
    rows = []
    for u in urls:
        try:
            time.sleep(REQUEST_DELAY)
            r = requests.get(u, headers=headers, timeout=8)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            # try table rows first
            trs = soup.select("table tbody tr")
            if not trs:
                # alternative: covers uses matchup boxes
                trs = soup.find_all("div", class_="cmg_matchup_game_box")
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