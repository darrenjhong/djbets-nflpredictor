# covers_odds.py
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time
from team_logo_map import canonical_team_name

REQUEST_DELAY = 0.8

def parse_covers_game_row(tr):
    try:
        text = tr.get_text(" ", strip=True)

        matchup_td = tr.find("td", class_="matchup") if hasattr(tr, "find") else None
        teams = []
        if matchup_td:
            spans = matchup_td.find_all("span", class_="team-name")
            for s in spans:
                teams.append(s.get_text(strip=True))

        if len(teams) < 2:
            pieces = text.split(" at ")
            if len(pieces) == 2:
                teams = [pieces[0].strip(), pieces[1].strip()]

        if len(teams) < 2:
            return None

        # ⭐ canonical names for logos, ESPN + everywhere
        away = canonical_team_name(teams[0])
        home = canonical_team_name(teams[1])

        # parse spread + OU
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
                continue

        return {"home": home, "away": away, "spread": spread, "over_under": ou}

    except Exception:
        return None


def fetch_covers_for_week(year: int, week: int):
    urls = [
        f"https://www.covers.com/sports/nfl/matchups?selectedDate={year}-W{week}",
        "https://www.covers.com/sports/nfl/matchups",
    ]
    headers = {"User-Agent": "Mozilla/5.0 DJBetsBot/1.0"}
    rows = []

    for u in urls:
        try:
            time.sleep(REQUEST_DELAY)
            r = requests.get(u, headers=headers, timeout=8)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            trs = soup.select("table tbody tr")
            if not trs:
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