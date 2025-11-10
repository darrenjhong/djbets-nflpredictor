import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path
import time

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

BASE_URL = "https://www.sportsoddshistory.com/nfl-game-odds/{}"

def scrape_season(season: int) -> pd.DataFrame:
    """Scrape SportsOddsHistory for a given NFL season."""
    url = BASE_URL.format(season)
    print(f"📡 Fetching {season} from {url}")
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Tables are week-by-week
    tables = soup.find_all("table")
    all_games = []

    for table in tables:
        rows = table.find_all("tr")[1:]  # skip headers
        for row in rows:
            cols = [c.get_text(strip=True) for c in row.find_all("td")]
            if len(cols) < 9:  # skip header/empty rows
                continue

            try:
                date = cols[0]
                away_team = cols[1]
                away_score = int(cols[2]) if cols[2].isdigit() else None
                home_team = cols[4]
                home_score = int(cols[5]) if cols[5].isdigit() else None
                spread = float(cols[6].replace("PK", "0")) if cols[6] not in ["", "PK"] else 0.0
                over_under = float(cols[7]) if cols[7] != "" else None
                week = None
                all_games.append({
                    "season": season,
                    "week": week,
                    "date": date,
                    "away_team": away_team,
                    "home_team": home_team,
                    "away_score": away_score,
                    "home_score": home_score,
                    "spread": spread,
                    "over_under": over_under
                })
            except Exception as e:
                print(f"⚠️ Skipping row: {e}")
                continue

    df = pd.DataFrame(all_games)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def fetch_all_history(start_year=2010, end_year=datetime.now().year):
    """Scrape multiple seasons and save to CSV."""
    all_dfs = []
    for year in range(start_year, end_year + 1):
        df = scrape_season(year)
        if not df.empty:
            all_dfs.append(df)
        time.sleep(1.5)  # be polite to the site
    hist = pd.concat(all_dfs, ignore_index=True)
    hist.to_csv(DATA_DIR / "historical_odds.csv", index=False)
    print(f"✅ Saved {len(hist)} games to historical_odds.csv")
    return hist

if __name__ == "__main__":
    fetch_all_history()
