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
    """Scrape SportsOddsHistory for a given NFL season safely."""
    url = BASE_URL.format(season)
    print(f"📡 Fetching {season} from {url}")
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        print(f"⚠️ Failed to fetch {season}: {r.status_code}")
        return pd.DataFrame()

    soup = BeautifulSoup(r.text, "html.parser")
    tables = soup.find_all("table")
    all_games = []

    for table in tables:
        rows = table.find_all("tr")
        headers = [th.get_text(strip=True) for th in rows[0].find_all("th")]
        for row in rows[1:]:
            cols = [c.get_text(strip=True) for c in row.find_all("td")]
            # Ensure enough columns to parse (SportsOddsHistory tables have at least 8)
            if len(cols) < 8:
                continue
            try:
                date = cols[0]
                away_team = cols[1]
                away_score = None
                home_team = None
                home_score = None
                spread = None
                over_under = None

                # Handle the varying column layouts
                if len(cols) >= 8:
                    # Normal pattern
                    away_score = _to_int(cols[2])
                    home_team = cols[4]
                    home_score = _to_int(cols[5])
                    spread = _to_float(cols[6])
                    over_under = _to_float(cols[7])

                all_games.append({
                    "season": season,
                    "date": date,
                    "away_team": away_team,
                    "away_score": away_score,
                    "home_team": home_team,
                    "home_score": home_score,
                    "spread": spread,
                    "over_under": over_under
                })
            except Exception as e:
                print(f"⚠️ Skipping row ({season}): {e}")
                continue

    df = pd.DataFrame(all_games)

    # Ensure all expected columns exist
    for col in ["date", "away_team", "home_team", "spread", "over_under"]:
        if col not in df.columns:
            df[col] = None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["week"] = None
    print(f"✅ Parsed {len(df)} games for {season}")
    return df


def _to_int(val):
    try:
        return int(val)
    except:
        return None


def _to_float(val):
    try:
        return float(val.replace("PK", "0").replace("½", ".5"))
    except:
        return None


def fetch_all_history(start_year=2010, end_year=datetime.now().year):
    """Scrape all seasons and save a clean CSV."""
    all_dfs = []
    for year in range(start_year, end_year + 1):
        df = scrape_season(year)
        if not df.empty:
            all_dfs.append(df)
        time.sleep(1.5)  # polite delay

    if not all_dfs:
        raise RuntimeError("❌ No data parsed from SportsOddsHistory.com")

    hist = pd.concat(all_dfs, ignore_index=True)
    hist.to_csv(DATA_DIR / "historical_odds.csv", index=False)
    print(f"✅ Saved {len(hist)} total games to historical_odds.csv")
    return hist


if __name__ == "__main__":
    fetch_all_history()
