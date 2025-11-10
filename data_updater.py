# data_updater.py
import pandas as pd
from datetime import datetime
from pathlib import Path
from data_fetcher import fetch_all_history

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DATA_PATH = DATA_DIR / "historical_odds.csv"

def refresh_data():
    """Fetch new data from SportsOddsHistory and overwrite the local CSV."""
    print("🔄 Fetching updated NFL data...")
    df = fetch_all_history()
    if not df.empty:
        df.to_csv(DATA_PATH, index=False)
        print(f"✅ Data updated successfully with {len(df)} records.")
    else:
        print("⚠️ No new data fetched.")
    return df

def check_last_update():
    """Return when data was last updated."""
    if DATA_PATH.exists():
        mtime = datetime.fromtimestamp(DATA_PATH.stat().st_mtime)
        return mtime.strftime("%Y-%m-%d %H:%M:%S")
    return "Never"
