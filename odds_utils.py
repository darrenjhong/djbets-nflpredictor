# odds_utils.py
"""
OddsAPI client wrapper with caching.
- Reads ODDS_API_KEY from environment or ./data/odds_api_key.txt
- Caches responses to cache_path (JSON)
- Safe: if key lacks historical access, it will return empty payloads for past games without crashing
"""

import os
import json
from pathlib import Path
from typing import Optional
import time
import requests

DEFAULT_ODDS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"

class OddsAPIClient:
    def __init__(self, cache_path: Path = Path("cache/odds_cache.json"), max_calls: int = 100):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        # load cache
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}
        else:
            self._cache = {}
        self.max_calls = max_calls
        # key
        self.key = os.getenv("ODDS_API_KEY")
        if not self.key:
            # try local file
            token_file = Path("data/odds_api_key.txt")
            if token_file.exists():
                self.key = token_file.read_text().strip()
        # simple call counter per process
        self._calls = 0

    def _save_cache(self):
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2)

    def get_odds_for_matchup(self, home_team: str, away_team: str, kickoff_ts=None):
        """
        Return a dict with { "spread": float or None, "over_under": float or None, "raw": ... } where possible.
        Only calls API if we have a key and if kickoff is in the future (Option B). Cached responses served when available.
        """
        # canonical key
        key = f"{away_team}__at__{home_team}"
        if key in self._cache:
            return self._cache[key]

        if self._calls >= self.max_calls:
            # reached safe cap
            return {}

        if not self.key:
            # no key -> nothing
            self._cache[key] = {}
            self._save_cache()
            return {}

        # check kickoff: only fetch if kickoff in future or None (option B: skip past games)
        import datetime
        if kickoff_ts is not None:
            try:
                dt = pd.to_datetime(kickoff_ts)
                if dt.tzinfo is None:
                    dt = dt.tz_localize("UTC")
                if dt < datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc):
                    # past game -> do not fetch historical odds
                    self._cache[key] = {}
                    self._save_cache()
                    return {}
            except Exception:
                pass

        # make a best-effort call to OddsAPI events endpoint, filtered by date if possible
        params = {
            "apiKey": self.key,
            "regions": "us",
            "markets": "spreads,totals",
            "oddsFormat": "american",
        }
        # optionally include date
        if kickoff_ts is not None:
            try:
                dtstr = pd.to_datetime(kickoff_ts).strftime("%Y-%m-%d")
                params["dateFormat"] = "iso"
                params["bookmakers"] = "DraftKings"  # optional
            except Exception:
                pass

        try:
            resp = requests.get(DEFAULT_ODDS_URL, params=params, timeout=8)
            self._calls += 1
            if resp.status_code != 200:
                # don't spam errors; return empty and cache
                self._cache[key] = {}
                self._save_cache()
                return {}
            data = resp.json()
            # find matching event by team names
            chosen = None
            for ev in data:
                teams = ev.get("teams", [])
                # simple case-insensitive match
                if home_team and away_team and any(home_team.lower() in t.lower() for t in teams) and any(away_team.lower() in t.lower() for t in teams):
                    chosen = ev
                    break
            if not chosen:
                # save empty
                self._cache[key] = {}
                self._save_cache()
                return {}
            # extract spread & totals from bookmakers/markets if present
            spread = None
            over_under = None
            # choose first bookmaker
            bks = chosen.get("bookmakers", [])
            if bks:
                markets = bks[0].get("markets", [])
                for m in markets:
                    if m.get("key") == "spreads":
                        outcomes = m.get("outcomes", [])
                        # outcomes have names and point values; home spread is where outcome['name']==home_team
                        for o in outcomes:
                            name = o.get("name","")
                            price = o.get("point")
                            if home_team.lower() in name.lower():
                                # point is how many points home is favored by (American convention may vary)
                                spread = float(price) if price is not None else None
                    if m.get("key") == "totals":
                        outcomes = m.get("outcomes", [])
                        # outcomes for totals will have 'point' for total
                        for o in outcomes:
                            over_under = o.get("point") or over_under
            out = {"spread": spread, "over_under": over_under, "raw": chosen}
            self._cache[key] = out
            self._save_cache()
            return out
        except Exception:
            self._cache[key] = {}
            self._save_cache()
            return {}

# if pandas used inside odds_utils
import pandas as pd