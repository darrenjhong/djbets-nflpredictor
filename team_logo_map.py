# team_logo_map.py
# Unified mapping for team names → logo filenames

import os

# Normalization helper
def normalize_team(name: str):
    if not isinstance(name, str):
        return None
    return (
        name.lower()
        .replace(".", "")
        .replace("’", "")
        .replace("'", "")
        .replace("the ", "")
        .strip()
    )

# Master lookup table for all NFL teams
TEAM_NAME_MAP = {
    "49ers": "49ers",
    "san francisco 49ers": "49ers",

    "bears": "bears",
    "chicago bears": "bears",

    "bengals": "bengals",
    "cincinnati bengals": "bengals",

    "bills": "bills",
    "buffalo bills": "bills",

    "broncos": "broncos",
    "denver broncos": "broncos",

    "browns": "browns",
    "cleveland browns": "browns",

    "buccaneers": "buccaneers",
    "tampa bay buccaneers": "buccaneers",

    "cardinals": "cardinals",
    "arizona cardinals": "cardinals",

    "chargers": "chargers",
    "los angeles chargers": "chargers",

    "chiefs": "chiefs",
    "kansas city chiefs": "chiefs",

    "colts": "colts",
    "indianapolis colts": "colts",

    "commanders": "commanders",
    "washington commanders": "commanders",

    "cowboys": "cowboys",
    "dallas cowboys": "cowboys",

    "dolphins": "dolphins",
    "miami dolphins": "dolphins",

    "eagles": "eagles",
    "philadelphia eagles": "eagles",

    "falcons": "falcons",
    "atlanta falcons": "falcons",

    "giants": "giants",
    "new york giants": "giants",

    "jaguars": "jaguars",
    "jacksonville jaguars": "jaguars",

    "jets": "jets",
    "new york jets": "jets",

    "lions": "lions",
    "detroit lions": "lions",

    "packers": "packers",
    "green bay packers": "packers",

    "panthers": "panthers",
    "carolina panthers": "panthers",

    "patriots": "patriots",
    "new england patriots": "patriots",

    "raiders": "raiders",
    "las vegas raiders": "raiders",

    "rams": "rams",
    "los angeles rams": "rams",

    "ravens": "ravens",
    "baltimore ravens": "ravens",

    "saints": "saints",
    "new orleans saints": "saints",

    "seahawks": "seahawks",
    "seattle seahawks": "seahawks",

    "steelers": "steelers",
    "pittsburgh steelers": "steelers",

    "texans": "texans",
    "houston texans": "texans",

    "titans": "titans",
    "tennessee titans": "titans",

    "vikings": "vikings",
    "minnesota vikings": "vikings",
}

LOGO_DIR = "public/logos"

def lookup_logo(team_name: str):
    """
    Returns the path to a local logo file based on the normalized team name.
    Falls back to a placeholder if logo not found.
    """
    if not team_name:
        return None

    key = normalize_team(team_name)
    if key in TEAM_NAME_MAP:
        filename = TEAM_NAME_MAP[key] + ".png"
        full_path = os.path.join(LOGO_DIR, filename)
        if os.path.exists(full_path):
            return full_path

    return None  # let the caller display no image