# team_logo_map.py
# Canonical mapping for NFL teams (City + Mascot). Exports TEAM_CANONICAL and lookup helpers.

import re, os

TEAM_CANONICAL = {
    "arizona": "Arizona Cardinals",
    "cardinals": "Arizona Cardinals",
    "atlanta": "Atlanta Falcons",
    "falcons": "Atlanta Falcons",
    "baltimore": "Baltimore Ravens",
    "ravens": "Baltimore Ravens",
    "buffalo": "Buffalo Bills",
    "bills": "Buffalo Bills",
    "carolina": "Carolina Panthers",
    "panthers": "Carolina Panthers",
    "chicago": "Chicago Bears",
    "bears": "Chicago Bears",
    "cincinnati": "Cincinnati Bengals",
    "bengals": "Cincinnati Bengals",
    "cleveland": "Cleveland Browns",
    "browns": "Cleveland Browns",
    "dallas": "Dallas Cowboys",
    "cowboys": "Dallas Cowboys",
    "denver": "Denver Broncos",
    "broncos": "Denver Broncos",
    "detroit": "Detroit Lions",
    "lions": "Detroit Lions",
    "green bay": "Green Bay Packers",
    "greenbay": "Green Bay Packers",
    "packers": "Green Bay Packers",
    "houston": "Houston Texans",
    "texans": "Houston Texans",
    "indianapolis": "Indianapolis Colts",
    "colts": "Indianapolis Colts",
    "jacksonville": "Jacksonville Jaguars",
    "jaguars": "Jacksonville Jaguars",
    "kansas city": "Kansas City Chiefs",
    "kansascity": "Kansas City Chiefs",
    "chiefs": "Kansas City Chiefs",
    "las vegas": "Las Vegas Raiders",
    "lasvegas": "Las Vegas Raiders",
    "raiders": "Las Vegas Raiders",
    "los angeles chargers": "Los Angeles Chargers",
    "la chargers": "Los Angeles Chargers",
    "chargers": "Los Angeles Chargers",
    "los angeles rams": "Los Angeles Rams",
    "rams": "Los Angeles Rams",
    "miami": "Miami Dolphins",
    "dolphins": "Miami Dolphins",
    "minnesota": "Minnesota Vikings",
    "vikings": "Minnesota Vikings",
    "new england": "New England Patriots",
    "patriots": "New England Patriots",
    "new orleans": "New Orleans Saints",
    "saints": "New Orleans Saints",
    "new york giants": "New York Giants",
    "ny giants": "New York Giants",
    "giants": "New York Giants",
    "new york jets": "New York Jets",
    "ny jets": "New York Jets",
    "jets": "New York Jets",
    "philadelphia": "Philadelphia Eagles",
    "eagles": "Philadelphia Eagles",
    "pittsburgh": "Pittsburgh Steelers",
    "steelers": "Pittsburgh Steelers",
    "san francisco": "San Francisco 49ers",
    "sanfrancisco": "San Francisco 49ers",
    "49ers": "San Francisco 49ers",
    "san francisco 49ers": "San Francisco 49ers",
    "seattle": "Seattle Seahawks",
    "seahawks": "Seattle Seahawks",
    "tampa bay": "Tampa Bay Buccaneers",
    "tampabay": "Tampa Bay Buccaneers",
    "buccaneers": "Tampa Bay Buccaneers",
    "tennessee": "Tennessee Titans",
    "titans": "Tennessee Titans",
    "washington": "Washington Commanders",
    "commanders": "Washington Commanders"
}

def canonical_from_string(s: str):
    """
    Turn an arbitrary team string (espn, covers, etc.) into canonical City + Mascot.
    Best-effort, case-insensitive fuzzy mapping.
    """
    if not s or not isinstance(s, str):
        return None
    t = s.lower().strip()
    t = re.sub(r"[^a-z0-9 ]", "", t)  # remove punctuation
    # direct exact matches for multiword keys
    for k, v in TEAM_CANONICAL.items():
        if k in t:
            return v
    # fallback: try splitting words and match any token
    toks = t.split()
    for tok in toks:
        if tok in TEAM_CANONICAL:
            return TEAM_CANONICAL[tok]
    return None

def lookup_logo(canonical_team: str, logos_dir="public/logos"):
    """
    Given canonical team name, returns path to expected logo file if exists.
    Expected filename: canonical lower -> spaces to "_" -> remove dots -> .png (common)
    Example: "Kansas City Chiefs" -> "public/logos/kansas_city_chiefs.png"
    """
    if not canonical_team:
        return None
    fname = canonical_team.lower().replace(" ", "_").replace(".", "").replace("'", "") + ".png"
    path = os.path.join(logos_dir, fname)
    if os.path.exists(path):
        return path
    # try jpg
    jpg = path[:-4] + ".jpg"
    if os.path.exists(jpg):
        return jpg
    # fallback: return None
    return None