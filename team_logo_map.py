# team_logo_map.py
# Mapping from canonical team keys (used in schedule files) to logo filenames
# Put your PNGs in public/logos/ (e.g., public/logos/bears.png)

TEAM_LOGO = {
    "ARI": "cardinals.png",
    "ATL": "falcons.png",
    "BAL": "ravens.png",
    "BUF": "bills.png",
    "CAR": "panthers.png",
    "CHI": "bears.png",
    "CIN": "bengals.png",
    "CLE": "browns.png",
    "DAL": "cowboys.png",
    "DEN": "broncos.png",
    "DET": "lions.png",
    "GB":  "packers.png",
    "HOU": "texans.png",
    "IND": "colts.png",
    "JAX": "jaguars.png",
    "KC":  "chiefs.png",
    "LV":  "raiders.png",
    "LAC": "chargers.png",
    "LAR": "rams.png",
    "MIA": "dolphins.png",
    "MIN": "vikings.png",
    "NE":  "patriots.png",
    "NO":  "saints.png",
    "NYG": "giants.png",
    "NYJ": "jets.png",
    "PHI": "eagles.png",
    "PIT": "steelers.png",
    "SEA": "seahawks.png",
    "SF":  "49ers.png",
    "TB":  "buccaneers.png",
    "TEN": "titans.png",
    "WAS": "commanders.png",
    # add aliases/long names if you need them (e.g., "bears": "bears.png")
}

def lookup_logo(team_key: str) -> str:
    """
    Return relative path to logo file in public/logos/ for the given team code or name.
    """
    if not team_key:
        return None
    key = team_key.strip().upper()
    # if they provided full name like "bears", try lower-case match
    if key in TEAM_LOGO:
        return f"public/logos/{TEAM_LOGO[key]}"
    lower = team_key.strip().lower()
    for k,v in TEAM_LOGO.items():
        if v.startswith(lower) or k.lower()==lower:
            return f"public/logos/{v}"
    # fallback: try <lower>.png
    return f"public/logos/{lower}.png"