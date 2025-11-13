# team_logo_map.py
# Canonical NFL team names + logo lookup

import os

# canonical names → filename (in /public/logos/)
TEAM_LOGO_FILES = {
    "arizona cardinals": "arizona_cardinals.png",
    "atlanta falcons": "atlanta_falcons.png",
    "baltimore ravens": "baltimore_ravens.png",
    "buffalo bills": "buffalo_bills.png",
    "carolina panthers": "carolina_panthers.png",
    "chicago bears": "chicago_bears.png",
    "cincinnati bengals": "cincinnati_bengals.png",
    "cleveland browns": "cleveland_browns.png",
    "dallas cowboys": "dallas_cowboys.png",
    "denver broncos": "denver_broncos.png",
    "detroit lions": "detroit_lions.png",
    "green bay packers": "green_bay_packers.png",
    "houston texans": "houston_texans.png",
    "indianapolis colts": "indianapolis_colts.png",
    "jacksonville jaguars": "jacksonville_jaguars.png",
    "kansas city chiefs": "kansas_city_chiefs.png",
    "las vegas raiders": "las_vegas_raiders.png",
    "los angeles chargers": "los_angeles_chargers.png",
    "los angeles rams": "los_angeles_rams.png",
    "miami dolphins": "miami_dolphins.png",
    "minnesota vikings": "minnesota_vikings.png",
    "new england patriots": "new_england_patriots.png",
    "new orleans saints": "new_orleans_saints.png",
    "new york giants": "new_york_giants.png",
    "new york jets": "new_york_jets.png",
    "philadelphia eagles": "philadelphia_eagles.png",
    "pittsburgh steelers": "pittsburgh_steelers.png",
    "san francisco 49ers": "san_francisco_49ers.png",
    "seattle seahawks": "seattle_seahawks.png",
    "tampa bay buccaneers": "tampa_bay_buccaneers.png",
    "tennessee titans": "tennessee_titans.png",
    "washington commanders": "washington_commanders.png",
}

# team_logo_map.py
TEAM_CANONICAL = {
    "bears": "chicago_bears",
    "chicago": "chicago_bears",
    "chicago bears": "chicago_bears",

    "packers": "green_bay_packers",
    "green bay": "green_bay_packers",
    "green bay packers": "green_bay_packers",

    "lions": "detroit_lions",
    "detroit": "detroit_lions",
    "detroit lions": "detroit_lions",

    # ... repeat for ALL 32 teams ...
}

def canonical_team_name(name):
    n = str(name).lower().strip()
    return TEAM_CANONICAL.get(n, n.replace(" ", "_"))