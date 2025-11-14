# team_logo_map.py
<<<<<<< HEAD
# Canonical mapping (keeps consistent with public/logos/*)
def canonical_team_name(name: str) -> str:
    if not name:
        return ""
    n = str(name).lower().strip()
    # Mapping dictionary (abbreviated; robust fallback uses underscore)
    MAP = {
        "bears": "chicago_bears", "chicago": "chicago_bears", "chi": "chicago_bears",
        "packers": "green_bay_packers", "green bay": "green_bay_packers", "gb": "green_bay_packers",
        "lions": "detroit_lions", "detroit": "detroit_lions", "det": "detroit_lions",
        "vikings": "minnesota_vikings", "minnesota": "minnesota_vikings", "min": "minnesota_vikings",

        "cowboys": "dallas_cowboys", "dallas": "dallas_cowboys", "dal": "dallas_cowboys",
        "eagles": "philadelphia_eagles", "philadelphia": "philadelphia_eagles", "phi": "philadelphia_eagles",
        "giants": "new_york_giants", "new york giants": "new_york_giants", "nyg": "new_york_giants",
        "commanders": "washington_commanders", "washington": "washington_commanders", "wsh": "washington_commanders",

        "tampa": "tampa_bay_buccaneers", "buccaneers": "tampa_bay_buccaneers", "bucs": "tampa_bay_buccaneers",
        "saints": "new_orleans_saints", "new orleans": "new_orleans_saints",

        "49ers": "san_francisco_49ers", "san francisco": "san_francisco_49ers", "niners": "san_francisco_49ers",
        "seahawks": "seattle_seahawks", "seattle": "seattle_seahawks",
        "rams": "los_angeles_rams", "los angeles rams": "los_angeles_rams", "lar": "los_angeles_rams",

        "cardinals": "arizona_cardinals", "arizona": "arizona_cardinals",
        "falcons": "atlanta_falcons", "atlanta": "atlanta_falcons",

        "ravens": "baltimore_ravens", "baltimore": "baltimore_ravens",
        "bengals": "cincinnati_bengals", "cincinnati": "cincinnati_bengals",
        "steelers": "pittsburgh_steelers", "pittsburgh": "pittsburgh_steelers",
        "browns": "cleveland_browns", "cleveland": "cleveland_browns",

        "bills": "buffalo_bills", "buffalo": "buffalo_bills",
        "dolphins": "miami_dolphins", "miami": "miami_dolphins",
        "patriots": "new_england_patriots", "new england": "new_england_patriots",
        "jets": "new_york_jets", "new york jets": "new_york_jets",

        "colts": "indianapolis_colts", "indianapolis": "indianapolis_colts",
        "jaguars": "jacksonville_jaguars", "jacksonville": "jacksonville_jaguars",
        "texans": "houston_texans", "houston": "houston_texans",
        "titans": "tennessee_titans", "tennessee": "tennessee_titans",

        "chiefs": "kansas_city_chiefs", "kansas city": "kansas_city_chiefs", "kc": "kansas_city_chiefs",
        "chargers": "los_angeles_chargers", "los angeles chargers": "los_angeles_chargers", "lac": "los_angeles_chargers",
        "broncos": "denver_broncos", "denver": "denver_broncos",
        "raiders": "las_vegas_raiders", "las vegas": "las_vegas_raiders",
    }

    if n in MAP:
        return MAP[n]
    # try to normalize common forms, remove punctuation
    n2 = n.replace(".", "").replace("'", "").replace("-", " ").strip()
    if n2 in MAP:
        return MAP[n2]
    # last fallback: convert spaces to underscores
    return n2.replace(" ", "_")
=======

# ---------------------------------------------------------
# Canonical team name normalizer
# Ensures anything (ESPN, Covers, OddsAPI, user input)
# → becomes "city_team" format matching your /logos folder.
#
# Example:
#   "Bears"          → chicago_bears
#   "Chicago Bears"  → chicago_bears
#   "CHI"            → chicago_bears
#
# ---------------------------------------------------------

TEAM_CANONICAL = {
    # --------------------
    # NFC NORTH
    # --------------------
    "chicago bears": "chicago_bears",
    "chicago": "chicago_bears",
    "bears": "chicago_bears",
    "chi": "chicago_bears",

    "green bay packers": "green_bay_packers",
    "green bay": "green_bay_packers",
    "packers": "green_bay_packers",
    "gb": "green_bay_packers",

    "detroit lions": "detroit_lions",
    "detroit": "detroit_lions",
    "lions": "detroit_lions",
    "det": "detroit_lions",

    "minnesota vikings": "minnesota_vikings",
    "minnesota": "minnesota_vikings",
    "vikings": "minnesota_vikings",
    "min": "minnesota_vikings",

    # --------------------
    # NFC EAST
    # --------------------
    "dallas cowboys": "dallas_cowboys",
    "dallas": "dallas_cowboys",
    "cowboys": "dallas_cowboys",
    "dal": "dallas_cowboys",

    "philadelphia eagles": "philadelphia_eagles",
    "philadelphia": "philadelphia_eagles",
    "eagles": "philadelphia_eagles",
    "phi": "philadelphia_eagles",

    "washington commanders": "washington_commanders",
    "washington": "washington_commanders",
    "commanders": "washington_commanders",
    "wsH": "washington_commanders",
    "football team": "washington_commanders",
    "redskins": "washington_commanders",

    "new york giants": "new_york_giants",
    "ny giants": "new_york_giants",
    "giants": "new_york_giants",
    "nyg": "new_york_giants",

    # --------------------
    # NFC SOUTH
    # --------------------
    "tampa bay buccaneers": "tampa_bay_buccaneers",
    "tampa bay": "tampa_bay_buccaneers",
    "buccaneers": "tampa_bay_buccaneers",
    "bucs": "tampa_bay_buccaneers",
    "tb": "tampa_bay_buccaneers",

    "new orleans saints": "new_orleans_saints",
    "new orleans": "new_orleans_saints",
    "saints": "new_orleans_saints",
    "no": "new_orleans_saints",

    "carolina panthers": "carolina_panthers",
    "carolina": "carolina_panthers",
    "panthers": "carolina_panthers",
    "car": "carolina_panthers",

    "atlanta falcons": "atlanta_falcons",
    "atlanta": "atlanta_falcons",
    "falcons": "atlanta_falcons",
    "atl": "atlanta_falcons",

    # --------------------
    # NFC WEST
    # --------------------
    "san francisco 49ers": "san_francisco_49ers",
    "49ers": "san_francisco_49ers",
    "niners": "san_francisco_49ers",
    "sf": "san_francisco_49ers",

    "los angeles rams": "los_angeles_rams",
    "la rams": "los_angeles_rams",
    "rams": "los_angeles_rams",
    "lar": "los_angeles_rams",

    "seattle seahawks": "seattle_seahawks",
    "seattle": "seattle_seahawks",
    "seahawks": "seattle_seahawks",
    "sea": "seattle_seahawks",

    "arizona cardinals": "arizona_cardinals",
    "arizona": "arizona_cardinals",
    "cardinals": "arizona_cardinals",
    "cards": "arizona_cardinals",
    "ari": "arizona_cardinals",

    # --------------------
    # AFC NORTH
    # --------------------
    "baltimore ravens": "baltimore_ravens",
    "baltimore": "baltimore_ravens",
    "ravens": "baltimore_ravens",
    "bal": "baltimore_ravens",

    "cincinnati bengals": "cincinnati_bengals",
    "cincinnati": "cincinnati_bengals",
    "bengals": "cincinnati_bengals",
    "cin": "cincinnati_bengals",

    "pittsburgh steelers": "pittsburgh_steelers",
    "pittsburgh": "pittsburgh_steelers",
    "steelers": "pittsburgh_steelers",
    "pit": "pittsburgh_steelers",

    "cleveland browns": "cleveland_browns",
    "cleveland": "cleveland_browns",
    "browns": "cleveland_browns",
    "cle": "cleveland_browns",

    # --------------------
    # AFC EAST
    # --------------------
    "buffalo bills": "buffalo_bills",
    "buffalo": "buffalo_bills",
    "bills": "buffalo_bills",
    "buf": "buffalo_bills",

    "miami dolphins": "miami_dolphins",
    "miami": "miami_dolphins",
    "dolphins": "miami_dolphins",
    "mia": "miami_dolphins",

    "new england patriots": "new_england_patriots",
    "new england": "new_england_patriots",
    "patriots": "new_england_patriots",
    "pats": "new_england_patriots",
    "ne": "new_england_patriots",

    "new york jets": "new_york_jets",
    "ny jets": "new_york_jets",
    "jets": "new_york_jets",
    "nyj": "new_york_jets",

    # --------------------
    # AFC SOUTH
    # --------------------
    "indianapolis colts": "indianapolis_colts",
    "indianapolis": "indianapolis_colts",
    "colts": "indianapolis_colts",
    "ind": "indianapolis_colts",

    "jacksonville jaguars": "jacksonville_jaguars",
    "jacksonville": "jacksonville_jaguars",
    "jaguars": "jacksonville_jaguars",
    "jags": "jacksonville_jaguars",
    "jax": "jacksonville_jaguars",

    "houston texans": "houston_texans",
    "houston": "houston_texans",
    "texans": "houston_texans",
    "hou": "houston_texans",

    "tennessee titans": "tennessee_titans",
    "tennessee": "tennessee_titans",
    "titans": "tennessee_titans",
    "ten": "tennessee_titans",

    # --------------------
    # AFC WEST
    # --------------------
    "kansas city chiefs": "kansas_city_chiefs",
    "kansas city": "kansas_city_chiefs",
    "chiefs": "kansas_city_chiefs",
    "kc": "kansas_city_chiefs",

    "los angeles chargers": "los_angeles_chargers",
    "la chargers": "los_angeles_chargers",
    "chargers": "los_angeles_chargers",
    "lac": "los_angeles_chargers",

    "denver broncos": "denver_broncos",
    "denver": "denver_broncos",
    "broncos": "denver_broncos",
    "den": "denver_broncos",

    "las vegas raiders": "las_vegas_raiders",
    "vegas": "las_vegas_raiders",
    "raiders": "las_vegas_raiders",
    "lv": "las_vegas_raiders",
    "oakland raiders": "las_vegas_raiders",
    "oak": "las_vegas_raiders",
}

def canonical_team_name(name: str) -> str:
    """
    Convert ESPN/Covers/OddsAPI/team variations → canonical name used in /logos.
    """
    if not name:
        return ""

    n = str(name).lower().strip()

    # direct hit
    if n in TEAM_CANONICAL:
        return TEAM_CANONICAL[n]

    # Try removing plural "s"
    if n.endswith("s") and n[:-1] in TEAM_CANONICAL:
        return TEAM_CANONICAL[n[:-1]]

    # last fallback
    return n.replace(" ", "_")

# ---------------------------------------------------------
# Backwards-compatibility wrappers so older code continues to work
# ---------------------------------------------------------

def canonical_from_string(name: str) -> str:
    """Alias used by older modules. Redirect → canonical_team_name."""
    return canonical_team_name(name)


def canonical_name_for_display(name: str) -> str:
    """UI-friendly wrapper for display normalization."""
    return canonical_team_name(name)
>>>>>>> 83e4cd8c405192af3350849f65cd9e3058c42b44
