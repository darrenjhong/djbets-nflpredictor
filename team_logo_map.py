# team_logo_map.py
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
