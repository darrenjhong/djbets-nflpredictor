# team_logo_map.py
"""
Canonical mappings for team names -> logo filenames in public/logos/
Use these mappings to convert incoming ESPN/Covers strings to canonical filenames.
"""

import os

LOGO_DIR = os.path.join("public", "logos")

TEAM_CANONICAL = {
    "bears":"chicago_bears","chicago_bears":"chicago_bears","chicago":"chicago_bears","chi":"chicago_bears",
    "packers":"green_bay_packers","green_bay_packers":"green_bay_packers","green bay":"green_bay_packers","gb":"green_bay_packers",
    "lions":"detroit_lions","detroit_lions":"detroit_lions","detroit":"detroit_lions","det":"detroit_lions",
    "vikings":"minnesota_vikings","minnesota_vikings":"minnesota_vikings","minnesota":"minnesota_vikings","min":"minnesota_vikings",
    "cowboys":"dallas_cowboys","dallas_cowboys":"dallas_cowboys","dallas":"dallas_cowboys","dal":"dallas_cowboys",
    "eagles":"philadelphia_eagles","philadelphia_eagles":"philadelphia_eagles","philadelphia":"philadelphia_eagles","phi":"philadelphia_eagles",
    "giants":"new_york_giants","new_york_giants":"new_york_giants","nyg":"new_york_giants",
    "washington_commanders":"washington_commanders","commanders":"washington_commanders","washington":"washington_commanders",
    "tampa_bay_buccaneers":"tampa_bay_buccaneers","buccaneers":"tampa_bay_buccaneers","bucs":"tampa_bay_buccaneers",
    "saints":"new_orleans_saints","new_orleans_saints":"new_orleans_saints","no":"new_orleans_saints",
    "panthers":"carolina_panthers","carolina_panthers":"carolina_panthers",
    "falcons":"atlanta_falcons","atlanta_falcons":"atlanta_falcons",
    "49ers":"san_francisco_49ers","san_francisco_49ers":"san_francisco_49ers","niners":"san_francisco_49ers",
    "rams":"los_angeles_rams","los_angeles_rams":"los_angeles_rams","lar":"los_angeles_rams",
    "seahawks":"seattle_seahawks","seattle_seahawks":"seattle_seahawks",
    "cardinals":"arizona_cardinals","arizona_cardinals":"arizona_cardinals","ari":"arizona_cardinals",
    "ravens":"baltimore_ravens","baltimore_ravens":"baltimore_ravens","bal":"baltimore_ravens",
    "bengals":"cincinnati_bengals","cincinnati_bengals":"cincinnati_bengals","cin":"cincinnati_bengals",
    "steelers":"pittsburgh_steelers","pittsburgh_steelers":"pittsburgh_steelers","pit":"pittsburgh_steelers",
    "browns":"cleveland_browns","cleveland_browns":"cleveland_browns","cle":"cleveland_browns",
    "bills":"buffalo_bills","buffalo_bills":"buffalo_bills","buf":"buffalo_bills",
    "dolphins":"miami_dolphins","miami_dolphins":"miami_dolphins","mia":"miami_dolphins",
    "patriots":"new_england_patriots","new_england_patriots":"new_england_patriots","ne":"new_england_patriots",
    "jets":"new_york_jets","new_york_jets":"new_york_jets","nyj":"new_york_jets",
    "colts":"indianapolis_colts","indianapolis_colts":"indianapolis_colts","ind":"indianapolis_colts",
    "jaguars":"jacksonville_jaguars","jacksonville_jaguars":"jacksonville_jaguars","jax":"jacksonville_jaguars",
    "texans":"houston_texans","houston_texans":"houston_texans","hou":"houston_texans",
    "titans":"tennessee_titans","tennessee_titans":"tennessee_titans","ten":"tennessee_titans",
    "chiefs":"kansas_city_chiefs","kansas_city_chiefs":"kansas_city_chiefs","kc":"kansas_city_chiefs",
    "chargers":"los_angeles_chargers","los_angeles_chargers":"los_angeles_chargers","lac":"los_angeles_chargers",
    "broncos":"denver_broncos","denver_broncos":"denver_broncos","den":"denver_broncos",
    "raiders":"las_vegas_raiders","las_vegas_raiders":"las_vegas_raiders","lv":"las_vegas_raiders","oak":"las_vegas_raiders",
    "rams":"los_angeles_rams"
}

TEAM_CANONICAL.update({
    "nyg": "new_york_giants",
    "ny giants": "new_york_giants",
    "nyj": "new_york_jets",
    "ny jets": "new_york_jets",
    "la chargers": "los_angeles_chargers",
    "la rams": "los_angeles_rams",
})


def canonical_team_name(name: str) -> str:
    if not name:
        return ""
    n = str(name).lower().strip()

    # Normalize common punctuation from Covers/ESPN
    n = n.replace(".", "")          # "n.y. giants" -> "ny giants"
    n = n.replace("@", "")          # just in case
    n = n.replace("  ", " ")

    # Extra aliases for Covers-style abbreviations
    if n in ("ny giants", "ny gaints"):  # typo-safe
        n = "nyg"
    if n in ("ny jets",):
        n = "nyj"
    if n in ("la chargers",):
        n = "lac"
    if n in ("la rams",):
        n = "lar"

    if n in TEAM_CANONICAL:
        return TEAM_CANONICAL[n]

    # try remove trailing s
    if n.endswith("s") and n[:-1] in TEAM_CANONICAL:
        return TEAM_CANONICAL[n[:-1]]

    return n.replace(" ", "_")


def get_logo_path(canonical_name: str) -> str:
    if not canonical_name:
        return ""
    fname = f"{canonical_name}.png"
    p = os.path.join(LOGO_DIR, fname)
    if os.path.exists(p):
        return p
    # try jpg
    p2 = os.path.join(LOGO_DIR, f"{canonical_name}.jpg")
    if os.path.exists(p2):
        return p2
    return ""