# team_logo_map.py
# Minimal canonical mapping and lookup helper used by the app

from typing import Dict

# Minimal canonical mapping: map common display names to canonical filename base
# You said you normalized logos to canonical names already. If they match team full names,
# this mapping helps map ESPN names to canonical ones.
TEAM_NAME_MAP: Dict[str, str] = {
    # lowercased display name -> canonical filename base (without extension)
    "chicago bears": "chicago_bears",
    "bears": "chicago_bears",
    "green bay packers": "green_bay_packers",
    "packers": "green_bay_packers",
    "new england patriots": "new_england_patriots",
    "patriots": "new_england_patriots",
    "new york jets": "new_york_jets",
    "jets": "new_york_jets",
    "kansas city chiefs": "kansas_city_chiefs",
    "chiefs": "kansas_city_chiefs",
    "baltimore ravens": "baltimore_ravens",
    "ravens": "baltimore_ravens",
    # add more mappings as you need — keep keys lowercased
}

# canonical_from_string: attempts to match and produce canonical base name
def canonical_from_string(name: str) -> str:
    if not name:
        return ""
    n = name.strip().lower()
    if n in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[n]
    # fallback: remove punctuation and spaces -> underscore
    out = n.replace(".", "").replace(" ", "_").replace("'", "").replace("’", "")
    return out


def lookup_logo(display_name: str, logos_dir="public/logos") -> str:
    """
    Returns a candidate filename (without extension) that can be used to build a path,
    or empty string if not determinable.
    Does not check file existence (get_logo_path in utils will check).
    """
    if not display_name:
        return ""
    return canonical_from_string(display_name)