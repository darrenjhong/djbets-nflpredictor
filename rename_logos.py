# rename_logos.py
"""
Automatically rename your existing logo files to canonical team names:
   "Chicago Bears" -> "chicago_bears.png"
   "Kansas City Chiefs" -> "kansas_city_chiefs.png"
Works with .png or .jpg.

Place this in your repo root and run:
    python rename_logos.py
"""

import os
import re
from difflib import get_close_matches
from team_logo_map import TEAM_CANONICAL

LOGO_DIR = "public/logos"

def canonical_filename(team):
    """Convert canonical team name → canonical filename."""
    return team.lower().replace(" ", "_").replace(".", "").replace("'", "") + ".png"

def simplify(s):
    """Lowercase alphanumerics for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", s.lower())

def guess_team_from_filename(fname):
    base = os.path.splitext(fname)[0]
    simp = simplify(base)

    # fuzzy match against all simplifed canonical keys
    possible = []
    for k, v in TEAM_CANONICAL.items():
        if simplify(k) in simp:
            possible.append(v)
        if simplify(v) in simp:
            possible.append(v)

    # If multiple matches or none, try best fuzzy match over team names
    if not possible:
        team_list = list(set(TEAM_CANONICAL.values()))
        match = get_close_matches(base.replace("_", " ").lower(), team_list, n=1, cutoff=0.4)
        if match:
            possible.append(match[0])

    return possible[0] if possible else None


def main():
    if not os.path.exists(LOGO_DIR):
        print(f"Logo directory not found: {LOGO_DIR}")
        return

    files = [f for f in os.listdir(LOGO_DIR) if f.lower().endswith((".png", ".jpg"))]

    if not files:
        print("No logo files found in public/logos/")
        return

    rename_map = []

    print("🔍 Scanning files...\n")
    for f in files:
        team = guess_team_from_filename(f)
        if not team:
            print(f"⚠️  Could not identify team file: {f}")
            continue

        new_name = canonical_filename(team)
        if f != new_name:
            rename_map.append((f, new_name))

    if not rename_map:
        print("All files already appear canonical — nothing to rename.")
        return

    print("\n📋 Rename plan:")
    for old, new in rename_map:
        print(f"  {old}  →  {new}")

    confirm = input("\nProceed with renaming? (y/n): ").strip().lower()
    if confirm != "y":
        print("❌ Aborted.")
        return

    # Perform renaming
    for old, new in rename_map:
        old_path = os.path.join(LOGO_DIR, old)
        new_path = os.path.join(LOGO_DIR, new)
        os.rename(old_path, new_path)

    print("\n✅ Done! Your logos are now canonical.\n")
    print("You can now run the app and all logos should match automatically.")


if __name__ == "__main__":
    main()