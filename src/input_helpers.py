"""
AstroPlanner — Shared CLI Input Helpers

Validated, retry-on-bad-input prompt functions used to collect a
WeeklyPlanRequest from the terminal (Colab cell input()). Kept separate
from models.py so a future frontend can replace this file entirely without
touching the schemas it produces.

v2: adds prompts for the new (optional) Preferences block and the new
(optional) telescope/camera type + mount tracking fields, on top of the
original validation helpers — nothing about the original retry/sanitize
behavior was weakened.
"""

import re
from datetime import date as date_type, timedelta

from models import WeeklyPlanRequest


def prompt_float(label: str, min_val: float = None, max_val: float = None) -> float:
    """Repeatedly prompts until a valid float within optional bounds is entered."""
    while True:
        raw = input(f"{label}: ").strip()
        try:
            value = float(raw)
        except ValueError:
            print("  Invalid number, try again.")
            continue
        if min_val is not None and value < min_val:
            print(f"  Must be greater than {min_val}.")
            continue
        if max_val is not None and value > max_val:
            print(f"  Must be at most {max_val}.")
            continue
        return value


def prompt_choice(label: str, choices: list[str]) -> str:
    """Repeatedly prompts until input matches one of the allowed choices (case-insensitive)."""
    choices_lower = [c.lower() for c in choices]
    while True:
        raw = input(f"{label} [{'/'.join(choices)}]: ").strip().lower()
        if raw in choices_lower:
            return raw
        print(f"  Must be one of: {', '.join(choices)}.")


def prompt_yes_no(label: str) -> bool:
    """Repeatedly prompts until a y/n answer is given."""
    while True:
        raw = input(f"{label} [y/n]: ").strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please enter y or n.")


def prompt_int(label: str, min_val: int = None, max_val: int = None) -> int:
    """Repeatedly prompts until a valid integer within optional bounds is entered."""
    while True:
        raw = input(f"{label}: ").strip()
        try:
            value = int(raw)
        except ValueError:
            print("  Invalid whole number, try again.")
            continue
        if min_val is not None and value < min_val:
            print(f"  Must be at least {min_val}.")
            continue
        if max_val is not None and value > max_val:
            print(f"  Must be at most {max_val}.")
            continue
        return value


# Area type -> baseline Bortle estimate. Ordered brightest-first to match
# how people naturally describe where they live (city vs. suburb vs.
# rural), rather than requiring recognition of specific astronomical
# effects like the original class-by-class ladder did.
AREA_TYPE_BORTLE_MAP = [
    ("Large city downtown / city center", 9),
    ("City or urban neighborhood", 8),
    ("Suburb of a city", 6),
    ("Small town", 5),
    ("Rural area, near a town", 4),
    ("Rural area, far from towns", 3),
    ("Very remote (desert, mountains, designated dark-sky area)", 2),
]


def estimate_bortle_scale_interactive() -> int:
    """
    User-friendly self-assessment for users who don't already know their
    Bortle number: pick the area type that best matches the observing
    site, then answer one plain-language sanity-check question. Chosen
    over a stricter multi-question ladder because most users can
    confidently describe their surroundings ("suburb", "rural") without
    needing to recognize specific astronomical effects (zodiacal light,
    M33 naked-eye visibility, etc.).

    The sanity check exists because area type alone is a rough proxy —
    e.g. a "rural" site near an unlit highway or industrial glow can be
    brighter than the label suggests, and a "suburb" backing onto a dark
    park/coastline can be darker. If the Milky Way answer disagrees with
    what the area type would predict, the estimate is nudged two classes
    toward what was actually reported; otherwise the area-type baseline
    is used as-is.

    NOT a measured value — a deliberately rough estimate for planning
    purposes, not a substitute for an actual SQM reading.
    """
    labels = [label for label, _ in AREA_TYPE_BORTLE_MAP]
    print("  Which best describes where you'll be observing from?")
    for i, label in enumerate(labels, 1):
        print(f"    {i}. {label}")

    while True:
        raw = input("  Enter a number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(labels):
            base_bortle = AREA_TYPE_BORTLE_MAP[int(raw) - 1][1]
            break
        print(f"  Please enter a number from 1 to {len(labels)}.")

    milky_way_visible = prompt_yes_no(
        "  Quick check: on a clear, moonless night, can you see the Milky Way "
        "as a hazy band across the sky with just your eyes"
    )

    if base_bortle >= 6 and milky_way_visible:
        # Labeled as a brighter area, but the sky's actually darker than that.
        return max(1, base_bortle - 2)
    if base_bortle <= 4 and not milky_way_visible:
        # Labeled as a darker area, but the sky's actually brighter than that.
        return min(9, base_bortle + 2)
    return base_bortle


def prompt_multi_choice(label: str, choices: list[str]) -> list[str]:
    """
    Displays a numbered list and lets the user pick zero or more entries by
    comma-separated index (e.g. '1,3,4'). Unlike the single-value prompts
    above, this one is intentionally forgiving: it silently drops
    unparseable or out-of-range entries and de-duplicates rather than
    looping forever, since this field is optional/exploratory (favorite
    target categories) rather than a required, structured value.
    """
    print(f"{label}:")
    for i, choice in enumerate(choices, 1):
        print(f"  {i}. {choice}")
    raw = input("Enter numbers separated by commas (or press Enter to skip): ").strip()
    if not raw:
        return []

    selected = []
    for part in raw.split(","):
        part = part.strip()
        if not part.isdigit():
            continue
        idx = int(part) - 1
        if 0 <= idx < len(choices):
            selected.append(choices[idx])

    return list(dict.fromkeys(selected))  # de-dupe, preserve order


def sanitize_free_text(raw: str, max_len: int = 300) -> str:
    """
    Sanitizes freeform text fields that may later be passed to an LLM agent
    as context (e.g. 'notes', fed to the chatbot in notebook 06). Strips
    control characters and enforces a length cap to reduce prompt-injection
    surface and prevent context/token-eating paste-bombs. Reserved for
    genuinely freeform fields — structured-looking fields (name, choices,
    numbers) should use a dedicated format-specific validator instead.
    """
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", raw).strip()
    return cleaned[:max_len]


def prompt_name(label: str = "Name", max_len: int = 100) -> str:
    """
    Prompts for a person's name, retrying until input contains only letters,
    spaces, hyphens, and apostrophes (covers names like 'Anne-Marie' or
    'O'Brien'), with no digits or symbols.
    """
    name_pattern = re.compile(r"^[A-Za-z][A-Za-z\s\-']{1,99}$")
    while True:
        raw = input(f"{label}: ")
        cleaned = sanitize_free_text(raw, max_len=max_len)
        if name_pattern.match(cleaned):
            return cleaned
        print("  Name must contain only letters, spaces, hyphens, or apostrophes (no numbers/symbols).")


def collect_weekly_plan_request_cli() -> WeeklyPlanRequest:
    """
    Collects user profile data via terminal prompts, validating and retrying
    on bad input at the source, then returns a Pydantic-validated
    WeeklyPlanRequest. No date is collected — the system always generates
    a plan for the upcoming 7 days from today. This is intentionally the
    ONLY function that depends on the input method (CLI here) — everything
    downstream consumes WeeklyPlanRequest objects regardless of how they
    were built, so this function can later be swapped for a web form
    without touching anything else in the system.
    """
    print("=== AstroPlanner: New User Profile ===")

    raw = {
        "user": {
            "name": prompt_name(),
            "latitude": prompt_float("Latitude (decimal degrees)", -90, 90),
            "longitude": prompt_float("Longitude (decimal degrees)", -180, 180),
            "experience_level": prompt_choice(
                "Experience level", ["beginner", "intermediate", "advanced"]
            ),
            "telescope": {
                "aperture_mm": prompt_float("Telescope aperture (mm)", 10, 1000),
                "focal_length_mm": prompt_float("Telescope focal length (mm)", 100, 5000),
            },
        },
    }

    bortle_choice = prompt_choice(
        "Bortle dark-sky scale for your site: do you already know the number, "
        "want a guided estimate based on what you can see, or want to skip it",
        ["know", "estimate", "skip"],
    )
    if bortle_choice == "know":
        raw["user"]["bortle_scale"] = prompt_int(
            "Bortle scale (1=excellent dark sky, 9=inner-city)", 1, 9
        )
    elif bortle_choice == "estimate":
        raw["user"]["bortle_scale"] = estimate_bortle_scale_interactive()

    if prompt_yes_no("Do you know your telescope's optical design?"):
        raw["user"]["telescope"]["type"] = prompt_choice(
            "Telescope type", ["refractor", "reflector", "catadioptric"]
        )

    if prompt_yes_no("Set observing preferences (mode + favorite targets)?"):
        raw["user"]["preferences"] = {
            "mode": prompt_choice("Observation mode", ["visual", "astrophotography"]),
            "favorite_targets": prompt_multi_choice(
                "Preferred target categories",
                [
                    "planet",
                    "moon",
                    "galaxy",
                    "nebula",
                    "open_cluster",
                    "globular_cluster",
                    "double_star",
                ],
            ),
        }

    if prompt_yes_no("Do you have a camera for imaging?"):
        raw["user"]["camera"] = {
            "sensor_width_mm": prompt_float("Sensor width (mm)", 4, 50),
            "sensor_height_mm": prompt_float("Sensor height (mm)", 4, 50),
            "pixel_size_um": prompt_float("Pixel size (µm)", 1, 15),
        }
        if prompt_yes_no("Do you know your camera type?"):
            raw["user"]["camera"]["type"] = prompt_choice(
                "Camera type", ["dslr", "mirrorless", "dedicated_astro"]
            )

    if prompt_yes_no("Do you know your mount type?"):
        raw["user"]["mount"] = {
            "type": prompt_choice("Mount type", ["alt-az", "equatorial"]),
            "goto_capable": prompt_yes_no("GoTo capable?"),
            "tracking": prompt_yes_no("Motorized tracking capable?"),
        }

    notes_raw = input("Notes (optional, press Enter to skip): ")
    if notes_raw.strip():
        raw["notes"] = sanitize_free_text(notes_raw)

    return WeeklyPlanRequest(**raw)
