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

    if prompt_yes_no("Do you know your site's Bortle dark-sky scale? (1=excellent dark sky, 9=inner-city)"):
        raw["user"]["bortle_scale"] = prompt_int("Bortle scale", 1, 9)

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
