"""
AstroPlanner — Shared CLI Input Helpers

Validated, retry-on-bad-input prompt functions used to collect a
WeeklyPlanRequest from the terminal (Colab cell input()). Kept separate
from models.py so a future frontend can replace this file entirely without
touching the schemas it produces.
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

    if prompt_yes_no("Do you have a camera for imaging?"):
        raw["user"]["camera"] = {
            "sensor_width_mm": prompt_float("Sensor width (mm)", 4, 50),
            "sensor_height_mm": prompt_float("Sensor height (mm)", 4, 50),
            "pixel_size_um": prompt_float("Pixel size (µm)", 1, 15),
        }

    if prompt_yes_no("Do you know your mount type?"):
        raw["user"]["mount"] = {
            "type": prompt_choice("Mount type", ["alt-az", "equatorial"]),
            "goto_capable": prompt_yes_no("GoTo capable?"),
        }

    notes_raw = input("Notes (optional, press Enter to skip): ")
    if notes_raw.strip():
        raw["notes"] = sanitize_free_text(notes_raw)

    return WeeklyPlanRequest(**raw)
