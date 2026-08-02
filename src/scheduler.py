"""
AstroPlanner — Observation Scheduler

Extracted from notebooks/06_observation_scheduler.ipynb. Takes the FoV
analyzer's output (weekly nights + setup_summary) and turns each night's
ranked objects into a non-overlapping timeline.
"""

from datetime import datetime, timedelta
from typing import Optional

DEFAULT_SESSION_MINUTES = {
    "planetary_target": 25,
    "fits_well": 40,
    "too_small": 30,
    "too_large": 45,
    "unknown": 30,
    "no_camera": 20,
}


def parse_local_time(time_str: str, night_date: str) -> Optional[datetime]:
    """
    Converts the mixed time formats from notebooks 03/04 into real
    datetimes on a shared timeline. Handles 'HH:MM', 'MM/DD HH:MM', and
    non-parseable labels (returns None, caller decides how to handle a
    missing bound).
    """
    if not time_str or not isinstance(time_str, str):
        return None

    year, month, day = int(night_date[:4]), int(night_date[5:7]), int(night_date[8:10])

    if "/" in time_str:  # 'MM/DD HH:MM'
        try:
            md, hm = time_str.split(" ")
            mm, dd = md.split("/")
            hh, mi = hm.split(":")
            return datetime(year, int(mm), int(dd), int(hh), int(mi))
        except Exception:
            return None

    if ":" in time_str and len(time_str) <= 5:  # 'HH:MM'
        try:
            hh, mi = time_str.split(":")
            hh, mi = int(hh), int(mi)
            dt = datetime(year, month, day, hh, mi)
            if hh < 12:
                dt += timedelta(days=1)
            return dt
        except Exception:
            return None

    return None


def get_object_window(obj: dict, night_date: str) -> tuple:
    """
    Returns (start_dt, end_dt, peak_dt) for one object's observing window,
    reconciling solar-system field names (rise_time/set_time/transit_time)
    vs deep-sky field names (above_30deg_from_local/until_local,
    peak_time_local). Falls back to a window centered on peak time when a
    boundary is a non-parseable label.
    """
    is_solar = obj.get("is_solar_system", False)

    if is_solar:
        start = parse_local_time(obj.get("rise_time"), night_date)
        end = parse_local_time(obj.get("set_time"), night_date)
        peak = parse_local_time(obj.get("transit_time"), night_date)
    else:
        start = parse_local_time(
            obj.get("above_30deg_from_local") or obj.get("rise_time_local"), night_date
        )
        end = parse_local_time(
            obj.get("above_30deg_until_local") or obj.get("set_time_local"), night_date
        )
        peak = parse_local_time(obj.get("peak_time_local"), night_date)

    if peak is None:
        return None, None, None

    if start is None:
        start = peak - timedelta(minutes=30)
    if end is None:
        end = peak + timedelta(minutes=30)

    return start, end, peak


def build_night_schedule(night: dict, max_objects: int = 8) -> dict:
    """Builds one night's schedule: night_session (non-overlapping slots
    during astronomical darkness) + daytime_bonus (solar system targets
    visible only in daylight)."""
    date_str = night["date"]
    night_scheduled, day_bonus = [], []

    dark_start, dark_end = None, None
    for obj in night["recommended_objects"]:
        if obj.get("is_solar_system"):
            continue
        s, e, _ = get_object_window(obj, date_str)
        if s is None or e is None:
            continue
        dark_start = s if dark_start is None else min(dark_start, s)
        dark_end = e if dark_end is None else max(dark_end, e)

    for obj in night["recommended_objects"]:
        start, end, peak = get_object_window(obj, date_str)
        if peak is None:
            continue

        period = obj.get("observable_period", "night")
        fov_fit = obj.get("fov_analysis", {}).get("fov_fit", "unknown")
        duration = timedelta(minutes=DEFAULT_SESSION_MINUTES.get(fov_fit, 30))

        slot_start = max(start, peak - duration / 2)
        slot_end = min(end, slot_start + duration)
        if slot_end <= slot_start:
            continue

        peak_is_dark = (
            dark_start is not None and dark_end is not None
            and dark_start <= peak <= dark_end
        )

        entry = {
            "name": obj["name"], "common_name": obj.get("common_name"),
            "target_type": obj.get("target_type"),
            "recommendation_score": obj.get("recommendation_score"),
            "fov_fit": fov_fit, "observable_period": period,
            "start": slot_start, "end": slot_end, "peak": peak,
            "note": obj.get("fov_analysis", {}).get("note", ""),
        }

        if obj.get("is_solar_system") and not peak_is_dark:
            day_bonus.append(entry)
            continue

        overlaps = any(slot_start < s["end"] and slot_end > s["start"] for s in night_scheduled)
        if overlaps:
            continue
        night_scheduled.append(entry)
        if len(night_scheduled) >= max_objects:
            break

    night_scheduled.sort(key=lambda s: s["start"])
    day_bonus.sort(key=lambda s: s["start"])
    return {"night_session": night_scheduled, "daytime_bonus": day_bonus[:3]}


def get_weekly_schedule(weekly_nights: list, max_objects: int = 8) -> list[dict]:
    """The one function the tool wrapper calls: builds all 7 nights'
    schedules and formats datetimes back to HH:MM strings for storage."""
    def fmt(slot):
        return {**{k: v for k, v in slot.items() if k not in ("start", "end", "peak")},
                "start_local": slot["start"].strftime("%H:%M"),
                "end_local": slot["end"].strftime("%H:%M"),
                "peak_local": slot["peak"].strftime("%H:%M")}

    weekly = []
    for night in weekly_nights:
        result = build_night_schedule(night, max_objects=max_objects)
        weekly.append({
            "date": night["date"],
            "timeline": [fmt(s) for s in result["night_session"]],
            "daytime_bonus": [fmt(s) for s in result["daytime_bonus"]],
        })
    return weekly
