from datetime import datetime, timedelta

from scheduler import (
    parse_local_time,
    get_object_window,
    build_night_schedule,
    get_weekly_schedule,
)


def test_parse_local_time_hhmm_evening():
    dt = parse_local_time("22:30", "2026-07-27")
    assert dt == datetime(2026, 7, 27, 22, 30)


def test_parse_local_time_hhmm_early_morning_rolls_to_next_day():
    # hours before 12:00 belong to the *following* calendar day within a night window
    dt = parse_local_time("02:15", "2026-07-27")
    assert dt == datetime(2026, 7, 28, 2, 15)


def test_parse_local_time_mm_dd_hhmm():
    dt = parse_local_time("07/28 05:10", "2026-07-27")
    assert dt == datetime(2026, 7, 28, 5, 10)


def test_parse_local_time_non_parseable_label_returns_none():
    assert parse_local_time("already up", "2026-07-27") is None
    assert parse_local_time("circumpolar", "2026-07-27") is None
    assert parse_local_time(None, "2026-07-27") is None


def test_get_object_window_deep_sky_falls_back_around_peak():
    obj = {
        "is_solar_system": False,
        "peak_time_local": "23:00",
        "above_30deg_from_local": "already above 30° at dusk",  # non-parseable -> None
        "above_30deg_until_local": "03:00",
    }
    start, end, peak = get_object_window(obj, "2026-07-27")
    assert peak == datetime(2026, 7, 27, 23, 0)
    assert start == peak - timedelta(minutes=30)  # fallback since 'from' wasn't parseable
    assert end == datetime(2026, 7, 28, 3, 0)


def test_get_object_window_missing_peak_returns_all_none():
    obj = {"is_solar_system": False, "peak_time_local": "not visible this window"}
    start, end, peak = get_object_window(obj, "2026-07-27")
    assert (start, end, peak) == (None, None, None)


def _night(objects):
    return {"date": "2026-07-27", "recommended_objects": objects}


def test_build_night_schedule_skips_overlapping_lower_priority_object():
    # Two deep-sky objects (fits_well -> 40 min sessions) with overlapping
    # windows. A comes first (higher score) and should win the slot; B's
    # slot overlaps A's and should be dropped, not double-booked.
    objects = [
        {
            "name": "A", "is_solar_system": False, "recommendation_score": 0.9,
            "peak_time_local": "22:00",
            "above_30deg_from_local": "21:30", "above_30deg_until_local": "22:30",
            "fov_analysis": {"fov_fit": "fits_well", "note": ""},
        },
        {
            "name": "B", "is_solar_system": False, "recommendation_score": 0.5,
            "peak_time_local": "22:10",
            "above_30deg_from_local": "21:50", "above_30deg_until_local": "22:40",
            "fov_analysis": {"fov_fit": "fits_well", "note": ""},
        },
    ]
    result = build_night_schedule(_night(objects))
    names = [s["name"] for s in result["night_session"]]
    assert names == ["A"]


def test_build_night_schedule_solar_system_daytime_goes_to_bonus():
    objects = [
        {
            "name": "Jupiter", "is_solar_system": True, "recommendation_score": 0.8,
            "transit_time": "07/27 12:00", "rise_time": "07/27 06:00", "set_time": "07/27 19:00",
            "observable_period": "day",
            "fov_analysis": {"fov_fit": "planetary_target", "note": ""},
        },
    ]
    result = build_night_schedule(_night(objects))
    assert result["night_session"] == []
    assert len(result["daytime_bonus"]) == 1
    assert result["daytime_bonus"][0]["name"] == "Jupiter"


def test_get_weekly_schedule_formats_times_as_strings():
    objects = [
        {
            "name": "A", "is_solar_system": False, "recommendation_score": 0.9,
            "peak_time_local": "22:00",
            "above_30deg_from_local": "21:30", "above_30deg_until_local": "22:30",
            "fov_analysis": {"fov_fit": "fits_well", "note": "well framed"},
        },
    ]
    weekly = get_weekly_schedule([_night(objects)])
    slot = weekly[0]["timeline"][0]
    # duration for fits_well = 40 min; slot_start = max(21:30, 22:00-20min) = 21:40
    # slot_end = min(22:30, 21:40+40min) = 22:20
    assert slot["start_local"] == "21:40"
    assert slot["end_local"] == "22:20"
    assert slot["peak_local"] == "22:00"
    assert slot["note"] == "well framed"
