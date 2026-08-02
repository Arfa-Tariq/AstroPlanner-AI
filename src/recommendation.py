"""
AstroPlanner — Recommendation Engine

Extracted from notebooks/04_recommendation_engine.ipynb. Same treatment as
weather.py / visibility.py: no Colab, no Drive, no notebook-only state.

Dependency note: get_weekly_recommendations needs a Skyfield `ts`
(timescale) and `eph` (ephemeris) to compute moon illumination. Rather
than loading its own copy of de421.bsp (the notebook's original
approach), this module takes ts/eph as parameters — callers should pass
in the same VisibilityEngine.ts / VisibilityEngine.eph already built for
that session, so the ~15MB ephemeris file is loaded once per
conversation, not once per tool call. See visibility.py's module
docstring for why this "build once, reuse" pattern matters here.
"""

from typing import Optional

import numpy as np

from models import UserProfile, TargetType

NGC_TYPE_TO_TARGET_TYPE = {
    "GX": TargetType.galaxy,
    "OC": TargetType.open_cluster,
    "GC": TargetType.globular_cluster,
    "BN": TargetType.nebula,
    "EN": TargetType.nebula,
    "RN": TargetType.nebula,
    "PN": TargetType.nebula,
    "SNR": TargetType.nebula,
    "SC": TargetType.open_cluster,
    "CL+N": TargetType.open_cluster,
    "G+C": TargetType.galaxy,
}


def classify_target_type(obj: dict) -> Optional[TargetType]:
    """Maps a visibility-engine object (planet/moon or NGC row) to TargetType."""
    if obj.get("is_solar_system"):
        return TargetType.moon if obj["name"] == "Moon" else TargetType.planet
    return NGC_TYPE_TO_TARGET_TYPE.get(obj.get("type"))


def get_moon_illumination(ts, eph, date_str: str) -> float:
    """
    Fraction of the Moon's disc illuminated at local midnight (UTC-naive
    approximation — fine for a nightly planning score, not precision
    photometry). 0.0 = new moon, 1.0 = full moon. ts/eph are the caller's
    already-loaded Skyfield objects (see module docstring).
    """
    from skyfield import almanac

    t = ts.utc(int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10]), 23, 59, 0)
    return float(almanac.fraction_illuminated(eph, "moon", t))


# ---------------------------------------------------------------------
# Per-factor scoring functions — each returns a float in [0, 1], higher
# is better, so they blend directly.
# ---------------------------------------------------------------------

def score_visibility(obj: dict) -> float:
    """Reuses the visibility engine's altitude/brightness inputs, recomputed
    here (the raw _score from visibility.py is stripped before storage)."""
    alt_component = obj["peak_altitude_deg"] / 90
    mag = obj.get("magnitude")
    if mag is None:
        mag_component = 0.5
    else:
        mag_component = float(np.clip((15 - mag) / 15, 0, 1))
    return 0.5 * alt_component + 0.5 * mag_component


def score_weather(night_weather: dict) -> float:
    """Directly reuses weather.py's 0-100 sky quality score."""
    return night_weather["sky_quality"]["sky_quality_score"] / 100


def score_moon(obj: dict, moon_illumination_fraction: float) -> float:
    """
    Combines nightly moon illumination with this object's angular
    separation from the Moon. Solar system objects aren't penalized by
    moon brightness. Deep-sky objects are penalized more when the moon
    is both bright AND close.
    """
    if obj.get("is_solar_system"):
        return 1.0

    separation = obj.get("moon_separation_deg")
    if separation is None:
        proximity_penalty = 0.5
    else:
        proximity_penalty = max(0.0, 1 - separation / 90)

    return 1 - (moon_illumination_fraction * proximity_penalty)


def score_equipment(obj: dict, user: UserProfile) -> float:
    """
    How well-matched this target is to the user's aperture and experience
    level. Beginners score best on comfortably bright targets; advanced
    users score well across the telescope's full reachable range.
    """
    if obj.get("is_solar_system"):
        return 1.0

    mag = obj.get("magnitude")
    if mag is None:
        return 0.5

    limiting_mag = 2.1 + 5 * np.log10(user.telescope.aperture_mm)
    headroom = limiting_mag - mag

    if user.experience_level.value == "beginner":
        return float(np.clip(headroom / 3, 0, 1))
    elif user.experience_level.value == "intermediate":
        return float(np.clip(0.5 + headroom / 6, 0, 1))
    else:  # advanced
        return 1.0 if headroom >= 0 else 0.0


def score_light_pollution(obj: dict, bortle: Optional[int]) -> Optional[float]:
    """
    Returns None (not 0) when bortle is unknown, so the caller drops this
    factor from the blend entirely rather than treating "unknown" as
    "worst case".
    """
    if bortle is None:
        return None
    if obj.get("is_solar_system"):
        return 1.0
    return float(np.clip(1 - (bortle - 1) / 9, 0.1, 1.0))


def compute_preference_multiplier(obj: dict, user: UserProfile) -> float:
    """Bounded bonus (1.0-1.15), not a core weighted factor — see score_object."""
    prefs = user.preferences
    if not prefs or not prefs.favorite_targets:
        return 1.0
    target_type = classify_target_type(obj)
    return 1.15 if target_type in prefs.favorite_targets else 1.0


def score_object(
    obj: dict,
    night_weather: dict,
    moon_illumination_fraction: float,
    user: UserProfile,
    bortle: Optional[int],
) -> dict:
    """Blends the available factors equally, then applies the preference
    multiplier on top. Returns the object with scoring fields attached."""
    factor_scores = {
        "visibility": score_visibility(obj),
        "weather": score_weather(night_weather),
        "moon": score_moon(obj, moon_illumination_fraction),
        "equipment": score_equipment(obj, user),
    }

    lp_score = score_light_pollution(obj, bortle)
    if lp_score is not None:
        factor_scores["light_pollution"] = lp_score

    base_score = sum(factor_scores.values()) / len(factor_scores)
    multiplier = compute_preference_multiplier(obj, user)
    final_score = round(base_score * multiplier, 4)

    target_type = classify_target_type(obj)

    return {
        **obj,
        "factor_scores": {k: round(v, 3) for k, v in factor_scores.items()},
        "preference_multiplier": multiplier,
        "target_type": target_type.value if target_type else None,
        "recommendation_score": final_score,
    }


def get_weekly_recommendations(
    user: UserProfile,
    weekly_sky_conditions: list,
    weekly_visibility: list,
    bortle: Optional[int],
    ts,
    eph,
    top_n: int = 15,
) -> list:
    """
    Joins weekly weather + visibility by date, scores every visible object
    per night, returns the top_n ranked recommendations per night. Nights
    present in one dataset but not the other are skipped rather than
    crashing the whole run.
    """
    weather_by_date = {n["date"]: n for n in weekly_sky_conditions}

    recommendations = []
    for night in weekly_visibility:
        date_str = night["date"]
        night_weather = weather_by_date.get(date_str)
        if night_weather is None:
            continue

        moon_illum = get_moon_illumination(ts, eph, date_str)

        scored = [
            score_object(obj, night_weather, moon_illum, user, bortle)
            for obj in night["objects"]
        ]
        scored.sort(key=lambda o: o["recommendation_score"], reverse=True)

        recommendations.append({
            "date": date_str,
            "day_offset": night["day_offset"],
            "moon_illumination_pct": round(moon_illum * 100, 1),
            "sky_quality_verdict": night_weather["sky_quality"]["verdict"],
            "bortle_scale_used": bortle,
            "recommended_objects": scored[:top_n],
        })

    return recommendations
