"""
AstroPlanner — Weather Intelligence Tool

Extracted from notebooks/02_weather_intelligence.ipynb. This is the first
notebook converted into a plain, importable module rather than a Colab
script — no drive.mount(), no !pip install, no notebook-only state.

Design intent: this file has ZERO knowledge of LLMs, chat, or tool-calling.
It only knows how to compute a 7-night sky-quality outlook from a
UserProfile. The orchestrator (built separately) is what turns
get_weekly_sky_conditions into something an LLM can call — this module
just needs to be a clean, predictable function to call.
"""

from datetime import date as date_type, datetime, timedelta
from typing import Optional

import requests

from models import UserProfile


def fetch_open_meteo_forecast(latitude: float, longitude: float, target_date: str) -> dict:
    """
    Fetches daily and hourly weather forecast from Open-Meteo for the given
    coordinates and date. No API key required. Returns cloud cover, humidity,
    wind speed, temperature, and precipitation probability — the general
    weather baseline. Seeing/transparency (astronomy-specific) are NOT
    available here and come from fetch_7timer_astro instead.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "cloudcover,relative_humidity_2m,wind_speed_10m,temperature_2m,precipitation_probability",
        "start_date": target_date,
        "end_date": target_date,
        "timezone": "auto",
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])

    target_hour = f"{target_date}T22:00"
    if target_hour in times:
        idx = times.index(target_hour)
    elif times:
        idx = len(times) // 2
    else:
        idx = None

    return {
        "source": "open-meteo",
        "date": target_date,
        "reference_time": times[idx] if idx is not None else None,
        "cloud_cover_pct": hourly.get("cloudcover", [None])[idx] if idx is not None else None,
        "humidity_pct": hourly.get("relative_humidity_2m", [None])[idx] if idx is not None else None,
        "wind_speed_kmh": hourly.get("wind_speed_10m", [None])[idx] if idx is not None else None,
        "temperature_c": hourly.get("temperature_2m", [None])[idx] if idx is not None else None,
        "precipitation_probability_pct": hourly.get("precipitation_probability", [None])[idx] if idx is not None else None,
    }


def fetch_7timer_astro(latitude: float, longitude: float, target_date: str) -> Optional[dict]:
    """
    Fetches the astronomy-specific forecast from 7Timer (seeing, transparency,
    cloud cover) for the given coordinates. Only forecasts ~3 days ahead from
    today; returns None for dates beyond that window or on any API failure,
    so callers degrade gracefully to weather-only data.
    """
    today = date_type.today()
    requested = date_type.fromisoformat(target_date)
    days_out = (requested - today).days

    if days_out < 0 or days_out > 3:
        return None

    url = "http://www.7timer.info/bin/api.pl"
    params = {"lon": longitude, "lat": latitude, "product": "astro", "output": "json"}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None

    series = data.get("dataseries", [])
    init_str = data.get("init")
    if not series or not init_str:
        return None

    init_dt = datetime.strptime(init_str, "%Y%m%d%H")
    target_dt = datetime.combine(requested, datetime.min.time()).replace(hour=22)

    best_entry = min(
        series,
        key=lambda e: abs((init_dt + timedelta(hours=e["timepoint"])) - target_dt),
    )

    return {
        "source": "7timer",
        "date": target_date,
        "cloud_cover_index": best_entry.get("cloudcover"),
        "seeing_index": best_entry.get("seeing"),
        "transparency_index": best_entry.get("transparency"),
    }


def compute_sky_quality_score(open_meteo_data: dict, astro_data: Optional[dict]) -> dict:
    """
    Combines general weather with astronomy-specific seeing/transparency
    (when available) into a 0-100 sky quality score and plain-language
    verdict. Pure rule-based scoring — no LLM involved.
    """
    score = 100
    reasons = []

    cloud_cover = open_meteo_data.get("cloud_cover_pct")
    if cloud_cover is not None:
        score -= cloud_cover * 0.6
        if cloud_cover > 70:
            reasons.append(f"Heavy cloud cover ({cloud_cover}%)")
        elif cloud_cover > 30:
            reasons.append(f"Moderate cloud cover ({cloud_cover}%)")

    precip = open_meteo_data.get("precipitation_probability_pct")
    if precip is not None and precip > 30:
        score -= precip * 0.3
        reasons.append(f"Precipitation risk ({precip}%)")

    wind = open_meteo_data.get("wind_speed_kmh")
    if wind is not None and wind > 25:
        score -= min((wind - 25), 30)
        reasons.append(f"High wind ({wind} km/h) — may cause telescope vibration")

    humidity = open_meteo_data.get("humidity_pct")
    if humidity is not None and humidity >= 85:
        score -= 10
        reasons.append(f"High humidity ({humidity}%) — dew risk on optics")

    if astro_data:
        seeing = astro_data.get("seeing_index")
        transparency = astro_data.get("transparency_index")
        if seeing is not None:
            score -= (seeing - 1) * 5
            if seeing >= 6:
                reasons.append("Poor atmospheric seeing — fine detail will be blurry")
        if transparency is not None:
            score -= (transparency - 1) * 5
            if transparency >= 6:
                reasons.append("Poor sky transparency — faint objects harder to see")

    score = max(0, min(100, round(score)))

    if score >= 80:
        verdict = "Excellent"
    elif score >= 60:
        verdict = "Good"
    elif score >= 40:
        verdict = "Fair"
    elif score >= 20:
        verdict = "Poor"
    else:
        verdict = "Not recommended"

    return {
        "sky_quality_score": score,
        "verdict": verdict,
        "reasons": reasons if reasons else ["Clear conditions, no major concerns"],
    }


def get_weekly_sky_conditions(user_profile: UserProfile, start_date: date_type) -> list[dict]:
    """
    Generates a 7-day sky conditions outlook (start_date through
    start_date+6) for the user's location. Days 0-3 include full data:
    weather + 7Timer seeing/transparency. Days 4-6 are 'extended_outlook':
    weather only, since no free seeing/transparency forecast extends
    that far.

    THIS is the function that will become a tool. Its signature —
    (UserProfile, date) in, list[dict] out — is exactly what the
    orchestrator's tool schema will describe to the LLM.
    """
    lat = user_profile.latitude
    lon = user_profile.longitude

    weekly_plan = []
    for offset in range(7):
        day = start_date + timedelta(days=offset)
        day_str = day.isoformat()

        weather = fetch_open_meteo_forecast(lat, lon, day_str)
        astro = fetch_7timer_astro(lat, lon, day_str)
        quality = compute_sky_quality_score(weather, astro)

        weekly_plan.append({
            "date": day_str,
            "day_offset": offset,
            "data_confidence": "full" if astro is not None else "extended_outlook",
            "weather": weather,
            "astro_conditions": astro,
            "sky_quality": quality,
        })

    return weekly_plan
