"""
AstroPlanner — Celestial Visibility Tool

Extracted from notebooks/03_celestial_visibility(v3).ipynb (the v3 version,
not the original — v3 fixes the UTC-noon search-window bug and honest
rise/set pairing, no reason to build on top of the buggy one).

Design note — the "build once, reuse" problem:
Unlike weather.py, this notebook has 3 expensive one-time setup steps that
must NOT be redone for every one of the 7 nights:
  1. Downloading/loading the NGC catalog (network + parsing ~14k rows)
  2. Loading the Skyfield ephemeris file (de421.bsp)
  3. Building the Astroplan Observer + FixedTarget list from the filtered
     catalog

The notebook already got this half-right (it builds these once, outside
the per-night loop) — this module keeps that shape, just wrapped as a
class so a caller (the orchestrator) can build it ONCE per conversation
and reuse it across multiple tool calls, instead of re-downloading the
catalog and ephemeris on every single question the user asks.
"""

import os
import warnings
from collections import defaultdict
from datetime import datetime, timedelta, timezone, date as date_type
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from astropy.coordinates import EarthLocation, SkyCoord, AltAz, get_body
from astropy.time import Time
import astropy.units as u
from astropy.utils.iers import conf as iers_conf
from astroplan import Observer, FixedTarget, AltitudeConstraint, is_observable

from skyfield.api import Loader, wgs84, N, E
from skyfield import almanac
from timezonefinder import TimezoneFinder

from models import UserProfile

# Astroplan/Astropy will try to auto-download IERS tables; disable that,
# same as the notebook — the bundled table is accurate enough for planning.
iers_conf.auto_download = False
iers_conf.auto_max_age = None
warnings.filterwarnings("ignore", message=".*IERS.*")
warnings.filterwarnings("ignore", message=".*NonRotation.*")
warnings.filterwarnings("ignore", message=".*failed to download.*")
warnings.filterwarnings("ignore", message=".*Angular separation.*")
warnings.filterwarnings("ignore", message=".*unable to download.*")

USEFUL_OBJECT_TYPES = {
    "GX", "OC", "GC", "BN", "EN", "RN",
    "PN", "SNR", "SC", "CL+N", "G+C",
}

PLANETS = [
    ("moon", "Moon", "Moon", None),
    ("jupiter barycenter", "Jupiter", "Planet", -2.9),
    ("saturn barycenter", "Saturn", "Planet", 0.7),
    ("mars barycenter", "Mars", "Planet", 1.0),
    ("venus barycenter", "Venus", "Planet", -4.5),
    ("mercury barycenter", "Mercury", "Planet", -0.5),
]

TYPE_CAPS = {
    "GX": 30, "OC": 15, "GC": 15,
    "BN": 8, "EN": 8, "RN": 5,
    "PN": 10, "SNR": 5, "SC": 3,
    "CL+N": 5, "G+C": 3,
}


def get_observer_timezone(latitude: float, longitude: float) -> ZoneInfo:
    """Looks up IANA timezone from coordinates; falls back to UTC if unknown."""
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=latitude, lng=longitude)
    if tz_name is None:
        return ZoneInfo("UTC")
    return ZoneInfo(tz_name)


def compute_limiting_magnitude(aperture_mm: float) -> float:
    """Standard visual limiting magnitude: 2.1 + 5 * log10(aperture_mm)."""
    return 2.1 + 5 * np.log10(aperture_mm)


def _parse_ra(s):
    try:
        h, m, sec = str(s).split(":")
        return float(h) * 15 + float(m) * 0.25 + float(sec) * (15 / 3600)
    except Exception:
        return np.nan


def _parse_dec(s):
    try:
        parts = str(s).split(":")
        sign = -1 if str(s).startswith("-") else 1
        return sign * (abs(float(parts[0])) + float(parts[1]) / 60 + float(parts[2]) / 3600)
    except Exception:
        return np.nan


def load_ngc_catalog(data_dir: str) -> pd.DataFrame:
    """Downloads (once, cached to data_dir) and loads the OpenNGC catalog."""
    cache_path = os.path.join(data_dir, "ngc_catalog.csv")
    if not os.path.exists(cache_path):
        url = "https://raw.githubusercontent.com/mattiaverga/OpenNGC/master/database_files/NGC.csv"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(cache_path, "w") as f:
            f.write(r.text)
    df = pd.read_csv(cache_path, sep=";", low_memory=False)
    cols = ["Name", "Type", "RA", "Dec", "V-Mag", "B-Mag", "MajAx", "Common names"]
    df = df[[c for c in cols if c in df.columns]]
    df["magnitude"] = pd.to_numeric(df["V-Mag"], errors="coerce")
    df.loc[df["magnitude"].isna(), "magnitude"] = pd.to_numeric(
        df.loc[df["magnitude"].isna(), "B-Mag"], errors="coerce"
    )
    return df


def prefilter_catalog(df: pd.DataFrame, aperture_mm: float, latitude: float) -> pd.DataFrame:
    """Type filter -> magnitude filter (vs. telescope limit) -> declination filter (vs. latitude)."""
    df = df.copy()
    df["ra_deg"] = df["RA"].apply(_parse_ra)
    df["dec_deg"] = df["Dec"].apply(_parse_dec)
    df = df.dropna(subset=["ra_deg", "dec_deg"])

    lim_mag = compute_limiting_magnitude(aperture_mm)

    df = df[df["Type"].isin(USEFUL_OBJECT_TYPES)]
    df = df[df["magnitude"].isna() | (df["magnitude"] <= lim_mag)]
    df["max_altitude"] = 90 - abs(latitude - df["dec_deg"])
    df = df[df["max_altitude"] >= 30]

    return df.reset_index(drop=True)


def build_observer(latitude: float, longitude: float) -> Observer:
    return Observer(
        location=EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg, height=0 * u.m),
        name="observer",
    )


def build_target_list(df: pd.DataFrame) -> tuple[list, list]:
    targets, rows = [], []
    for _, row in df.iterrows():
        try:
            coord = SkyCoord(ra=row["ra_deg"] * u.deg, dec=row["dec_deg"] * u.deg)
            targets.append(FixedTarget(coord=coord, name=str(row["Name"])))
            rows.append(row)
        except Exception:
            continue
    return targets, rows


def get_night_window(observer: Observer, date_str: str):
    """Returns (night_start, night_end) astropy Times for astronomical twilight, or (None, None)."""
    try:
        midnight = Time(f"{date_str} 23:59:00")
        night_start = observer.twilight_evening_astronomical(midnight, which="nearest")
        night_end = observer.twilight_morning_astronomical(midnight, which="nearest")
        if (night_end - night_start).to(u.hour).value <= 0:
            return None, None
        return night_start, night_end
    except Exception:
        return None, None


def get_local_noon_window(ts, date_str: str, local_tz: ZoneInfo):
    """v3 fix: local-noon -> local-noon search window (not UTC-noon), so nothing gets clipped."""
    year, month, day = int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10])
    local_noon_start = datetime(year, month, day, 12, 0, 0, tzinfo=local_tz)
    local_noon_end = local_noon_start + timedelta(days=1)
    t0 = ts.from_datetime(local_noon_start.astimezone(timezone.utc))
    t1 = ts.from_datetime(local_noon_end.astimezone(timezone.utc))
    return t0, t1


class VisibilityEngine:
    """
    Holds the 3 expensive setup objects (catalog, ephemeris, observer+targets)
    so they're built ONCE and reused across every tool call in a conversation,
    instead of re-downloading/rebuilding per question. Create one instance
    per (location, aperture) combination — the orchestrator is responsible
    for caching instances across turns of the same session.
    """

    def __init__(self, latitude: float, longitude: float, aperture_mm: float, data_dir: str = "./data"):
        os.makedirs(data_dir, exist_ok=True)
        self.latitude = latitude
        self.longitude = longitude
        self.local_tz = get_observer_timezone(latitude, longitude)

        # 1. Ephemeris — loaded/cached once.
        loader = Loader(data_dir)
        self.ts = loader.timescale()
        self.eph = loader("de421.bsp")

        # 2. Catalog — downloaded/cached once, then filtered for this setup.
        raw_catalog = load_ngc_catalog(data_dir)
        filtered = prefilter_catalog(raw_catalog, aperture_mm, latitude)

        # 3. Observer + target list — built once from the filtered catalog.
        self.observer = build_observer(latitude, longitude)
        self.targets, self.valid_rows = build_target_list(filtered)

    def _get_planet_visibility(self, date_str: str) -> list[dict]:
        t0, t1 = get_local_noon_window(self.ts, date_str, self.local_tz)
        window_days = t1 - t0
        skyfield_location = wgs84.latlon(self.latitude * N, self.longitude * E)
        observer_sf = self.eph["earth"] + skyfield_location
        astropy_night_start, astropy_night_end = get_night_window(self.observer, date_str)
        t_grid = t0 + np.linspace(0, window_days, 145)

        if astropy_night_start is not None:
            t_night_start = self.ts.from_datetime(astropy_night_start.to_datetime().replace(tzinfo=timezone.utc))
            t_night_end = self.ts.from_datetime(astropy_night_end.to_datetime().replace(tzinfo=timezone.utc))
            night_t_grid = t_night_start + np.linspace(0, t_night_end - t_night_start, 30)
        else:
            night_t_grid = None

        try:
            local_noon_astropy = Time(datetime(
                int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10]), 12, 0, 0, tzinfo=self.local_tz
            ).astimezone(timezone.utc))
            sunrise = self.observer.sun_rise_time(local_noon_astropy, which="nearest")
            sunset = self.observer.sun_set_time(local_noon_astropy, which="nearest")
            t_sunrise = self.ts.from_datetime(sunrise.to_datetime().replace(tzinfo=timezone.utc))
            t_sunset = self.ts.from_datetime(sunset.to_datetime().replace(tzinfo=timezone.utc))
            day_start, day_end = (t_sunrise, t_sunset) if t_sunrise.tt < t_sunset.tt else (t_sunset, t_sunrise)
            day_t_grid = day_start + np.linspace(0, day_end - day_start, 30)
        except Exception:
            year, month, day = int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10])
            local_6am = datetime(year, month, day, 6, 0, 0, tzinfo=self.local_tz)
            local_6pm = datetime(year, month, day, 18, 0, 0, tzinfo=self.local_tz)
            t_day_start = self.ts.from_datetime(local_6am.astimezone(timezone.utc))
            t_day_end = self.ts.from_datetime(local_6pm.astimezone(timezone.utc))
            day_t_grid = t_day_start + np.linspace(0, t_day_end - t_day_start, 25)

        def fmt_local(t):
            return t.utc_datetime().astimezone(self.local_tz).strftime("%m/%d %H:%M")

        results = []
        for body_key, display_name, obj_type, typical_mag in PLANETS:
            try:
                body = self.eph[body_key]
                f = almanac.risings_and_settings(self.eph, body, wgs84.latlon(self.latitude * N, self.longitude * E), horizon_degrees=-0.5)
                times, events = almanac.find_discrete(t0, t1, f)

                alt0 = observer_sf.at(t0).observe(body).apparent().altaz()[0].degrees
                already_up = alt0 > -0.5

                rise_time = set_time = None
                if already_up:
                    rise_time = "already up"
                    for t_ev, ev in zip(times, events):
                        if ev == 0:
                            set_time = fmt_local(t_ev)
                            break
                else:
                    for i, (t_ev, ev) in enumerate(zip(times, events)):
                        if ev == 1:
                            rise_time = fmt_local(t_ev)
                            for t_ev2, ev2 in zip(times[i + 1:], events[i + 1:]):
                                if ev2 == 0:
                                    set_time = fmt_local(t_ev2)
                                    break
                            break

                if rise_time is not None and set_time is None:
                    set_time = "still up at window end"
                elif rise_time is None and set_time is None:
                    alt1 = observer_sf.at(t1).observe(body).apparent().altaz()[0].degrees
                    if alt0 > -0.5 and alt1 > -0.5:
                        rise_time = set_time = "up all window"
                    else:
                        rise_time = set_time = "not visible this window"

                astrometric = observer_sf.at(t_grid).observe(body).apparent()
                alt, az, _ = astrometric.altaz()
                peak_idx = int(np.argmax(alt.degrees))
                peak_alt = float(alt.degrees[peak_idx])
                transit_time = fmt_local(t_grid[peak_idx])

                if night_t_grid is not None:
                    night_alt = observer_sf.at(night_t_grid).observe(body).apparent().altaz()[0].degrees
                    visible_at_night = float(np.max(night_alt)) >= 30
                else:
                    visible_at_night = False
                day_alt = observer_sf.at(day_t_grid).observe(body).apparent().altaz()[0].degrees
                visible_at_day = float(np.max(day_alt)) >= 30

                if not visible_at_night and not visible_at_day:
                    continue
                observable_period = "both" if (visible_at_night and visible_at_day) else ("night" if visible_at_night else "day")

                results.append({
                    "name": display_name, "common_name": display_name, "type": obj_type,
                    "magnitude": typical_mag, "size_arcmin": None, "ra_deg": None, "dec_deg": None,
                    "peak_altitude_deg": round(peak_alt, 1), "transit_time": transit_time,
                    "rise_time": rise_time, "set_time": set_time,
                    "moon_separation_deg": None, "moon_warning": False, "transits_after_dawn": False,
                    "is_solar_system": True, "observable_period": observable_period,
                })
            except Exception:
                continue
        return results

    def _get_deep_sky_visibility(self, date_str: str) -> list[dict]:
        try:
            night_start, night_end = get_night_window(self.observer, date_str)
            if night_start is None:
                return []

            duration_hours = (night_end - night_start).to(u.hour).value
            n_steps = max(int(duration_hours * 6), 2)
            time_grid = night_start + np.linspace(0, duration_hours, n_steps) * u.hour
            time_labels = [t.to_datetime().replace(tzinfo=timezone.utc).astimezone(self.local_tz).strftime("%H:%M") for t in time_grid]

            observable_mask = is_observable(
                [AltitudeConstraint(min=30 * u.deg)], self.observer, self.targets,
                time_range=[night_start, night_end],
            )

            moon_coords = get_body("moon", time_grid, ephemeris="builtin")
            moon_icrs_grid = SkyCoord(ra=moon_coords.ra.deg * u.deg, dec=moon_coords.dec.deg * u.deg, frame="icrs")

            visible = []
            for target, row, is_obs in zip(self.targets, self.valid_rows, observable_mask):
                if not is_obs:
                    continue
                try:
                    altaz = target.coord.transform_to(AltAz(obstime=time_grid, location=self.observer.location))
                    alts = altaz.alt.deg
                    peak_idx = int(np.argmax(alts))
                    peak_alt = float(alts[peak_idx])
                    peak_time = time_labels[peak_idx]

                    above = alts >= 30
                    rising_indices = np.where(above)[0]
                    if len(rising_indices) == 0:
                        continue

                    above_from = "already above 30° at dusk" if rising_indices[0] == 0 else time_labels[rising_indices[0]]
                    above_until = "still above 30° at dawn" if rising_indices[-1] == n_steps - 1 else time_labels[rising_indices[-1]]
                    transits_after_dawn = peak_idx == n_steps - 1

                    moon_sep = float(moon_icrs_grid[peak_idx].separation(target.coord).deg)
                    mag = row["magnitude"]
                    common = str(row.get("Common names", "") or "").split(";")[0].strip() or None

                    visible.append({
                        "name": str(row["Name"]), "common_name": common, "type": str(row.get("Type", "Unknown")),
                        "magnitude": float(mag) if not pd.isna(mag) else None,
                        "size_arcmin": float(row["MajAx"]) if "MajAx" in row and not pd.isna(row.get("MajAx")) else None,
                        "ra_deg": round(float(row["ra_deg"]), 4), "dec_deg": round(float(row["dec_deg"]), 4),
                        "peak_altitude_deg": round(peak_alt, 1), "peak_time_local": peak_time,
                        "altitude_threshold_deg": 30,
                        "above_30deg_from_local": above_from, "above_30deg_until_local": above_until,
                        "transits_after_dawn": transits_after_dawn,
                        "moon_separation_deg": round(moon_sep, 1), "moon_warning": moon_sep < 30,
                        "is_solar_system": False, "observable_period": "night",
                    })
                except Exception:
                    continue

            for obj in visible:
                b = max(0, (15 - (obj["magnitude"] or 15)) / 15)
                a = obj["peak_altitude_deg"] / 90
                obj["_score"] = 0.5 * a + 0.5 * b
            visible.sort(key=lambda x: x["_score"], reverse=True)

            counts = defaultdict(int)
            diverse = []
            for obj in visible:
                t = obj["type"]
                if counts[t] < TYPE_CAPS.get(t, 5):
                    diverse.append(obj)
                    counts[t] += 1
                if len(diverse) >= 100:
                    break
            for obj in diverse:
                obj.pop("_score", None)
            return diverse
        except Exception:
            return []

    def get_weekly_visibility(self, start_date: date_type) -> list[dict]:
        """The one method the orchestrator/tool schema calls. Combines planets + deep-sky per night."""
        weekly = []
        for offset in range(7):
            date_str = (start_date + timedelta(days=offset)).strftime("%Y-%m-%d")
            nightly = self._get_planet_visibility(date_str) + self._get_deep_sky_visibility(date_str)
            weekly.append({
                "date": date_str, "day_offset": offset, "timezone": str(self.local_tz),
                "visible_object_count": len(nightly), "objects": nightly,
            })
        return weekly


def get_weekly_visibility(user: UserProfile, start_date: date_type, data_dir: str = "./data") -> list[dict]:
    """
    Convenience one-shot function matching weather.py's style, for simple
    callers that don't need to reuse a VisibilityEngine across multiple
    calls. Prefer VisibilityEngine directly when calling this repeatedly
    in the same conversation/session — this rebuilds everything each time.
    """
    engine = VisibilityEngine(user.latitude, user.longitude, user.telescope.aperture_mm, data_dir)
    return engine.get_weekly_visibility(start_date)
