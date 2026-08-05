import pytest

from recommendation import (
    classify_target_type,
    score_visibility,
    score_weather,
    score_moon,
    score_equipment,
    score_light_pollution,
    compute_preference_multiplier,
    score_object,
)
from models import TargetType, Preferences, ObservationMode


def test_classify_target_type_solar_system():
    assert classify_target_type({"is_solar_system": True, "name": "Moon"}) == TargetType.moon
    assert classify_target_type({"is_solar_system": True, "name": "Jupiter"}) == TargetType.planet


def test_classify_target_type_ngc_mapping():
    assert classify_target_type({"is_solar_system": False, "type": "GX"}) == TargetType.galaxy
    assert classify_target_type({"is_solar_system": False, "type": "PN"}) == TargetType.nebula
    assert classify_target_type({"is_solar_system": False, "type": "totally_unknown"}) is None


def test_score_visibility_bounds():
    bright_high = score_visibility({"peak_altitude_deg": 90, "magnitude": -2})
    faint_low = score_visibility({"peak_altitude_deg": 10, "magnitude": 14})
    assert 0 <= faint_low <= bright_high <= 1.001


def test_score_visibility_missing_magnitude_uses_midpoint():
    result = score_visibility({"peak_altitude_deg": 90, "magnitude": None})
    # alt_component = 1.0, mag_component defaults to 0.5 -> 0.5*1 + 0.5*0.5 = 0.75
    assert result == pytest.approx(0.75)


def test_score_weather_matches_raw_score():
    assert score_weather({"sky_quality": {"sky_quality_score": 73}}) == pytest.approx(0.73)


def test_score_moon_solar_system_never_penalized():
    assert score_moon({"is_solar_system": True}, moon_illumination_fraction=1.0) == 1.0


def test_score_moon_deep_sky_penalized_more_when_close_and_bright():
    close = score_moon({"is_solar_system": False, "moon_separation_deg": 5}, 1.0)
    far = score_moon({"is_solar_system": False, "moon_separation_deg": 89}, 1.0)
    assert close < far


def test_score_moon_no_moonlight_no_penalty_regardless_of_separation():
    close = score_moon({"is_solar_system": False, "moon_separation_deg": 5}, 0.0)
    assert close == pytest.approx(1.0)


class _FakeTelescope:
    def __init__(self, aperture_mm):
        self.aperture_mm = aperture_mm


class _FakeExperience:
    def __init__(self, value):
        self.value = value


class _FakeUser:
    def __init__(self, aperture_mm, experience_value):
        self.telescope = _FakeTelescope(aperture_mm)
        self.experience_level = _FakeExperience(experience_value)


def test_score_equipment_beginner_rewards_headroom():
    user = _FakeUser(aperture_mm=200, experience_value="beginner")
    obj_faint = {"is_solar_system": False, "magnitude": 8.7}   # near the limiting magnitude
    obj_bright = {"is_solar_system": False, "magnitude": 5.0}  # comfortably bright
    assert score_equipment(obj_bright, user) > score_equipment(obj_faint, user)


def test_score_equipment_advanced_flat_across_reachable_range():
    user = _FakeUser(aperture_mm=200, experience_value="advanced")
    obj_faint_but_reachable = {"is_solar_system": False, "magnitude": 12.0}
    obj_bright = {"is_solar_system": False, "magnitude": 5.0}
    # advanced users aren't penalized for chasing faint (but still reachable) targets
    assert score_equipment(obj_faint_but_reachable, user) == score_equipment(obj_bright, user) == 1.0


def test_score_equipment_solar_system_always_full():
    user = _FakeUser(aperture_mm=60, experience_value="beginner")
    assert score_equipment({"is_solar_system": True}, user) == 1.0


def test_score_light_pollution_unknown_returns_none():
    assert score_light_pollution({"is_solar_system": False}, bortle=None) is None


def test_score_light_pollution_solar_system_always_full():
    assert score_light_pollution({"is_solar_system": True}, bortle=9) == 1.0


def test_score_light_pollution_worse_bortle_scores_lower():
    good_sky = score_light_pollution({"is_solar_system": False}, bortle=2)
    bad_sky = score_light_pollution({"is_solar_system": False}, bortle=9)
    assert bad_sky < good_sky


def test_compute_preference_multiplier_boosts_matching_favorite(user_no_camera):
    user_no_camera.preferences = Preferences(mode=ObservationMode.visual, favorite_targets=[TargetType.nebula])
    nebula_obj = {"is_solar_system": False, "type": "PN"}
    other_obj = {"is_solar_system": False, "type": "GX"}
    assert compute_preference_multiplier(nebula_obj, user_no_camera) == 1.15
    assert compute_preference_multiplier(other_obj, user_no_camera) == 1.0


def test_compute_preference_multiplier_no_preferences_is_neutral(user_no_camera):
    obj = {"is_solar_system": False, "type": "PN"}
    assert compute_preference_multiplier(obj, user_no_camera) == 1.0


def test_score_object_drops_light_pollution_when_bortle_unknown(user_no_camera):
    obj = {
        "is_solar_system": False, "type": "GX",
        "peak_altitude_deg": 60, "magnitude": 9.0, "moon_separation_deg": 50,
    }
    night_weather = {"sky_quality": {"sky_quality_score": 80}}
    result = score_object(obj, night_weather, moon_illumination_fraction=0.2, user=user_no_camera, bortle=None)
    assert "light_pollution" not in result["factor_scores"]
    assert 0 <= result["recommendation_score"] <= 1.15  # multiplier can push slightly above 1


def test_score_object_includes_light_pollution_when_bortle_known(user_no_camera):
    obj = {
        "is_solar_system": False, "type": "GX",
        "peak_altitude_deg": 60, "magnitude": 9.0, "moon_separation_deg": 50,
    }
    night_weather = {"sky_quality": {"sky_quality_score": 80}}
    result = score_object(obj, night_weather, moon_illumination_fraction=0.2, user=user_no_camera, bortle=6)
    assert "light_pollution" in result["factor_scores"]
    assert result["target_type"] == "galaxy"
