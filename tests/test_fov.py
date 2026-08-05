import pytest

from fov import (
    compute_fov_arcmin,
    compute_pixel_scale_arcsec_per_px,
    classify_sampling,
    classify_deep_sky_fit,
    classify_solar_system_target,
    get_weekly_fov_analysis,
)


def test_compute_fov_arcmin_basic():
    fov_w, fov_h = compute_fov_arcmin(23.5, 15.6, 1000)
    assert fov_w == pytest.approx(3437.7468 * 23.5 / 1000, rel=1e-6)
    assert fov_h == pytest.approx(3437.7468 * 15.6 / 1000, rel=1e-6)
    assert fov_w > fov_h  # sensor is wider than tall


def test_pixel_scale_and_sampling_bands():
    scale = compute_pixel_scale_arcsec_per_px(3.76, 1000)
    assert scale == pytest.approx(206264.8 * 0.00376 / 1000, rel=1e-6)

    assert classify_sampling(0.5) == "oversampled"
    assert classify_sampling(1.0) == "well_matched"
    assert classify_sampling(2.5) == "well_matched"
    assert classify_sampling(3.0) == "undersampled"


def test_classify_deep_sky_fit_unknown_size():
    result = classify_deep_sky_fit(None, 500, 400)
    assert result["fov_fit"] == "unknown"
    assert result["fit_ratio"] is None


def test_classify_deep_sky_fit_too_large():
    # frame_min_dim = 400; object 500 arcmin -> ratio 1.25 > 1.1
    result = classify_deep_sky_fit(500, 500, 400)
    assert result["fov_fit"] == "too_large"
    assert result["fit_ratio"] == pytest.approx(1.25)


def test_classify_deep_sky_fit_too_small():
    result = classify_deep_sky_fit(20, 500, 400)  # ratio 0.05 < 0.10
    assert result["fov_fit"] == "too_small"


def test_classify_deep_sky_fit_fits_well():
    result = classify_deep_sky_fit(200, 500, 400)  # ratio 0.5
    assert result["fov_fit"] == "fits_well"


def test_classify_solar_system_target_known_body():
    result = classify_solar_system_target("Jupiter", 1.5)
    assert result["fov_fit"] == "planetary_target"
    assert result["typical_angular_size_arcsec"] == 40.0
    assert result["sampling"] == "well_matched"


def test_classify_solar_system_target_unknown_body():
    result = classify_solar_system_target("Pluto", 1.5)
    assert result["typical_angular_size_arcsec"] is None
    assert result["typical_angular_size_arcmin"] is None


def _fake_recommendations(with_solar=True):
    objects = [
        {"name": "NGC1", "is_solar_system": False, "size_arcmin": 200, "recommendation_score": 0.5},
        {"name": "NGC2", "is_solar_system": False, "size_arcmin": None, "recommendation_score": 0.4},
    ]
    if with_solar:
        objects.append({"name": "Jupiter", "is_solar_system": True, "recommendation_score": 0.9})
    return [{"date": "2026-01-01", "recommended_objects": objects}]


def test_get_weekly_fov_analysis_no_camera(user_no_camera):
    weekly, setup = get_weekly_fov_analysis(_fake_recommendations(), user_no_camera)
    assert setup is None
    for obj in weekly[0]["recommended_objects"]:
        assert obj["fov_analysis"]["fov_fit"] == "no_camera"


def test_get_weekly_fov_analysis_with_camera(user_with_camera):
    weekly, setup = get_weekly_fov_analysis(_fake_recommendations(), user_with_camera)
    assert setup is not None
    assert setup["sampling"] in ("oversampled", "well_matched", "undersampled")

    fits = {o["name"]: o["fov_analysis"]["fov_fit"] for o in weekly[0]["recommended_objects"]}
    assert fits["Jupiter"] == "planetary_target"
    assert fits["NGC2"] == "unknown"  # no size_arcmin on record
    assert fits["NGC1"] in ("fits_well", "too_large", "too_small")
