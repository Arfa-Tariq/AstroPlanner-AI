"""
AstroPlanner — Field-of-View Analyzer

Extracted from notebooks/05_field_of_view_analyzer.ipynb. No external
astronomy library dependency — just trigonometry — so this module has
no expensive setup to cache, unlike weather/visibility/recommendation.
"""

from typing import Optional

from models import UserProfile

ARCMIN_PER_RADIAN = 3437.7468
ARCSEC_PER_RADIAN = 206264.8
MOON_DIAMETER_ARCMIN = 31.0

# Rough mean apparent size (arcsec) for solar system bodies. Varies
# night to night with orbital distance, but a fixed typical value is
# good enough to steer the imaging-technique recommendation.
SOLAR_SYSTEM_TYPICAL_SIZE_ARCSEC = {
    "Moon": 1860.0,
    "Jupiter": 40.0,
    "Saturn": 17.0,
    "Mars": 10.0,
    "Venus": 25.0,
    "Mercury": 7.0,
}


def compute_fov_arcmin(sensor_width_mm: float, sensor_height_mm: float, focal_length_mm: float) -> tuple[float, float]:
    """FoV (arcmin) = 3437.75 * sensor_dimension_mm / focal_length_mm."""
    fov_w = ARCMIN_PER_RADIAN * sensor_width_mm / focal_length_mm
    fov_h = ARCMIN_PER_RADIAN * sensor_height_mm / focal_length_mm
    return fov_w, fov_h


def compute_pixel_scale_arcsec_per_px(pixel_size_um: float, focal_length_mm: float) -> float:
    """Image scale arcsec/px — governs oversampled vs undersampled."""
    pixel_size_mm = pixel_size_um / 1000
    return ARCSEC_PER_RADIAN * pixel_size_mm / focal_length_mm


def classify_sampling(arcsec_per_px: float) -> str:
    """<1.0"/px oversampled, 1.0-2.5"/px well_matched, >2.5"/px undersampled."""
    if arcsec_per_px < 1.0:
        return "oversampled"
    if arcsec_per_px <= 2.5:
        return "well_matched"
    return "undersampled"


def fov_arcmin_to_human(fov_w_arcmin: float, fov_h_arcmin: float) -> dict:
    """Converts raw arcmin FoV into degrees + a plain-language Moon-width comparison."""
    fov_w_deg = fov_w_arcmin / 60
    fov_h_deg = fov_h_arcmin / 60
    moons_across = fov_w_arcmin / MOON_DIAMETER_ARCMIN

    if moons_across >= 1:
        moon_note = f"\u2248{moons_across:.1f} full Moons could fit across the frame width"
    else:
        moon_note = f"the full Moon would take up \u2248{1/moons_across:.1f}\u00d7 the frame width"

    return {
        "fov_width_deg": round(fov_w_deg, 2),
        "fov_height_deg": round(fov_h_deg, 2),
        "human_summary": f"{fov_w_deg:.2f}\u00b0 \u00d7 {fov_h_deg:.2f}\u00b0  ({moon_note})",
    }


def classify_deep_sky_fit(object_size_arcmin: Optional[float], fov_w_arcmin: float, fov_h_arcmin: float) -> dict:
    """
    Compares object size against the FoV's shorter dimension.
    ratio > 1.1 -> too_large, ratio < 0.10 -> too_small, else fits_well.
    """
    frame_min_dim = min(fov_w_arcmin, fov_h_arcmin)

    if object_size_arcmin is None:
        return {
            "fov_fit": "unknown",
            "fit_ratio": None,
            "object_size_deg": None,
            "note": "No cataloged size for this object \u2014 fit could not be assessed.",
        }

    ratio = object_size_arcmin / frame_min_dim
    object_size_deg = object_size_arcmin / 60

    if ratio > 1.1:
        verdict = "too_large"
        note = (
            f"This object is {object_size_deg:.2f}\u00b0 across \u2014 about {ratio:.1f}\u00d7 wider "
            f"than your {frame_min_dim / 60:.2f}\u00b0 frame. Won't fit in a single frame; "
            "consider a mosaic or a shorter focal length."
        )
    elif ratio < 0.10:
        verdict = "too_small"
        note = (
            f"This object is only {object_size_deg:.2f}\u00b0 across \u2014 about {ratio * 100:.1f}% "
            "of your frame width. It'll appear as a small feature in the frame; consider a "
            "longer focal length or a Barlow/reducer."
        )
    else:
        verdict = "fits_well"
        note = (
            f"This object is {object_size_deg:.2f}\u00b0 across \u2014 a comfortable "
            f"{ratio * 100:.0f}% of your frame width. Well-framed for this setup."
        )

    return {
        "fov_fit": verdict,
        "fit_ratio": round(ratio, 3),
        "object_size_deg": round(object_size_deg, 3),
        "note": note,
    }


def classify_solar_system_target(obj_name: str, arcsec_per_px: float) -> dict:
    """Solar system bodies: judged on pixel-scale adequacy, not frame fit —
    planetary/lunar imaging is high-frame-rate capture + stacking."""
    typical_size_arcsec = SOLAR_SYSTEM_TYPICAL_SIZE_ARCSEC.get(obj_name)
    typical_size_arcmin = typical_size_arcsec / 60 if typical_size_arcsec is not None else None
    sampling = classify_sampling(arcsec_per_px)

    return {
        "fov_fit": "planetary_target",
        "fit_ratio": None,
        "typical_angular_size_arcsec": typical_size_arcsec,
        "typical_angular_size_arcmin": round(typical_size_arcmin, 2) if typical_size_arcmin else None,
        "sampling": sampling,
        "note": (
            "Solar system target \u2014 use high-frame-rate planetary/lunar capture "
            "and stacking rather than single-frame deep-sky FoV framing. "
            f"Current pixel scale is {sampling.replace('_', ' ')} for this technique "
            f"({arcsec_per_px:.2f}\"/px)."
        ),
    }


def get_weekly_fov_analysis(weekly_recommendations: list, user: UserProfile) -> tuple[list, Optional[dict]]:
    """
    Attaches FoV/sampling analysis to every recommended object, every
    night, non-destructively. Returns (weekly, setup_summary);
    setup_summary is None if no camera is on file, and every object is
    tagged fov_fit='no_camera' instead of being dropped.
    """
    if user.camera is None:
        weekly = []
        for night in weekly_recommendations:
            objects = [
                {**obj, "fov_analysis": {
                    "fov_fit": "no_camera",
                    "note": "No camera on file for this user \u2014 add camera specs to enable FoV analysis.",
                }}
                for obj in night["recommended_objects"]
            ]
            weekly.append({**night, "recommended_objects": objects})
        return weekly, None

    fov_w, fov_h = compute_fov_arcmin(
        user.camera.sensor_width_mm, user.camera.sensor_height_mm, user.telescope.focal_length_mm
    )
    arcsec_per_px = compute_pixel_scale_arcsec_per_px(
        user.camera.pixel_size_um, user.telescope.focal_length_mm
    )
    human = fov_arcmin_to_human(fov_w, fov_h)

    setup_summary = {
        "fov_width_arcmin": round(fov_w, 1),
        "fov_height_arcmin": round(fov_h, 1),
        "fov_width_deg": human["fov_width_deg"],
        "fov_height_deg": human["fov_height_deg"],
        "fov_diagonal_arcmin": round((fov_w ** 2 + fov_h ** 2) ** 0.5, 1),
        "pixel_scale_arcsec_per_px": round(arcsec_per_px, 2),
        "sampling": classify_sampling(arcsec_per_px),
        "human_summary": human["human_summary"],
    }

    weekly = []
    for night in weekly_recommendations:
        objects = []
        for obj in night["recommended_objects"]:
            if obj.get("is_solar_system"):
                fov_analysis = classify_solar_system_target(obj["name"], arcsec_per_px)
            else:
                fov_analysis = classify_deep_sky_fit(obj.get("size_arcmin"), fov_w, fov_h)
            objects.append({**obj, "fov_analysis": fov_analysis})
        weekly.append({**night, "recommended_objects": objects})

    return weekly, setup_summary
