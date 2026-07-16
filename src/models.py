"""
AstroPlanner — Shared Pydantic Models

This module holds every schema reused across the project's notebooks
(and later, the backend). Import from here rather than redefining these
classes in each notebook, so there is exactly one source of truth for
the data shapes that flow between Weather, Visibility, Recommendation,
FoV, and the LLM chatbot layer.

v2 additions (kept optional/backward-compatible so existing fixtures
like data/current_request.json and notebooks 02/03 keep working
unchanged):
    - Preferences (observation mode + favorite target categories),
      consumed starting with the Phase 4 Recommendation Engine notebook.
    - TelescopeType / CameraType on the equipment specs, consumed
      starting with the Phase 5 Field-of-View Analyzer notebook.
    - MountSpec.tracking, an additional capability flag alongside the
      existing goto_capable.
"""

from datetime import date as date_type, timedelta
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ExperienceLevel(str, Enum):
    """
    Self-reported astronomy experience level of the user.

    Used downstream by the Recommendation Engine to adjust target difficulty
    (e.g. beginners get bright, easy Messier objects; advanced users get
    faint NGC objects or narrowband targets) and to tune how much explanation
    the LLM assistant includes in its responses.
    """
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class ObservationMode(str, Enum):
    """
    Whether the user is primarily observing visually (through the eyepiece)
    or imaging (astrophotography). Used by the Recommendation Engine to
    weight targets differently — e.g. visual observers favor bright,
    high-contrast targets; imagers favor targets that suit their FoV and
    can tolerate longer integration times.
    """
    visual = "visual"
    astrophotography = "astrophotography"


class TargetType(str, Enum):
    """
    Broad celestial object categories a user can mark as a favorite.
    Used by the Recommendation Engine (Phase 4) to boost or filter
    candidate targets according to user interest, on top of the
    visibility/equipment/weather-based ranking.
    """
    planet = "planet"
    moon = "moon"
    galaxy = "galaxy"
    nebula = "nebula"
    open_cluster = "open_cluster"
    globular_cluster = "globular_cluster"
    double_star = "double_star"


class Preferences(BaseModel):
    """
    Soft preferences layered on top of the user's hard equipment/location
    constraints. Not consumed by the Weather Intelligence (02) or
    Celestial Visibility (03) notebooks — those only need location and
    aperture. Reserved for the Recommendation Engine (Phase 4), which
    ranks/filters targets according to mode and favorite_targets.
    """
    mode: ObservationMode = Field(
        ...,
        description=(
            "Primary use case for this session: visual observing or "
            "astrophotography. Drives how the Recommendation Engine "
            "weights target brightness vs. imaging suitability."
        ),
    )
    favorite_targets: list[TargetType] = Field(
        default_factory=list,
        description=(
            "Celestial object categories the user is most interested in. "
            "Empty list means no preference — all categories are weighted "
            "equally by the Recommendation Engine."
        ),
    )


class TelescopeType(str, Enum):
    """
    Optical design of the telescope. Reserved for the Field-of-View
    Analyzer (Phase 5), where design affects central obstruction and
    illumination/vignetting across the sensor, and for the Recommendation
    Engine, where design affects suitability for planetary vs. deep-sky
    targets (e.g. long-focal-ratio refractors vs. fast reflectors).
    """
    refractor = "refractor"
    reflector = "reflector"
    catadioptric = "catadioptric"


class TelescopeSpec(BaseModel):
    """
    Optical specifications of the user's telescope.

    These values drive two downstream calculations:
    1. Field-of-View Analyzer — focal_length_mm combined with the camera's
       sensor dimensions determines the imaging FoV in arcminutes.
    2. Limiting magnitude / resolving power estimates — aperture_mm determines
       how faint an object the telescope can usefully observe, which feeds
       into the Recommendation Engine's target ranking.
    """
    type: Optional[TelescopeType] = Field(
        default=None,
        description=(
            "Optical design: 'refractor', 'reflector', or 'catadioptric'. "
            "Optional — left None if the user doesn't know or it isn't "
            "collected yet. Not currently consumed by the Weather or "
            "Visibility notebooks; reserved for the FoV Analyzer and "
            "Recommendation Engine."
        ),
    )
    aperture_mm: float = Field(
        ...,
        description=(
            "Diameter of the telescope's primary lens or mirror, in millimeters. "
            "Determines light-gathering power and resolving ability; larger "
            "apertures reveal fainter objects and finer detail. Realistic amateur "
            "range: 10mm-1000mm."
        ),
        ge=10,
        le=1000,
    )
    focal_length_mm: float = Field(
        ...,
        description=(
            "Distance from the primary optic to the focal point, in millimeters. "
            "Used together with camera sensor size to compute imaging field of view, "
            "and together with aperture to derive focal_ratio. Realistic amateur "
            "range: 100mm-5000mm."
        ),
        ge=100,
        le=5000,
    )
    focal_ratio: Optional[float] = Field(
        default=None,
        description=(
            "Focal ratio (f/number) of the optical system, calculated as "
            "focal_length_mm / aperture_mm if not provided directly. Lower "
            "values (e.g. f/4) are 'faster' and better suited to wide-field "
            "deep-sky imaging; higher values (e.g. f/10) are 'slower' and "
            "better suited to high-magnification planetary/lunar imaging."
        ),
    )


class CameraType(str, Enum):
    """
    Camera hardware category. Reserved for the FoV Analyzer and
    astrophotography acquisition recommendations (e.g. dedicated astro
    cameras support cooling/gain settings that DSLRs/mirrorless bodies
    don't expose the same way).
    """
    dslr = "dslr"
    mirrorless = "mirrorless"
    dedicated_astro = "dedicated_astro"


class CameraSpec(BaseModel):
    """
    Imaging sensor specifications for the user's astrophotography camera.

    Sensor dimensions and pixel size are required inputs for the Field-of-View
    Analyzer, which determines whether a given celestial target will fit
    perfectly, be too large, or be too small in a single frame.
    """
    type: Optional[CameraType] = Field(
        default=None,
        description=(
            "Camera hardware category: 'dslr', 'mirrorless', or "
            "'dedicated_astro'. Optional — left None if not collected yet. "
            "Reserved for the FoV Analyzer and acquisition recommendations."
        ),
    )
    sensor_width_mm: float = Field(
        ...,
        description="Physical width of the camera's image sensor, in millimeters. Realistic range: 4mm-50mm.",
        ge=4,
        le=50,
    )
    sensor_height_mm: float = Field(
        ...,
        description="Physical height of the camera's image sensor, in millimeters. Realistic range: 4mm-50mm.",
        ge=4,
        le=50,
    )
    pixel_size_um: float = Field(
        ...,
        description=(
            "Size of a single pixel on the sensor, in micrometers. Used to "
            "calculate image scale (arcseconds per pixel) together with the "
            "telescope's focal length, which affects whether the setup is "
            "well-matched ('oversampled' or 'undersampled') for a given target. "
            "Realistic range: 1µm-15µm."
        ),
        ge=1,
        le=15,
    )


class MountType(str, Enum):
    """
    Mount type. Kept as an enum (rather than a free string) so invalid
    values are rejected at the schema layer instead of downstream.
    Values are unchanged from the original string field, so existing
    fixtures with type: "alt-az" / "equatorial" continue to validate.
    """
    alt_az = "alt-az"
    equatorial = "equatorial"


class MountSpec(BaseModel):
    """
    Specifications of the telescope mount, which governs tracking accuracy
    and how easily the user can locate and follow targets across the sky.

    goto_capable in particular affects the Recommendation Engine: non-GoTo
    mounts may warrant simpler, easier-to-locate targets for beginners.
    """
    type: MountType = Field(
        ...,
        description=(
            "Mount type: 'alt-az' or 'equatorial'. Equatorial mounts can "
            "track the sky's rotation along a single axis, which matters for "
            "long-exposure astrophotography; alt-az mounts are simpler but "
            "introduce field rotation during long exposures."
        ),
    )
    goto_capable: bool = Field(
        ...,
        description=(
            "Whether the mount can automatically slew to and track a chosen "
            "target. Affects how detailed the Observation Scheduler's pointing "
            "instructions need to be."
        ),
    )
    tracking: Optional[bool] = Field(
        default=None,
        description=(
            "Whether the mount has motorized sidereal tracking at all "
            "(independent of goto_capable — a mount can track without being "
            "goto-capable). Optional — left None if not collected yet. "
            "Relevant for astrophotography exposure-length recommendations."
        ),
    )


class UserProfile(BaseModel):
    """
    Complete profile of a single user, combining their location, experience
    level, and equipment. This is the core identity object reused across
    every downstream module: Weather Intelligence (uses location), Celestial
    Visibility Engine (uses location), Recommendation Engine (uses
    experience_level + equipment + preferences), and Field-of-View Analyzer
    (uses telescope + camera).
    """
    name: str = Field(..., description="Display name of the user.")
    latitude: float = Field(
        ...,
        description=(
            "Observing site latitude in decimal degrees (positive = North, "
            "negative = South). Required for computing object rise/set times, "
            "altitude/azimuth, and for fetching local weather data."
        ),
        ge=-90,
        le=90,
    )
    longitude: float = Field(
        ...,
        description=(
            "Observing site longitude in decimal degrees (positive = East, "
            "negative = West). Required alongside latitude for all astronomical "
            "and weather calculations."
        ),
        ge=-180,
        le=180,
    )
    experience_level: ExperienceLevel = Field(
        ...,
        description="See ExperienceLevel; used to tailor target difficulty and explanation depth.",
    )
    bortle_scale: Optional[int] = Field(
        default=None,
        description=(
            "Self-reported Bortle dark-sky scale for the observing site, "
            "1 (excellent dark sky) to 9 (inner-city sky). A static site "
            "property, like latitude/longitude, rather than a nightly "
            "value — collected once here instead of fetched per-night. "
            "Optional; if None, the Recommendation Engine (Phase 4) falls "
            "back to an automatic lat/lon-based lookup, and drops the "
            "light-pollution factor entirely if that also fails."
        ),
        ge=1,
        le=9,
    )
    preferences: Optional[Preferences] = Field(
        default=None,
        description=(
            "Optional soft preferences (observation mode + favorite target "
            "categories). Left None if not collected. Not consumed by "
            "Weather Intelligence or Celestial Visibility — reserved for "
            "the Recommendation Engine."
        ),
    )
    telescope: TelescopeSpec = Field(
        ..., description="The user's primary telescope specifications."
    )
    camera: Optional[CameraSpec] = Field(
        default=None,
        description=(
            "The user's astrophotography camera, if any. Optional — left to "
            "the user whether to provide it. If absent, downstream modules "
            "(e.g. Field-of-View Analyzer) skip imaging-specific calculations "
            "and fall back to visual-observing recommendations only."
        ),
    )
    mount: Optional[MountSpec] = Field(
        default=None,
        description=(
            "The user's mount specifications, if known. Optional — left to "
            "the user whether to provide it."
        ),
    )


class WeeklyPlanRequest(BaseModel):
    """
    Top-level request object representing a request to generate a full
    7-day observation plan for a user, covering generated_at through
    generated_at + 6 days.
    This is the single input object passed into every downstream
    notebook/module, and later becomes the structured input referenced
    by the LangChain tools and the LLM chatbot.
    """
    user: UserProfile = Field(..., description="The requesting user's full profile.")
    generated_at: date_type = Field(
        default_factory=date_type.today,
        description=(
            "The date this plan's 7-day window starts from (today through "
            "today + 6 days), fixed once at creation time in notebook 01 "
            "and persisted in current_request.json. Every downstream "
            "notebook (02, 03, 04, ...) reads THIS value instead of "
            "independently calling today() — otherwise notebooks run on "
            "different calendar days end up with 7-day windows that don't "
            "fully overlap, silently dropping nights where they disagree. "
            "Only recomputed if this field is genuinely absent from the "
            "loaded JSON (e.g. an old pre-v2 fixture)."
        ),
    )
    notes: Optional[str] = Field(
        default=None,
        description=(
            "Freeform notes from the user, e.g. 'want to image a galaxy' or "
            "'first time using this telescope'. Passed to the LLM chatbot as "
            "conversational context — sanitized at collection time and "
            "wrapped in explicit delimiters at prompt-construction time "
            "(notebook 06) to guard against prompt injection."
        ),
        max_length=300,
    )


# Backward-compatible alias — earlier notebooks/snippets may reference
# ObservationRequest; it now means the same thing as WeeklyPlanRequest.
ObservationRequest = WeeklyPlanRequest
