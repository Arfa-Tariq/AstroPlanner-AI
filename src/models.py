"""
AstroPlanner — Shared Pydantic Models

This module holds every schema reused across the project's notebooks
(and later, the backend). Import from here rather than redefining these
classes in each notebook, so there is exactly one source of truth for
the data shapes that flow between Weather, Visibility, Recommendation,
FoV, and the LLM chatbot layer.
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
    aperture_mm: float = Field(
        ...,
        description=(
            "Diameter of the telescope's primary lens or mirror, in millimeters. "
            "Determines light-gathering power and resolving ability; larger "
            "apertures reveal fainter objects and finer detail. Realistic amateur "
            "range: 10mm-1000mm."
        ),
        gt=10,
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
        gt=100,
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


class CameraSpec(BaseModel):
    """
    Imaging sensor specifications for the user's astrophotography camera.

    Sensor dimensions and pixel size are required inputs for the Field-of-View
    Analyzer, which determines whether a given celestial target will fit
    perfectly, be too large, or be too small in a single frame.
    """
    sensor_width_mm: float = Field(
        ...,
        description="Physical width of the camera's image sensor, in millimeters. Realistic range: 4mm-50mm.",
        gt=4,
        le=50,
    )
    sensor_height_mm: float = Field(
        ...,
        description="Physical height of the camera's image sensor, in millimeters. Realistic range: 4mm-50mm.",
        gt=4,
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
        gt=1,
        le=15,
    )


class MountSpec(BaseModel):
    """
    Specifications of the telescope mount, which governs tracking accuracy
    and how easily the user can locate and follow targets across the sky.

    goto_capable in particular affects the Recommendation Engine: non-GoTo
    mounts may warrant simpler, easier-to-locate targets for beginners.
    """
    type: str = Field(
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


class UserProfile(BaseModel):
    """
    Complete profile of a single user, combining their location, experience
    level, and equipment. This is the core identity object reused across
    every downstream module: Weather Intelligence (uses location), Celestial
    Visibility Engine (uses location), Recommendation Engine (uses
    experience_level + equipment), and Field-of-View Analyzer (uses telescope
    + camera).
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
    7-day observation plan for a user. There is no date field — the system
    always plans from today through today + 6 days. This is the single
    input object passed into every downstream notebook/module, and later
    becomes the structured input referenced by the LangChain tools and the
    LLM chatbot.
    """
    user: UserProfile = Field(..., description="The requesting user's full profile.")
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
