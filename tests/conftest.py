"""
Shared fixtures for the pure-function test suite (fov.py, scheduler.py,
recommendation.py). These tests need src/ on sys.path but need NOTHING
else — no Groq, no Supabase, no network, no ephemeris download. Run them
first, before touching chat_cli.py or the live orchestrator, since they
isolate "is the math right" from "does the whole stack talk to itself".
"""

import os
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pytest

from models import UserProfile, TelescopeSpec, CameraSpec, MountSpec


@pytest.fixture
def user_no_camera():
    return UserProfile(
        name="Test",
        latitude=33.2,
        longitude=32.4,
        experience_level="beginner",
        bortle_scale=6,
        telescope=TelescopeSpec(aperture_mm=200, focal_length_mm=1000),
    )


@pytest.fixture
def user_with_camera():
    return UserProfile(
        name="Test",
        latitude=33.2,
        longitude=32.4,
        experience_level="intermediate",
        bortle_scale=4,
        telescope=TelescopeSpec(aperture_mm=200, focal_length_mm=1000),
        camera=CameraSpec(sensor_width_mm=23.5, sensor_height_mm=15.6, pixel_size_um=3.76),
        mount=MountSpec(type="equatorial", goto_capable=True, tracking=True),
    )
