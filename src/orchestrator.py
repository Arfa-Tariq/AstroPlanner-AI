"""
AstroPlanner — Orchestrator (LangGraph + Supabase)

Two fixes over the previous version, worth understanding rather than
just diffing:

1. STALE PROMPT BUG: the old code built the system prompt once, at
   import time (`prompt=build_system_prompt()`), before any session
   existed. LangGraph's create_react_agent lets `prompt` be a CALLABLE
   instead of a fixed string — `dynamic_prompt(state, config)` below is
   invoked fresh on every turn, so "recent sessions" always reflects
   what's actually in Supabase right now. This callable IS the context
   builder from the architecture doc — it's not a separate graph node
   because nothing else needs to happen between "message arrives" and
   "agent runs" for this app's scope.

2. PIPELINE AS ONE TOOL: rather than five separate stage tools the LLM
   would have to call in the right order (risk: it calls fov before
   recommendation, or forgets scheduler), `create_observation_plan` runs
   weather -> visibility -> recommendation -> fov -> scheduler
   server-side as one unit and persists every stage to Supabase. This
   matches the guide's own pipeline diagram, where "Pipeline Executes"
   is one block from the user's perspective. Narrower follow-up
   questions get their own tools instead of re-running everything.
"""

import os
import json
from datetime import date

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

import weather
import recommendation
import fov as fov_module
import scheduler
from visibility import VisibilityEngine
from models import UserProfile
import storage

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.environ["GROQ_API_KEY"])

DATA_DIR = os.environ.get("ASTROPLANNER_DATA_DIR", "./data")
os.makedirs(DATA_DIR, exist_ok=True)

# Caches — same "build once per conversation, not once per call" idea as
# VisibilityEngine's own internal ephemeris/catalog caching. Keyed so
# multiple users/locations in the same process don't collide or
# needlessly re-download the NGC catalog.
_visibility_engine_cache: dict = {}

# thread_id -> user_id. A real app would resolve this from an auth
# session; here chat() takes user_name explicitly and this dict is what
# lets the dynamic_prompt callable (which only receives thread_id via
# LangGraph's config) look up which user it's building context for.
_thread_user_cache: dict[str, str] = {}


def get_or_build_visibility_engine(latitude: float, longitude: float, aperture_mm: float) -> VisibilityEngine:
    key = (latitude, longitude, aperture_mm)
    if key not in _visibility_engine_cache:
        _visibility_engine_cache[key] = VisibilityEngine(latitude, longitude, aperture_mm, data_dir=DATA_DIR)
    return _visibility_engine_cache[key]


def _trim_schedule_for_chat(weekly_schedule: list, max_nights: int = 1) -> list:
    """
    Schedules are already compact, but Groq's free tier caps requests at
    12,000 tokens/minute — small enough that even 3 nights with prose
    'note' fields can blow past it, especially once conversation history
    accumulates across turns. Trimmed hard: 1 night, and the long-form
    'note' string dropped from each slot (full data is always still in
    Supabase via session_id — call get_session_context for it).
    """
    trimmed = []
    for night in weekly_schedule[:max_nights]:
        def strip_note(slot):
            return {k: v for k, v in slot.items() if k != "note"}
        trimmed.append({
            "date": night["date"],
            "timeline": [strip_note(s) for s in night["timeline"]],
            "daytime_bonus": [strip_note(s) for s in night["daytime_bonus"]],
        })
    return trimmed


# ---------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------

@tool
def create_observation_plan(
    user_name: str,
    latitude: float,
    longitude: float,
    aperture_mm: float,
    focal_length_mm: float,
    experience_level: str = "beginner",
    sensor_width_mm: float = None,
    sensor_height_mm: float = None,
    pixel_size_um: float = None,
    bortle_scale: int = None,
    notes: str = None,
) -> str:
    """Runs the full observation-planning pipeline (weather, visibility,
    recommendations, field-of-view, and a night-by-night schedule) for a
    location and telescope, and saves every stage as a new Observation
    Session. Use this when the user wants a new plan, e.g. "plan my
    observing session for this week" or "what should I look at tonight
    from [location]". Ask for latitude/longitude and aperture_mm if not
    given — do not guess coordinates. sensor_* / pixel_size_um are only
    needed if the user has a camera for astrophotography; omit them for
    visual-only observing."""
    stub_camera = None
    if sensor_width_mm and sensor_height_mm and pixel_size_um:
        stub_camera = {
            "sensor_width_mm": sensor_width_mm,
            "sensor_height_mm": sensor_height_mm,
            "pixel_size_um": pixel_size_um,
        }

    user = UserProfile(
        name=user_name, latitude=latitude, longitude=longitude,
        experience_level=experience_level, bortle_scale=bortle_scale,
        telescope={"aperture_mm": aperture_mm, "focal_length_mm": focal_length_mm},
        camera=stub_camera,
    )

    user_id = storage.get_or_create_user(user_name)
    equipment_id = storage.save_equipment(user_id, user)
    session_id = storage.create_session(
        user_id=user_id, latitude=latitude, longitude=longitude,
        generated_at=date.today().isoformat(), equipment_id=equipment_id, notes=notes,
    )

    engine = get_or_build_visibility_engine(latitude, longitude, aperture_mm)

    weekly_weather = weather.get_weekly_sky_conditions(user, date.today())
    storage.save_stage_result(session_id, "weather", weekly_weather)

    weekly_visibility = engine.get_weekly_visibility(date.today())
    storage.save_stage_result(session_id, "visibility", weekly_visibility)

    weekly_recommendations = recommendation.get_weekly_recommendations(
        user, weekly_weather, weekly_visibility, bortle_scale, engine.ts, engine.eph,
    )
    storage.save_stage_result(session_id, "recommendation", weekly_recommendations)

    weekly_fov, setup_summary = fov_module.get_weekly_fov_analysis(weekly_recommendations, user)
    storage.save_stage_result(session_id, "fov", {"nights": weekly_fov, "setup_summary": setup_summary})

    weekly_schedule = scheduler.get_weekly_schedule(weekly_fov)
    storage.save_stage_result(session_id, "schedule", weekly_schedule)

    # Groq's tool-message parser rejects content that isn't a non-empty
    # string (e.g. a raw dict, or "[]") — every tool below returns a JSON
    # string instead of a Python object for this reason, with an explicit
    # fallback message when a result would otherwise be empty.
    return json.dumps({
        "session_id": session_id,
        "setup_summary": setup_summary,
        "schedule_preview": _trim_schedule_for_chat(weekly_schedule),
        "note": "Full 7-night data is saved under this session_id — call get_session_context for more.",
    }, default=str)


@tool
def get_session_context(session_id: str) -> str:
    """Retrieves everything saved for a PAST observation session by its
    id: weather, visibility, recommendations, field-of-view analysis, and
    schedule. Use when the user asks about a specific past plan, or asks
    "why" something was or wasn't recommended and the answer requires
    looking at saved reasoning rather than guessing. Session ids are
    listed in the system context — look there before asking the user."""
    result = storage.get_full_session(session_id)
    if not result or not result.get("session"):
        return f"No session found with id {session_id}."
    return json.dumps(result, default=str)


@tool
def get_recent_sessions(user_name: str, limit: int = 5) -> str:
    """Lists a user's recent observation sessions (id, location, date,
    revision number) without their full data. Use this to find a
    session_id before calling get_session_context, or to answer
    "what have I planned recently" without loading everything."""
    user_id = storage.get_or_create_user(user_name)
    sessions = storage.list_recent_sessions(user_id, limit=limit)
    if not sessions:
        return "No past sessions found for this user."
    return json.dumps(sessions, default=str)


def build_tools() -> list:
    return [create_observation_plan, get_session_context, get_recent_sessions]


# ---------------------------------------------------------------------
# Dynamic context builder — see module docstring, fix #1
# ---------------------------------------------------------------------

def dynamic_prompt(state, config) -> list:
    """
    Called by create_react_agent on every turn (not once at startup).
    Builds a minimal system message: a short list of the user's recent
    sessions, cheap enough to include always, with full data one tool
    call away via get_session_context. This is the "context should
    always be minimal and relevant" principle from the architecture doc,
    implemented as a prompt function instead of a separate graph node.
    """
    thread_id = config.get("configurable", {}).get("thread_id")
    user_id = _thread_user_cache.get(thread_id)

    base = (
        "You are AstroPlanner, an astronomy observation planning assistant. "
        "You never perform astronomy calculations yourself — always call a "
        "tool for weather, visibility, recommendations, or schedules rather "
        "than estimating or guessing them."
    )

    if user_id:
        recent = storage.list_recent_sessions(user_id, limit=5)
        if recent:
            lines = [
                f"- session_id={s['id']}, {s['generated_at']}, "
                f"lat={s['latitude']}, lon={s['longitude']}, rev={s['revision_number']}"
                for s in recent
            ]
            base += (
                "\n\nThis user has these recent sessions (call get_session_context "
                "only if the user asks about one — don't mention them unprompted):\n"
                + "\n".join(lines)
            )

    return [SystemMessage(content=base)] + state["messages"]


agent = create_react_agent(
    model=llm,
    tools=build_tools(),
    prompt=dynamic_prompt,
    checkpointer=MemorySaver(),
)


def chat(user_message: str, user_name: str = "chat_user", thread_id: str = "default") -> str:
    """
    Runs one turn. thread_id identifies the conversation for LangGraph's
    checkpointer (same thread_id = agent remembers prior turns).
    user_name resolves to a Supabase user_id, cached against thread_id so
    dynamic_prompt (which only sees thread_id via config) can look up the
    right user's recent sessions.
    """
    if thread_id not in _thread_user_cache:
        _thread_user_cache[thread_id] = storage.get_or_create_user(user_name)

    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]}, config)
    return result["messages"][-1].content


if __name__ == "__main__":
    print(chat(
        "Plan my observing session for tonight at latitude 33.2, longitude 32.4, "
        "with an 8-inch (200mm) telescope, 1000mm focal length, beginner level.",
        user_name="andrew",
    ))
