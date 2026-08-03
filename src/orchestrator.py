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
from langchain_core.runnables import RunnableConfig
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

# thread_id -> {"user_id": ..., "user_name": ...}. Identity is resolved
# ONCE per thread_id inside chat() (from a real Python argument, not from
# conversation text) and looked up here by tools/prompt via RunnableConfig
# — never asked of the LLM. A tool parameter like `user_name` would force
# the model to infer identity from the message text, which it can't do
# reliably (nothing in "what have I planned recently?" says who "I" is),
# and a wrong guess silently creates a phantom user in Supabase with zero
# sessions. This is the fix for exactly that bug.
_thread_identity_cache: dict[str, dict] = {}


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

def _run_pipeline(user: UserProfile, session_id: str) -> dict:
    """
    The actual weather -> visibility -> recommendation -> fov -> scheduler
    pipeline, saving each stage to Supabase under session_id. Extracted
    out of create_observation_plan so revise_observation_plan can call the
    exact same logic against a NEW session that's linked back to an
    existing one via parent_session_id, instead of duplicating the
    5-stage sequence. Returns the data needed for a trimmed chat reply.
    """
    engine = get_or_build_visibility_engine(user.latitude, user.longitude, user.telescope.aperture_mm)

    weekly_weather = weather.get_weekly_sky_conditions(user, date.today())
    storage.save_stage_result(session_id, "weather", weekly_weather)

    weekly_visibility = engine.get_weekly_visibility(date.today())
    storage.save_stage_result(session_id, "visibility", weekly_visibility)

    weekly_recommendations = recommendation.get_weekly_recommendations(
        user, weekly_weather, weekly_visibility, user.bortle_scale, engine.ts, engine.eph,
    )
    storage.save_stage_result(session_id, "recommendation", weekly_recommendations)

    weekly_fov, setup_summary = fov_module.get_weekly_fov_analysis(weekly_recommendations, user)
    storage.save_stage_result(session_id, "fov", {"nights": weekly_fov, "setup_summary": setup_summary})

    weekly_schedule = scheduler.get_weekly_schedule(weekly_fov, max_objects=8)
    storage.save_stage_result(session_id, "schedule", weekly_schedule)

    return {"setup_summary": setup_summary, "weekly_schedule": weekly_schedule}


@tool
def create_observation_plan(
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
    config: RunnableConfig = None,
) -> str:
    """Runs the full observation-planning pipeline (weather, visibility,
    recommendations, field-of-view, and a night-by-night schedule) for a
    location and telescope, and saves every stage as a new Observation
    Session. Use this when the user wants a NEW plan for a location/setup
    not seen before, e.g. "plan my observing session for this week" or
    "what should I look at tonight from [location]". If the user instead
    wants to redo or tweak an EXISTING plan (e.g. "regenerate with fewer
    targets"), use revise_observation_plan instead. Ask for
    latitude/longitude and aperture_mm if not given — do not guess
    coordinates. sensor_* / pixel_size_um are only needed for
    astrophotography; omit them for visual-only observing. Do NOT ask the
    user to identify themselves — identity is resolved automatically."""
    thread_id = config["configurable"]["thread_id"]
    identity = _thread_identity_cache.get(thread_id)
    if identity is None:
        return "No user identity bound to this conversation — this is a setup bug, not something to ask the user about."
    user_id, user_name = identity["user_id"], identity["user_name"]

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

    equipment_id = storage.save_equipment(user_id, user)
    session_id = storage.create_session(
        user_id=user_id, latitude=latitude, longitude=longitude,
        generated_at=date.today().isoformat(), equipment_id=equipment_id, notes=notes,
    )

    pipeline_result = _run_pipeline(user, session_id)

    # Groq's tool-message parser rejects content that isn't a non-empty
    # string (e.g. a raw dict, or "[]") — every tool below returns a JSON
    # string instead of a Python object for this reason, with an explicit
    # fallback message when a result would otherwise be empty.
    return json.dumps({
        "session_id": session_id,
        "setup_summary": pipeline_result["setup_summary"],
        "schedule_preview": _trim_schedule_for_chat(pipeline_result["weekly_schedule"]),
        "note": "Full 7-night data is saved under this session_id — call get_session_context for more.",
    }, default=str)


@tool
def revise_observation_plan(parent_session_id: str, notes: str = None, config: RunnableConfig = None) -> str:
    """Re-runs the observation-planning pipeline for the SAME location and
    equipment as an existing session, saving the result as a new revision
    linked back to it. Use this when the user wants to redo, regenerate,
    or update a plan they already have — e.g. "regenerate tonight's plan"
    or "run that again" — rather than create_observation_plan, which
    would start an unrelated fresh session. Find parent_session_id via
    get_recent_sessions or the session list already in context."""
    thread_id = config["configurable"]["thread_id"]
    identity = _thread_identity_cache.get(thread_id)
    if identity is None:
        return "No user identity bound to this conversation — this is a setup bug, not something to ask the user about."
    user_id, user_name = identity["user_id"], identity["user_name"]

    parent = storage.get_full_session(parent_session_id)
    parent_meta = (parent or {}).get("session")
    if not parent_meta:
        return f"No session found with id {parent_session_id} — cannot create a revision of it."

    equipment = storage.get_equipment(parent_meta["equipment_id"]) if parent_meta.get("equipment_id") else None
    if not equipment:
        return f"Session {parent_session_id} has no equipment on file — cannot revise it. Use create_observation_plan instead."

    user = UserProfile(
        name=user_name,
        latitude=parent_meta["latitude"], longitude=parent_meta["longitude"],
        experience_level=equipment["experience_level"], bortle_scale=equipment.get("bortle_scale"),
        telescope=equipment["telescope"], camera=equipment.get("camera"),
        mount=equipment.get("mount"), preferences=equipment.get("preferences"),
    )

    session_id = storage.create_session(
        user_id=user_id, latitude=user.latitude, longitude=user.longitude,
        generated_at=date.today().isoformat(), equipment_id=parent_meta["equipment_id"],
        notes=notes, parent_session_id=parent_session_id,
    )

    pipeline_result = _run_pipeline(user, session_id)

    return json.dumps({
        "session_id": session_id,
        "revision_of": parent_session_id,
        "setup_summary": pipeline_result["setup_summary"],
        "schedule_preview": _trim_schedule_for_chat(pipeline_result["weekly_schedule"]),
        "note": "This is a new revision linked to the original session — call get_session_context for more.",
    }, default=str)


def _trim_session_for_chat(full_session: dict, max_nights: int = 2, top_n_objects: int = 5) -> dict:
    """
    get_full_session returns EVERYTHING (all 7 nights, every raw field —
    hourly weather, every catalog candidate's ra/dec, etc). That's correct
    for Supabase storage but far too large for a chat turn — one real call
    requested 83k tokens against Groq's per-request limit. Trims to what a
    "why was X recommended" question actually needs: per-night verdict +
    top-N scored objects with their factor breakdown, for the first
    max_nights nights. Full data is still in Supabase if a deeper look is
    ever needed.
    """
    session = full_session.get("session") or {}
    recommendation = full_session.get("recommendation") or []
    fov = (full_session.get("fov") or {}).get("nights") or []
    fov_by_date = {n["date"]: n for n in fov}

    nights_out = []
    for night in recommendation[:max_nights]:
        fov_night = fov_by_date.get(night["date"])
        fov_objects_by_name = (
            {o["name"]: o.get("fov_analysis") for o in fov_night["recommended_objects"]}
            if fov_night else {}
        )

        objects_out = []
        for obj in night["recommended_objects"][:top_n_objects]:
            fov_info = fov_objects_by_name.get(obj["name"], {})
            objects_out.append({
                "name": obj.get("name"),
                "common_name": obj.get("common_name"),
                "target_type": obj.get("target_type"),
                "recommendation_score": obj.get("recommendation_score"),
                "factor_scores": obj.get("factor_scores"),
                "fov_fit": fov_info.get("fov_fit") if fov_info else None,
            })

        nights_out.append({
            "date": night["date"],
            "moon_illumination_pct": night.get("moon_illumination_pct"),
            "sky_quality_verdict": night.get("sky_quality_verdict"),
            "top_recommended_objects": objects_out,
        })

    return {
        "session_id": session.get("id"),
        "latitude": session.get("latitude"),
        "longitude": session.get("longitude"),
        "generated_at": session.get("generated_at"),
        "revision_number": session.get("revision_number"),
        "nights": nights_out,
        "note": f"Showing top {top_n_objects} objects for the first {max_nights} night(s) only — full data is in Supabase under this session_id.",
    }


@tool
def get_session_context(session_id: str) -> str:
    """Retrieves a summary of a PAST observation session by its id:
    per-night sky quality verdict and the top-scored recommended objects
    with their factor breakdown (visibility, weather, moon, equipment,
    light pollution). Use when the user asks about a specific past plan,
    or asks "why" something was or wasn't recommended. Session ids are
    listed in the system context — look there before asking the user."""
    result = storage.get_full_session(session_id)
    if not result or not result.get("session"):
        return f"No session found with id {session_id}."
    return json.dumps(_trim_session_for_chat(result), default=str)


@tool
def get_recent_sessions(limit: int = 5, config: RunnableConfig = None) -> str:
    """Lists the current user's recent observation sessions (id, location,
    date, revision number) without their full data. Use this to find a
    session_id before calling get_session_context, or to answer
    "what have I planned recently" without loading everything."""
    thread_id = config["configurable"]["thread_id"]
    identity = _thread_identity_cache.get(thread_id)
    if identity is None:
        return "No user identity bound to this conversation — this is a setup bug, not something to ask the user about."
    sessions = storage.list_recent_sessions(identity["user_id"], limit=limit)
    if not sessions:
        return "No past sessions found for this user."
    return json.dumps(sessions, default=str)


def build_tools() -> list:
    return [create_observation_plan, revise_observation_plan, get_session_context, get_recent_sessions]


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
    identity = _thread_identity_cache.get(thread_id)
    user_id = identity["user_id"] if identity else None

    base = (
        "You are AstroPlanner, an astronomy observation planning assistant. "
        "You never perform astronomy calculations yourself — always call a "
        "tool for weather, visibility, recommendations, or schedules rather "
        "than estimating or guessing them. If the user wants a brand new "
        "plan, use create_observation_plan. If they want to redo/regenerate "
        "an existing plan, use revise_observation_plan instead, so it's "
        "saved as a linked revision rather than an unrelated new session."
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
    user_name is resolved to a Supabase user_id HERE, from a real Python
    argument — not from the LLM — and cached against thread_id so every
    tool call and dynamic_prompt invocation in this thread can look up
    "who is this" via RunnableConfig instead of asking the model to
    infer or supply it.
    """
    if thread_id not in _thread_identity_cache:
        user_id = storage.get_or_create_user(user_name)
        _thread_identity_cache[thread_id] = {"user_id": user_id, "user_name": user_name}

    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]}, config)
    return result["messages"][-1].content


if __name__ == "__main__":
    print(chat(
        "Plan my observing session for tonight at latitude 33.2, longitude 32.4, "
        "with an 8-inch (200mm) telescope, 1000mm focal length, beginner level.",
        user_name="andrew",
    ))
