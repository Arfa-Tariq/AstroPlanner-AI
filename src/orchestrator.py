"""
AstroPlanner — Orchestrator (LangGraph version)

Replaces the hand-rolled while-loop from orchestrator.py v0/v1 with
LangGraph's prebuilt create_react_agent. Conceptually identical to what
you already built and tested — same idea: user message -> model decides
tool or answer -> if tool, run it, feed result back -> repeat. LangGraph
just implements that loop for you, as a compiled graph, and adds
conversation memory (via a checkpointer) for free.

What's genuinely new to learn here, vs. the raw version:
  - @tool decorator: turns a plain function into something the agent can
    call, using its docstring as the description the model sees (same
    role as the JSON schemas we wrote by hand before).
  - create_react_agent: builds the whole graph (call model -> check for
    tool calls -> run tools -> loop) in one line.
  - MemorySaver + thread_id: LangGraph's answer to "remember this
    conversation" — a thread_id is like a conversation/session key. Each
    distinct thread_id gets its own remembered message history.

weather.py, visibility.py, and storage.py are UNCHANGED — this file only
wraps them differently. That's the payoff of having kept them
framework-agnostic from the start.
"""

import os
from datetime import date

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from weather import get_weekly_sky_conditions
from visibility import VisibilityEngine
from models import UserProfile
import storage

storage.init_db()

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.environ["GROQ_API_KEY"])

DEFAULT_USER_NAME = "chat_user"
_visibility_engine_cache: dict = {}
_current_session_id: "int | None" = None


def get_or_create_session(latitude: float, longitude: float, aperture_mm: float = None) -> int:
    global _current_session_id
    if _current_session_id is None:
        _current_session_id = storage.create_session(DEFAULT_USER_NAME, latitude, longitude, aperture_mm)
    return _current_session_id


def get_or_build_visibility_engine(latitude: float, longitude: float, aperture_mm: float) -> VisibilityEngine:
    key = (latitude, longitude, aperture_mm)
    if key not in _visibility_engine_cache:
        _visibility_engine_cache[key] = VisibilityEngine(latitude, longitude, aperture_mm)
    return _visibility_engine_cache[key]


def _trim_visibility_for_chat(weekly: list, max_objects_per_night: int = 8) -> list:
    """Same trimming logic as before — still needed, LangChain doesn't do this for you."""
    trimmed = []
    for night in weekly:
        kept_objects = [
            {
                "name": obj.get("name"),
                "common_name": obj.get("common_name"),
                "type": obj.get("type"),
                "magnitude": obj.get("magnitude"),
                "peak_altitude_deg": obj.get("peak_altitude_deg"),
                "peak_or_transit_time": obj.get("peak_time_local") or obj.get("transit_time"),
                "rise": obj.get("rise_time") or obj.get("above_30deg_from_local"),
                "set": obj.get("set_time") or obj.get("above_30deg_until_local"),
                "is_solar_system": obj.get("is_solar_system"),
            }
            for obj in night["objects"][:max_objects_per_night]
        ]
        trimmed.append({
            "date": night["date"], "day_offset": night["day_offset"],
            "total_visible_object_count": night["visible_object_count"],
            "objects_shown": len(kept_objects), "objects": kept_objects,
        })
    return trimmed


# ---------------------------------------------------------------------
# Tools. The @tool decorator reads the function's type hints (for the
# parameter schema) and its docstring (for the description) — this
# REPLACES the JSON schema dicts we wrote by hand in the raw version.
# Write the docstring like you're still explaining it to the model.
# ---------------------------------------------------------------------

@tool
def weather_tool(latitude: float, longitude: float) -> list:
    """Gets a 7-night weather and sky-quality outlook (cloud cover, wind,
    humidity, seeing/transparency where available) for a specific
    latitude/longitude, starting today. Use for questions about weather,
    sky conditions, or whether tonight/this week is good for observing."""
    session_id = get_or_create_session(latitude, longitude)
    stub_user = UserProfile(
        name="chat_user", latitude=latitude, longitude=longitude,
        experience_level="beginner",
        telescope={"aperture_mm": 100, "focal_length_mm": 500},
    )
    result = get_weekly_sky_conditions(stub_user, date.today())
    storage.save_result(session_id, "weather_tool", result)
    return result


@tool
def visibility_tool(latitude: float, longitude: float, aperture_mm: float) -> list:
    """Gets celestial objects (planets, Moon, deep-sky objects) observable
    over the next 7 nights from a location with a telescope of a given
    aperture, including rise/transit/set times and peak altitude. Use for
    questions about what's visible or observable, or object rise/set times.
    Ask the user for aperture_mm if unknown — do not guess it silently."""
    session_id = get_or_create_session(latitude, longitude, aperture_mm)
    engine = get_or_build_visibility_engine(latitude, longitude, aperture_mm)
    full_result = engine.get_weekly_visibility(date.today())
    trimmed = _trim_visibility_for_chat(full_result)
    storage.save_result(session_id, "visibility_tool", trimmed)
    return trimmed


@tool
def get_past_session_results(session_id: int) -> list:
    """Retrieves saved results from a PAST observation session by its id,
    when the user refers to a previous plan/night/session not already
    visible earlier in this conversation. Session ids are listed in the
    system prompt — look there before asking the user for one."""
    return storage.get_session_results(session_id)


def build_system_prompt() -> str:
    """Same 'context builder' idea as before: cheap session metadata always
    present, full data fetched only on demand via get_past_session_results."""
    recent = storage.list_recent_sessions(DEFAULT_USER_NAME, limit=5)
    if not recent:
        return "You are AstroPlanner, an astronomy observation planning assistant."
    lines = [f"- session_id={s['id']}, {str(s['created_at'])[:10]}, lat={s['latitude']}, lon={s['longitude']}" for s in recent]
    return (
        "You are AstroPlanner, an astronomy observation planning assistant.\n"
        "The user has these past sessions available (call get_past_session_results "
        "only if they ask about one — don't mention them unprompted):\n" + "\n".join(lines)
    )


# ---------------------------------------------------------------------
# The agent itself. This ONE call replaces the entire hand-rolled
# while-loop from orchestrator.py. checkpointer=MemorySaver() means the
# graph remembers messages per thread_id automatically — you don't
# manage a `history` list by hand anymore.
# ---------------------------------------------------------------------
agent = create_react_agent(
    model=llm,
    tools=[weather_tool, visibility_tool, get_past_session_results],
    prompt=build_system_prompt(),
    checkpointer=MemorySaver(),
)


def chat(user_message: str, thread_id: str = "default") -> str:
    """
    Run one turn. thread_id identifies the conversation — same thread_id
    across calls = the agent remembers prior turns; a new thread_id =
    a fresh conversation (though get_past_session_results can still pull
    up old DB sessions regardless of thread).
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]}, config)
    return result["messages"][-1].content


if __name__ == "__main__":
    print(chat("How does the sky look this week at latitude 33.2, longitude 32.4?"))
