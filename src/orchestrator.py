"""
AstroPlanner — Orchestrator (v1: two tools, Groq)

This is the "Conversation Agent" from the architecture doc, built up
incrementally. Currently wired to 2 tools: weather and visibility.
Adding the remaining three (recommendation, fov, scheduler) is the same
3-step pattern each time — describe it in TOOLS, write a small adapter
function, add one elif line in run_agent_turn. The loop itself
(run_agent_turn) does not change as tools are added.

Requires: pip install groq
Requires: an env var GROQ_API_KEY (get one free at console.groq.com)
"""

 
import json
import os
 
from groq import Groq
 
from weather import get_weekly_sky_conditions
from visibility import VisibilityEngine
from models import UserProfile
 
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"  # supports tool calling on Groq
 
# ---------------------------------------------------------------------
# VisibilityEngine cache — this is the "build once, reuse" idea from
# visibility.py made real. Keyed by (lat, lon, aperture) so a second
# question about the SAME setup reuses the already-loaded catalog and
# ephemeris instead of redownloading them. A real session system will
# replace this dict later; for now it's enough to prove the pattern.
# ---------------------------------------------------------------------
_visibility_engine_cache: dict = {}
 
 
def get_or_build_visibility_engine(latitude: float, longitude: float, aperture_mm: float) -> VisibilityEngine:
    key = (latitude, longitude, aperture_mm)
    if key not in _visibility_engine_cache:
        _visibility_engine_cache[key] = VisibilityEngine(latitude, longitude, aperture_mm)
    return _visibility_engine_cache[key]
 
 
# ---------------------------------------------------------------------
# Step 1: Describe the tool to the LLM.
#
# This is JSON, not Python — it's the ONLY thing the model ever sees
# about get_weekly_sky_conditions. It cannot see your source code. If
# the description or parameter docs are vague, the model will guess
# wrong about when/how to call it. Be as explicit as you'd be
# explaining it to a new teammate.
# ---------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weekly_sky_conditions",
            "description": (
                "Gets a 7-night weather and sky-quality outlook (cloud cover, "
                "wind, humidity, seeing/transparency where available) for a "
                "specific latitude/longitude, starting today. Use this whenever "
                "the user asks about weather, sky conditions, or whether "
                "tonight/this week is good for observing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "Observing site latitude in decimal degrees.",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Observing site longitude in decimal degrees.",
                    },
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weekly_visibility",
            "description": (
                "Gets the list of celestial objects (planets, Moon, deep-sky "
                "objects like galaxies/nebulae/clusters) observable over the "
                "next 7 nights from a specific location with a telescope of a "
                "given aperture. Includes rise/transit/set times and peak "
                "altitude. Use this when the user asks what's visible, what "
                "they can observe, or wants object rise/set times."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "Observing site latitude in decimal degrees."},
                    "longitude": {"type": "number", "description": "Observing site longitude in decimal degrees."},
                    "aperture_mm": {
                        "type": "number",
                        "description": (
                            "Telescope aperture in millimeters. Determines how faint an "
                            "object can be seen. Ask the user if unknown; do not guess silently."
                        ),
                    },
                },
                "required": ["latitude", "longitude", "aperture_mm"],
            },
        },
    },
]
 
# Dispatch from tool name (a string the model sends) -> the real adapter
# function to run. See the if/elif chain inside run_agent_turn — that's
# where this mapping is actually used. Adding tool #3 later means: one
# new entry in TOOLS above, one new adapter function, one new elif line.
 
 
def call_weather_tool(latitude: float, longitude: float) -> list:
    """
    Thin adapter: the notebook function wants a UserProfile + start_date,
    but the LLM only knows lat/lon (see the schema above — deliberately
    minimal). This function bridges that gap so weather.py itself never
    has to know anything about how it's being invoked.
    """
    from datetime import date
 
    # Minimal throwaway profile — only lat/lon are used by this function.
    stub_user = UserProfile(
        name="chat_user",
        latitude=latitude,
        longitude=longitude,
        experience_level="beginner",
        telescope={"aperture_mm": 100, "focal_length_mm": 500},
    )
    return get_weekly_sky_conditions(stub_user, date.today())
 
 
def _trim_visibility_for_chat(weekly: list, max_objects_per_night: int = 8) -> list:
    """
    The full weekly_visibility output can easily be 100+ objects across 7
    nights, each with a dozen fields (RA/Dec, moon separation, etc). That's
    correct for internal use but far too large to hand an LLM — it blows
    past free-tier token-per-minute limits immediately. The model only
    needs enough to talk about the sky sensibly, not your full dataset.
    Keeps: name, type, peak altitude, transit/peak time, rise/set-ish
    fields, magnitude. Drops: RA/Dec, moon_separation_deg, size_arcmin,
    and caps each night to the top N objects (already sorted by
    brightness/altitude upstream, so top N = the best ones).
    """
    trimmed = []
    for night in weekly:
        kept_objects = []
        for obj in night["objects"][:max_objects_per_night]:
            kept_objects.append({
                "name": obj.get("name"),
                "common_name": obj.get("common_name"),
                "type": obj.get("type"),
                "magnitude": obj.get("magnitude"),
                "peak_altitude_deg": obj.get("peak_altitude_deg"),
                "peak_or_transit_time": obj.get("peak_time_local") or obj.get("transit_time"),
                "rise": obj.get("rise_time") or obj.get("above_30deg_from_local"),
                "set": obj.get("set_time") or obj.get("above_30deg_until_local"),
                "is_solar_system": obj.get("is_solar_system"),
            })
        trimmed.append({
            "date": night["date"],
            "day_offset": night["day_offset"],
            "total_visible_object_count": night["visible_object_count"],
            "objects_shown": len(kept_objects),
            "objects": kept_objects,
        })
    return trimmed
 
 
def call_visibility_tool(latitude: float, longitude: float, aperture_mm: float) -> list:
    """
    Adapter for the visibility tool. Unlike call_weather_tool, this doesn't
    rebuild everything each time — it fetches (or builds, first time only)
    a cached VisibilityEngine for this exact (lat, lon, aperture)
    combination, then just asks it for this week's visibility.
 
    Trims the result before returning — see _trim_visibility_for_chat.
    """
    from datetime import date
    engine = get_or_build_visibility_engine(latitude, longitude, aperture_mm)
    full_result = engine.get_weekly_visibility(date.today())
    return _trim_visibility_for_chat(full_result)
 
 
def run_agent_turn(user_message: str, history: list = None) -> tuple[str, list]:
    """
    One full turn: send the user's message (+ prior history) to Groq,
    execute any tool calls it asks for, feed results back, and loop
    until it gives a plain-text answer. Returns (reply_text, updated_history)
    so the caller can keep chatting across turns.
    """
    messages = (history or []) + [{"role": "user", "content": user_message}]
 
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        reply = response.choices[0].message
 
        # Case A: the model wants to call one or more tools.
        if reply.tool_calls:
            # Record the model's tool-call request in the conversation...
            messages.append(reply)
 
            # ...then execute each requested call and append its result.
            for tool_call in reply.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
 
                if fn_name == "get_weekly_sky_conditions":
                    result = call_weather_tool(**fn_args)
                elif fn_name == "get_weekly_visibility":
                    result = call_visibility_tool(**fn_args)
                else:
                    result = {"error": f"Unknown tool: {fn_name}"}
 
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, default=str),
                })
 
            # Loop back around: send the tool result back to the model
            # so it can either call another tool or write a final answer.
            continue
 
        # Case B: the model gave a final natural-language answer. Done.
        messages.append({"role": "assistant", "content": reply.content})
        return reply.content, messages
 
 
if __name__ == "__main__":
    reply, history = run_agent_turn(
        "How does the sky look this week for someone observing at "
        "latitude 33.2, longitude 32.4?"
    )
    print(reply)
