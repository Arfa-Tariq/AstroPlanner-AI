"""
AstroPlanner — Orchestrator (v0: one tool, Groq)

This is the smallest possible version of the "Conversation Agent" from
the architecture doc. It knows about exactly one tool (weather) so the
loop itself is easy to see clearly. Once this works, adding the other
four tools (visibility, recommendation, fov, scheduler) is just adding
more entries to TOOLS and TOOL_FUNCTIONS — the loop code below does not
change.

Requires: pip install groq
Requires: an env var GROQ_API_KEY (get one free at console.groq.com)
"""

import json
import os

from groq import Groq

from weather import get_weekly_sky_conditions
from models import UserProfile

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"  # supports tool calling on Groq


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
    }
]

# Maps the tool name the model asks for -> the real Python function to run.
# The model only ever sees the name "get_weekly_sky_conditions" (a string);
# this dict is how your code turns that string back into an actual callable.
TOOL_FUNCTIONS = {
    "get_weekly_sky_conditions": get_weekly_sky_conditions,
}


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
