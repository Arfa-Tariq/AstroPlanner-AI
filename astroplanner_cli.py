#!/usr/bin/env python
"""
AstroPlanner — one-file chatbot tester.

Run it, answer a couple of prompts, and you get a menu:

    1) Create a new observation plan (you type in your details)
    2) Chat with the bot (free-form, remembers context)
    3) View my recent sessions
    4) View one session's full stored data
    5) Run automated smoke test (exercises every tool once)
    6) Switch mode: direct Python  <->  real HTTP API
    7) Exit

MODE controls how your messages actually reach the bot:
  - "direct" imports orchestrator.py and calls chat() straight in-process.
  - "api"    starts uvicorn (api.py) in a background thread the first time
             you switch to it, then talks to it over real HTTP requests —
             this is the same path a frontend would use, so it's the
             right way to test api.py itself (routing, CORS, error
             handling), not just the orchestrator underneath it.
You can flip between them at any time with option 6 and keep chatting on
the same thread_id — useful for confirming both paths see the same data.

Setup (once):
    cp env.example .env
    # fill in SUPABASE_URL, SUPABASE_KEY, GROQ_API_KEY, DATABASE_URL
    pip install -r requirements.txt

Run from the repo root:
    python astroplanner_cli.py
"""

import json
import os
import sys
import threading
import time
import uuid

from dotenv import load_dotenv

load_dotenv()

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

REQUIRED_ENV = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
API_PORT = 8000

# ---------------------------------------------------------------------
# Env / startup checks
# ---------------------------------------------------------------------

def check_env():
    missing = [v for v in REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        print(f"Missing required env vars: {', '.join(missing)}")
        print("Copy env.example to .env in the repo root and fill them in first.")
        sys.exit(1)
    if not os.environ.get("DATABASE_URL"):
        print(
            "NOTE: DATABASE_URL not set — conversation memory won't persist "
            "across process restarts (in-memory checkpointing fallback). "
            "Fine for one CLI session.\n"
        )


# ---------------------------------------------------------------------
# API server management (only started if you switch to "api" mode)
# ---------------------------------------------------------------------

_api_server = None
_api_thread = None


def ensure_api_server_running():
    global _api_server, _api_thread
    if _api_thread and _api_thread.is_alive():
        return True

    try:
        import uvicorn
        import requests
    except ImportError:
        print("API mode needs 'uvicorn' and 'requests' installed (pip install -r requirements.txt).")
        return False

    print(f"Starting API server on port {API_PORT}...")
    config = uvicorn.Config("api:app", host="0.0.0.0", port=API_PORT, log_level="warning")
    _api_server = uvicorn.Server(config)
    _api_thread = threading.Thread(target=_api_server.run, daemon=True)
    _api_thread.start()

    for _ in range(20):
        try:
            r = requests.get(f"http://localhost:{API_PORT}/health", timeout=1)
            if r.status_code == 200:
                print(f"API server is up at http://localhost:{API_PORT}\n")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)

    print("WARNING: API server did not respond to /health in time — check the logs above.")
    return False


# ---------------------------------------------------------------------
# Mode-agnostic operations — every menu option calls these, not
# orchestrator/storage/requests directly, so switching mode (option 6)
# transparently changes HOW the call happens without changing any menu
# code.
# ---------------------------------------------------------------------

def send_chat(state, message: str) -> str:
    if state["mode"] == "api":
        import requests
        resp = requests.post(
            f"http://localhost:{API_PORT}/chat",
            json={"message": message, "user_name": state["user"], "thread_id": state["thread"]},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["reply"]
    else:
        from orchestrator import chat
        return chat(message, user_name=state["user"], thread_id=state["thread"])


def get_sessions(state, limit: int = 10) -> list:
    if state["mode"] == "api":
        import requests
        r = requests.get(f"http://localhost:{API_PORT}/users/{state['user']}/sessions", params={"limit": limit})
        r.raise_for_status()
        return r.json()
    else:
        import storage
        user_id = storage.get_or_create_user(state["user"])
        return storage.list_recent_sessions(user_id, limit=limit)


def get_session(state, session_id: str):
    if state["mode"] == "api":
        import requests
        r = requests.get(f"http://localhost:{API_PORT}/sessions/{session_id}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    else:
        import storage
        result = storage.get_full_session(session_id)
        if not result or not result.get("session"):
            return None
        return result


# ---------------------------------------------------------------------
# Menu option 1 — enter your setup, get a plan back
# ---------------------------------------------------------------------

def prompt_float(label, default=None):
    while True:
        raw = input(f"{label}{f' [{default}]' if default is not None else ''}: ").strip()
        if not raw and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a number.")


def prompt_choice(label, choices, default):
    choices_str = "/".join(choices)
    while True:
        raw = input(f"{label} ({choices_str}) [{default}]: ").strip().lower()
        if not raw:
            return default
        if raw in choices:
            return raw
        print(f"  Please enter one of: {choices_str}")


def create_plan_flow(state):
    print("\n--- New observation plan ---")
    print("Enter your location and equipment. Press Enter to accept a default where shown.\n")

    lat = prompt_float("Latitude (decimal degrees, -90..90)")
    lon = prompt_float("Longitude (decimal degrees, -180..180)")
    aperture = prompt_float("Telescope aperture (mm)", default=200)
    focal_length = prompt_float("Telescope focal length (mm)", default=1000)
    experience = prompt_choice("Experience level", ["beginner", "intermediate", "advanced"], default="beginner")

    message = (
        f"Plan my observing session for tonight at latitude {lat}, longitude {lon}, "
        f"with a {aperture}mm aperture telescope, {focal_length}mm focal length, "
        f"{experience} experience level."
    )

    has_camera = input("Do you have a camera for imaging? [y/N]: ").strip().lower() in ("y", "yes")
    if has_camera:
        sw = prompt_float("Sensor width (mm)", default=23.5)
        sh = prompt_float("Sensor height (mm)", default=15.6)
        px = prompt_float("Pixel size (µm)", default=3.76)
        message += f" I have a camera with a {sw}x{sh}mm sensor and {px}um pixels."

    bortle_raw = input("Bortle dark-sky scale (1-9, Enter to skip): ").strip()
    if bortle_raw:
        message += f" My Bortle scale is {bortle_raw}."

    notes = input("Any notes for the assistant (Enter to skip): ").strip()
    if notes:
        message += f" Notes: {notes}"

    print(f"\nSending: {message}\n")
    print("Working (this runs the full weather/visibility/recommendation/FoV/scheduler pipeline, can take a bit)...\n")

    reply = send_chat(state, message)
    print_reply("Assistant", reply)
    print("Tip: use option 2 to keep chatting about this plan, or option 4 with a session_id to see raw stored data.\n")


# ---------------------------------------------------------------------
# Menu option 2 — free-form chat
# ---------------------------------------------------------------------

def print_reply(label, text):
    print(f"{label}:")
    for line in str(text).splitlines() or [""]:
        print(f"  {line}")
    print()


def chat_loop(state):
    print("\n--- Chat mode ---")
    print(f"(mode={state['mode']}, user={state['user']}, thread={state['thread']})")
    print("Type a message and press Enter. /menu to go back, /new for a fresh thread, /exit to quit entirely.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return "exit"

        if not user_input:
            continue
        if user_input == "/menu":
            return "menu"
        if user_input == "/exit":
            return "exit"
        if user_input == "/new":
            state["thread"] = f"cli-{uuid.uuid4().hex[:8]}"
            print(f"New thread: {state['thread']}\n")
            continue

        reply = send_chat(state, user_input)
        print_reply("Assistant", reply)


# ---------------------------------------------------------------------
# Menu options 3/4 — inspect stored sessions
# ---------------------------------------------------------------------

def view_sessions_flow(state):
    sessions = get_sessions(state)
    if not sessions:
        print("No sessions yet for this user.\n")
        return
    print()
    for s in sessions:
        print(
            f"  {s['id']}  {s['generated_at']}  "
            f"lat={s['latitude']} lon={s['longitude']}  rev={s['revision_number']}"
        )
    print()


def view_session_detail_flow(state):
    session_id = input("Session id: ").strip()
    result = get_session(state, session_id)
    if not result:
        print(f"No session found with id {session_id}\n")
        return
    dumped = json.dumps(result, indent=2, default=str)
    print(dumped[:4000])
    if len(dumped) > 4000:
        print("...(truncated)")
    print()


# ---------------------------------------------------------------------
# Menu option 5 — automated smoke test
# ---------------------------------------------------------------------

SMOKE_STEPS = [
    (
        "create_observation_plan",
        "Plan my observing session for tonight at latitude 33.2, longitude 32.4, "
        "with an 8-inch (200mm) telescope, 1000mm focal length, beginner level.",
        lambda reply: "session" in reply.lower() or "plan" in reply.lower() or "ngc" in reply.lower(),
    ),
    (
        "get_session_context (implicit)",
        "Why was the first target on that list recommended?",
        lambda reply: len(reply.strip()) > 0,
    ),
    (
        "regenerate_schedule",
        "Regenerate that schedule but only show me 3 objects.",
        lambda reply: len(reply.strip()) > 0,
    ),
    (
        "revise_observation_plan",
        "Actually, redo the whole plan again from scratch for the same location and setup.",
        lambda reply: len(reply.strip()) > 0,
    ),
    (
        "get_recent_sessions",
        "What have I planned recently?",
        lambda reply: len(reply.strip()) > 0,
    ),
    (
        "search_knowledge_base",
        "What is the Bortle scale?",
        lambda reply: "bortle" in reply.lower(),
    ),
]


def smoke_test_flow(state):
    thread_id = f"smoke-{uuid.uuid4().hex[:8]}"
    smoke_state = {**state, "thread": thread_id}
    print(f"\n--- Smoke test (mode={state['mode']}, thread={thread_id}) ---\n")

    results = []
    for name, message, check in SMOKE_STEPS:
        print(f"--- {name} ---")
        print(f"You: {message}")
        t0 = time.time()
        reply = send_chat(smoke_state, message)
        elapsed = time.time() - t0
        print_reply("Assistant", reply)
        ok = check(reply)
        results.append((name, ok, elapsed))
        print(f"[{'PASS' if ok else 'CHECK MANUALLY'}]  ({elapsed:.1f}s)\n")

    print("=" * 60)
    print("Smoke test summary")
    print("=" * 60)
    for name, ok, elapsed in results:
        print(f"  {'PASS' if ok else 'CHECK':6}  {elapsed:5.1f}s  {name}")
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n_pass}/{len(results)} automated checks passed.")
    print(
        "These checks are loose keyword/length heuristics — read the replies above. "
        "Also worth checking Supabase: observation_sessions should have 2 new rows "
        "(create + revise) and each stage table should have matching rows.\n"
    )


# ---------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------

def print_menu(state):
    print("=" * 50)
    print(f"AstroPlanner CLI  —  mode={state['mode']}  user={state['user']}  thread={state['thread']}")
    print("=" * 50)
    print("1) Create a new observation plan (enter your details)")
    print("2) Chat with the bot")
    print("3) View my recent sessions")
    print("4) View one session's full data")
    print("5) Run automated smoke test (exercises every tool)")
    print(f"6) Switch mode (currently '{state['mode']}' -> other: '{'api' if state['mode'] == 'direct' else 'direct'}')")
    print("7) Exit")


def main():
    check_env()

    print("AstroPlanner chatbot tester\n")
    user = input("Your name (Enter for 'cli_tester'): ").strip() or "cli_tester"
    state = {
        "user": user,
        "thread": f"cli-{uuid.uuid4().hex[:8]}",
        "mode": "direct",  # "direct" (import orchestrator) or "api" (real HTTP)
    }

    while True:
        print()
        print_menu(state)
        choice = input("Choose an option: ").strip()

        if choice == "1":
            create_plan_flow(state)
        elif choice == "2":
            action = chat_loop(state)
            if action == "exit":
                break
        elif choice == "3":
            view_sessions_flow(state)
        elif choice == "4":
            view_session_detail_flow(state)
        elif choice == "5":
            smoke_test_flow(state)
        elif choice == "6":
            new_mode = "api" if state["mode"] == "direct" else "direct"
            if new_mode == "api":
                if not ensure_api_server_running():
                    print("Staying in 'direct' mode — API server failed to start.\n")
                    continue
            state["mode"] = new_mode
            print(f"Switched to '{state['mode']}' mode.\n")
        elif choice == "7":
            print("Bye.")
            break
        else:
            print("Not a valid option, try again.\n")


if __name__ == "__main__":
    main()
