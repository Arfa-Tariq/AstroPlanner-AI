#!/usr/bin/env python
"""
AstroPlanner — Interactive Chatbot Test CLI

Three modes, all against the REAL orchestrator (Groq + Supabase + the
weather/visibility/recommendation/fov/scheduler pipeline) — this is the
"does the whole stack actually work together" test, complementary to
tests/test_*.py (which check the pure math with no network/LLM at all).

    python chat_cli.py                    interactive chat loop
    python chat_cli.py --smoke            scripted run through every tool
    python chat_cli.py --long-convo       forces the summarization path

Setup (once):
    cp env.example .env
    # fill in SUPABASE_URL, SUPABASE_KEY, GROQ_API_KEY, DATABASE_URL
    pip install -r requirements.txt

Run from the repo root (so ./src is importable and env.example/.env are
found in the right place).

In-chat commands (interactive mode only):
    /sessions            list your recent observation sessions
    /session <id>        dump full stored data for one session
    /new                 start a fresh thread_id (wipes conversation memory)
    /help                show this docstring
    /exit or /quit       leave
"""

import argparse
import json
import os
import sys
import time
import uuid

from dotenv import load_dotenv

load_dotenv()

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

REQUIRED_ENV = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]


def check_env():
    missing = [v for v in REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        print(f"Missing required env vars: {', '.join(missing)}")
        print("Copy env.example to .env in the repo root and fill them in first.")
        sys.exit(1)
    if not os.environ.get("DATABASE_URL"):
        print(
            "NOTE: DATABASE_URL not set — conversation memory will NOT persist "
            "across process restarts (falls back to in-memory checkpointing). "
            "Fine for this CLI session, but each new `python chat_cli.py` run "
            "starts with a blank slate unless you set it.\n"
        )


def print_reply(label, text):
    print(f"\n{label}:")
    for line in str(text).splitlines() or [""]:
        print(f"  {line}")
    print()


# ---------------------------------------------------------------------
# Smoke test — sends one scripted message per tool, in an order that
# should make each one fire, and does a loose pass/fail check on the
# reply. This checks LLM tool-routing + the full stack, not the math —
# always read the printed replies yourself, the automated checks here
# are deliberately weak (keyword/length only) since wording varies.
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


def run_smoke_test(user_name: str, thread_id: str):
    from orchestrator import chat

    print(f"Running smoke test — user='{user_name}' thread='{thread_id}'\n")
    results = []
    for name, message, check in SMOKE_STEPS:
        print(f"--- {name} ---")
        print(f"You: {message}")
        t0 = time.time()
        reply = chat(message, user_name=user_name, thread_id=thread_id)
        elapsed = time.time() - t0
        print_reply("Assistant", reply)
        ok = check(reply)
        results.append((name, ok, elapsed))
        status = "PASS" if ok else "WEAK/FAIL — read the reply above manually"
        print(f"[{status}]  ({elapsed:.1f}s)\n")

    print("=" * 60)
    print("Smoke test summary")
    print("=" * 60)
    for name, ok, elapsed in results:
        print(f"  {'PASS' if ok else 'CHECK':5}  {elapsed:5.1f}s  {name}")
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n_pass}/{len(results)} automated checks passed.")
    print(
        "\nThese checks are intentionally loose — cross-check Supabase too: "
        "observation_sessions should have gained 2 new rows (create + revise), "
        "each linked stage table (weather/visibility/recommendation/fov/"
        "observation_schedules) should have matching rows, and the revised "
        "session's revision_number should be 2."
    )


def run_long_conversation_test(user_name: str, thread_id: str, n_messages: int = 13):
    """
    Sends n_messages throwaway turns to force orchestrator._summarize_and_trim
    to fire (default threshold: SUMMARIZE_AFTER_N_MESSAGES = 12 raw messages
    in graph state). Confirms a summary actually lands in Supabase.
    """
    from orchestrator import chat
    import storage

    print(f"Sending {n_messages} filler messages on thread '{thread_id}' to trigger summarization...\n")
    for i in range(n_messages):
        msg = f"This is filler message number {i + 1}, just reply OK."
        reply = chat(msg, user_name=user_name, thread_id=thread_id)
        preview = reply[:60] + ("..." if len(reply) > 60 else "")
        print(f"  [{i + 1}/{n_messages}] -> {preview}")

    user_id = storage.get_or_create_user(user_name)
    conversation_id = storage.get_or_create_conversation(user_id, thread_id)
    summary = storage.get_latest_conversation_summary(conversation_id)

    print()
    if summary:
        print("PASS — a conversation summary was saved:")
        print(f"  {summary}")
    else:
        print(
            "FAIL — no summary found. Either the threshold wasn't reached "
            "(check SUMMARIZE_AFTER_N_MESSAGES in orchestrator.py, currently 12) "
            "or the summarization step didn't run — check for errors above."
        )


def interactive_loop(user_name: str, thread_id: str):
    from orchestrator import chat
    import storage

    print(f"AstroPlanner chat — user='{user_name}' thread='{thread_id}'")
    print("Type a message and press Enter. /help for commands, /exit to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            print("Bye.")
            break
        if user_input == "/help":
            print(__doc__)
            continue
        if user_input == "/new":
            thread_id = f"cli-{uuid.uuid4().hex[:8]}"
            print(f"New thread: {thread_id}\n")
            continue
        if user_input == "/sessions":
            user_id = storage.get_or_create_user(user_name)
            sessions = storage.list_recent_sessions(user_id, limit=10)
            if not sessions:
                print("No sessions yet for this user.\n")
            else:
                for s in sessions:
                    print(
                        f"  {s['id']}  {s['generated_at']}  "
                        f"lat={s['latitude']} lon={s['longitude']}  rev={s['revision_number']}"
                    )
                print()
            continue
        if user_input.startswith("/session "):
            session_id = user_input.split(" ", 1)[1].strip()
            result = storage.get_full_session(session_id)
            if not result or not result.get("session"):
                print(f"No session found with id {session_id}\n")
            else:
                dumped = json.dumps(result, indent=2, default=str)
                print(dumped[:4000])
                if len(dumped) > 4000:
                    print("...(truncated)")
                print()
            continue

        reply = chat(user_input, user_name=user_name, thread_id=thread_id)
        print_reply("Assistant", reply)


def main():
    parser = argparse.ArgumentParser(description="Interactive/smoke-test CLI for the AstroPlanner chatbot.")
    parser.add_argument("--user", default="cli_tester", help="user_name to chat as (default: cli_tester)")
    parser.add_argument("--thread", default=None, help="thread_id to use (default: a fresh random one)")
    parser.add_argument("--smoke", action="store_true", help="run the automated tool-coverage smoke test")
    parser.add_argument("--long-convo", action="store_true", help="run the summarization/trim test")
    args = parser.parse_args()

    check_env()
    thread_id = args.thread or f"cli-{uuid.uuid4().hex[:8]}"

    if args.smoke:
        run_smoke_test(args.user, thread_id)
    elif args.long_convo:
        run_long_conversation_test(args.user, thread_id)
    else:
        interactive_loop(args.user, thread_id)


if __name__ == "__main__":
    main()
