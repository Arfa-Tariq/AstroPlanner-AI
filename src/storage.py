"""
AstroPlanner — Session Storage (Supabase)

Replaces the psycopg/generic-table version. Two real changes, not just a
different client library:

1. Typed tables per pipeline stage instead of one generic
   (session_id, tool_name, result_json) table — see db/supabase_schema.sql
   for the reasoning. This file's functions map 1:1 onto those tables.

2. Session revisions are now a first-class concept: create_session()
   accepts an optional parent_session_id, and revision_number is derived
   automatically (count of existing children + 1) rather than left for
   the caller to track by hand.

Uses the supabase-py client (table-builder API) rather than raw SQL —
this is the idiomatic way to talk to Supabase from Python, and it's what
you want if Auth / Row Level Security get added later, since the client
understands both.

Requires: pip install supabase
Requires env vars: SUPABASE_URL, SUPABASE_KEY (service-role key for a
backend process like this one — never ship the service-role key to a
browser client).
"""

import os
from typing import Optional

from supabase import create_client, Client

from models import UserProfile

_client: Optional[Client] = None


def get_client() -> Client:
    """
    Lazily creates one Supabase client and reuses it — same "build once"
    principle as VisibilityEngine's ephemeris loading, just for a network
    client instead of a large local file.
    """
    global _client
    if _client is None:
        _client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    return _client


# ---------------------------------------------------------------------
# Users + equipment
# ---------------------------------------------------------------------

def get_or_create_user(name: str) -> str:
    """Returns the user's id, creating a row if this name hasn't been seen.
    Simple name-based lookup for now — swap for Supabase Auth's user id
    once the app has real accounts instead of a single chat_user."""
    client = get_client()
    existing = client.table("users").select("id").eq("name", name).limit(1).execute()
    if existing.data:
        return existing.data[0]["id"]
    created = client.table("users").insert({"name": name}).execute()
    return created.data[0]["id"]


def save_equipment(user_id: str, user: UserProfile, label: str = "default") -> str:
    """
    Stores a snapshot of the user's telescope/camera/mount/preferences.
    Always inserts a new row rather than updating in place — past
    sessions should keep pointing at the equipment_id that was actually
    used at the time, even if the user's gear changes later.
    """
    client = get_client()
    row = {
        "user_id": user_id,
        "label": label,
        "telescope": user.telescope.model_dump(mode="json"),
        "camera": user.camera.model_dump(mode="json") if user.camera else None,
        "mount": user.mount.model_dump(mode="json") if user.mount else None,
        "experience_level": user.experience_level.value,
        "bortle_scale": user.bortle_scale,
        "preferences": user.preferences.model_dump(mode="json") if user.preferences else None,
    }
    created = client.table("equipment").insert(row).execute()
    return created.data[0]["id"]


# ---------------------------------------------------------------------
# Observation sessions
# ---------------------------------------------------------------------

def create_session(
    user_id: str,
    latitude: float,
    longitude: float,
    generated_at: str,
    equipment_id: Optional[str] = None,
    notes: Optional[str] = None,
    parent_session_id: Optional[str] = None,
) -> str:
    """
    Starts a new Observation Session. If parent_session_id is given, this
    is a revision of that session — revision_number is computed as
    (number of existing children of that parent) + 2 (the parent counts
    as revision 1), so callers never have to track the count themselves.
    """
    client = get_client()

    revision_number = 1
    if parent_session_id is not None:
        siblings = (
            client.table("observation_sessions")
            .select("id", count="exact")
            .eq("parent_session_id", parent_session_id)
            .execute()
        )
        revision_number = (siblings.count or 0) + 2

    row = {
        "user_id": user_id,
        "equipment_id": equipment_id,
        "parent_session_id": parent_session_id,
        "revision_number": revision_number,
        "latitude": latitude,
        "longitude": longitude,
        "generated_at": generated_at,
        "notes": notes,
    }
    created = client.table("observation_sessions").insert(row).execute()
    return created.data[0]["id"]


_STAGE_TABLES = {
    "weather": "weather_analyses",
    "visibility": "visibility_analyses",
    "recommendation": "recommendation_analyses",
    "fov": "fov_analyses",
    "schedule": "observation_schedules",
}


def save_stage_result(session_id: str, stage: str, result) -> None:
    """
    Attaches one pipeline stage's output to a session. `stage` must be one
    of weather/visibility/recommendation/fov/schedule — this is the single
    write path every tool wrapper in orchestrator.py calls after running
    its computation, so the mapping from stage name to table lives in
    exactly one place.
    """
    if stage not in _STAGE_TABLES:
        raise ValueError(f"Unknown stage '{stage}'. Expected one of {list(_STAGE_TABLES)}.")
    client = get_client()
    client.table(_STAGE_TABLES[stage]).insert({
        "session_id": session_id,
        "result": result,
    }).execute()


def get_latest_stage_result(session_id: str, stage: str) -> Optional[dict]:
    """Most recent result for one stage of one session, or None if that
    stage hasn't been run yet for this session."""
    if stage not in _STAGE_TABLES:
        raise ValueError(f"Unknown stage '{stage}'. Expected one of {list(_STAGE_TABLES)}.")
    client = get_client()
    rows = (
        client.table(_STAGE_TABLES[stage])
        .select("result, created_at")
        .eq("session_id", session_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return rows.data[0]["result"] if rows.data else None


def get_full_session(session_id: str) -> dict:
    """
    Everything known about one session: its metadata plus the latest
    result from every stage that has run so far. This is what a tool
    like get_past_session_results hands back to the LLM — one call
    instead of five.
    """
    client = get_client()
    session_row = client.table("observation_sessions").select("*").eq("id", session_id).single().execute()

    return {
        "session": session_row.data,
        "weather": get_latest_stage_result(session_id, "weather"),
        "visibility": get_latest_stage_result(session_id, "visibility"),
        "recommendation": get_latest_stage_result(session_id, "recommendation"),
        "fov": get_latest_stage_result(session_id, "fov"),
        "schedule": get_latest_stage_result(session_id, "schedule"),
    }


def list_recent_sessions(user_id: str, limit: int = 10) -> list[dict]:
    """
    Lightweight session list (metadata only, no stage results) — safe to
    put directly into the system prompt so the model KNOWS past sessions
    exist without the token cost of their full contents. The model calls
    get_full_session with an id from here if the user asks about one.
    """
    client = get_client()
    rows = (
        client.table("observation_sessions")
        .select("id, latitude, longitude, generated_at, revision_number, notes, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return rows.data


def get_most_recent_session(user_id: str) -> Optional[dict]:
    sessions = list_recent_sessions(user_id, limit=1)
    return sessions[0] if sessions else None


# ---------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------

def get_or_create_conversation(user_id: str, thread_id: str, session_id: Optional[str] = None) -> str:
    """thread_id is the LangGraph checkpointer's key — this table also
    persists conversations to Supabase for history/summarization,
    independent of LangGraph's own checkpointed state."""
    client = get_client()
    existing = client.table("conversations").select("id").eq("thread_id", thread_id).limit(1).execute()
    if existing.data:
        return existing.data[0]["id"]
    created = client.table("conversations").insert({
        "user_id": user_id, "thread_id": thread_id, "session_id": session_id,
    }).execute()
    return created.data[0]["id"]


def save_message(conversation_id: str, role: str, content: str) -> None:
    client = get_client()
    client.table("messages").insert({
        "conversation_id": conversation_id, "role": role, "content": content,
    }).execute()


def save_conversation_summary(conversation_id: str, summary: str) -> None:
    client = get_client()
    client.table("conversation_summaries").insert({
        "conversation_id": conversation_id, "summary": summary,
    }).execute()


def get_latest_conversation_summary(conversation_id: str) -> Optional[str]:
    client = get_client()
    rows = (
        client.table("conversation_summaries")
        .select("summary")
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return rows.data[0]["summary"] if rows.data else None


# ---------------------------------------------------------------------
# Semantic memory (pgvector) — see db/supabase_schema.sql: match_knowledge_base
# ---------------------------------------------------------------------

def search_knowledge_base(query_embedding: list[float], match_count: int = 5) -> list[dict]:
    """
    Calls the match_knowledge_base Postgres function via RPC rather than
    hand-writing the vector-search SQL here — keeps the similarity query
    (and its index usage) defined once, in the schema, next to the table
    and index it depends on.
    """
    client = get_client()
    result = client.rpc("match_knowledge_base", {
        "query_embedding": query_embedding,
        "match_count": match_count,
    }).execute()
    return result.data
