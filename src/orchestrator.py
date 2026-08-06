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

3. WEB SEARCH FALLBACK (Tavily): search_knowledge_base only covers a
   small seeded astronomy knowledge base, and the session tools only
   cover this user's own data. Anything outside both of those (current
   events, general "what is X" questions the seed set doesn't have,
   product questions, etc.) previously had no path except the model
   guessing from its own training data. web_search is a deliberately
   LAST-RESORT tool — the system prompt tells the model to try the
   session tools and search_knowledge_base first, since those are
   grounded in data this app actually owns, and only fall back to a
   live web search when neither applies. It degrades gracefully to a
   clear "not configured" message if TAVILY_API_KEY isn't set, rather
   than crashing the whole agent.
"""

import os
import json
from datetime import date

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from groq import BadRequestError
# ---------------------------------------------------------------------
# Checkpointer: Postgres (Supabase) if DATABASE_URL is set, else in-memory.
#
# MemorySaver keeps conversation state only in this Python process — a
# Colab runtime restart wipes it, even though everything ELSE (sessions,
# equipment, messages, summaries) is already safely in Supabase via
# storage.py. This closes that gap by pointing LangGraph's own
# checkpointer at the same Postgres instance.
#
# Note this is a DIFFERENT connection path than storage.py: storage.py
# talks to Supabase's REST/PostgREST layer via the supabase-py client
# (good for typed row CRUD, RLS-aware). The checkpointer needs raw SQL
# access to manage its own checkpoint tables, which only a direct
# Postgres connection provides — hence a separate DATABASE_URL env var
# (Project Settings -> Database -> Connection string), distinct from
# SUPABASE_URL/SUPABASE_KEY.
# ---------------------------------------------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL")

if DATABASE_URL:
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg_pool import ConnectionPool

    # prepare_threshold=0 disables prepared statements — REQUIRED against
    # Supabase's pooled connection (pgbouncer, transaction mode), which
    # doesn't support them. Without this, the first checkpoint write
    # fails with an unhelpful protocol-level error.
    def _configure_connection(conn):
        # prepare_threshold is a Connection ATTRIBUTE, not a psycopg.connect()
        # parameter — passing it inside connect kwargs (the previous version
        # of this code) is silently ignored, which is exactly what caused
        # the DuplicatePreparedStatement error: psycopg's default auto-prepare
        # behavior stayed on and collided with Supabase's pgbouncer pooler,
        # which doesn't support server-side prepared statements at all. This
        # `configure` callback runs against every new connection the pool
        # opens, which is the correct place to set it.
        conn.autocommit = True
        conn.prepare_threshold = None  # None fully disables auto-prepare; 0 does not

    _connection_pool = ConnectionPool(
        conninfo=DATABASE_URL,
        max_size=5,
        configure=_configure_connection,
    )
    checkpointer = PostgresSaver(_connection_pool)
    checkpointer.setup()  # idempotent — creates checkpoint tables on first run, no-ops after
else:
    print(
        "WARNING: DATABASE_URL not set — falling back to in-memory checkpointing. "
        "Conversation state will NOT survive a runtime restart. Set DATABASE_URL "
        "(Supabase: Project Settings -> Database -> Connection string, URI tab) to fix this."
    )
    checkpointer = MemorySaver()

import weather
import recommendation
import fov as fov_module
import scheduler
from visibility import VisibilityEngine
from models import UserProfile
import storage
import knowledge

# temperature=0.1 (not 0): near-deterministic tool selection — the main
# reliability lever for "did it call the right tool at all" — while still
# leaving _invoke_with_retry a real (if small) chance of getting a
# differently-sampled, well-formed tool call on retry after a Groq
# BadRequestError. temperature=0 would make retries pointless since the
# same malformed call would just be regenerated identically.
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.1,
)

DATA_DIR = os.environ.get("ASTROPLANNER_DATA_DIR", "./data")
os.makedirs(DATA_DIR, exist_ok=True)

# Optional — web_search degrades to a clear "not configured" message
# rather than raising if this isn't set, same pattern as DATABASE_URL
# above. Get a key at https://tavily.com (free tier available).
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# Caches — same "build once per conversation, not once per call" idea as
# VisibilityEngine's own internal ephemeris/catalog caching. Keyed so
# multiple users/locations in the same process don't collide or
# needlessly re-download the NGC catalog.
_visibility_engine_cache: dict = {}

# Lazily created once, same "build once" principle as storage.get_client()
# — avoids importing the tavily package at all (and avoids a client
# construction cost) unless web_search is actually called.
_tavily_client = None


def get_tavily_client():
    global _tavily_client
    if _tavily_client is None:
        from tavily import TavilyClient
        _tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    return _tavily_client


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
    today = date.today()  # computed once, reused for both — matches the
                           # notebooks' generated_at pattern; calling
                           # date.today() independently for weather vs.
                           # visibility risks a midnight rollover splitting
                           # the two 7-day windows apart.

    weekly_weather = weather.get_weekly_sky_conditions(user, today)
    storage.save_stage_result(session_id, "weather", weekly_weather)

    weekly_visibility = engine.get_weekly_visibility(today)
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

    conversation_id = storage.get_or_create_conversation(user_id, thread_id)
    storage.set_conversation_session(conversation_id, session_id)
   
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
def regenerate_schedule(session_id: str, max_objects: int = 8, notes: str = None, config: RunnableConfig = None) -> str:
    """Re-runs ONLY the scheduling stage for an existing session, reusing
    its already-computed weather/visibility/recommendation/fov data rather
    than re-running the full pipeline. Use this for requests that only
    change scheduling constraints — e.g. "give me a 2-hour window instead",
    "show me fewer targets", "only give me 4 objects tonight" — where the
    underlying sky/weather data hasn't changed. Saves the result as a new
    revision. Use revise_observation_plan instead if the user's location,
    equipment, or date has actually changed."""
    thread_id = config["configurable"]["thread_id"]
    identity = _thread_identity_cache.get(thread_id)
    if identity is None:
        return "No user identity bound to this conversation — this is a setup bug, not something to ask the user about."
    user_id = identity["user_id"]

    parent = storage.get_full_session(session_id)
    parent_meta = (parent or {}).get("session")
    fov_result = (parent or {}).get("fov")
    if not parent_meta or not fov_result:
        return f"Session {session_id} is missing FoV data — cannot regenerate its schedule. Use create_observation_plan or revise_observation_plan instead."

    new_session_id = storage.create_session(
        user_id=user_id, latitude=parent_meta["latitude"], longitude=parent_meta["longitude"],
        generated_at=parent_meta["generated_at"], equipment_id=parent_meta.get("equipment_id"),
        notes=notes, parent_session_id=session_id,
    )

    # Reuse every prior stage's result as-is — only the schedule is recomputed.
    for stage in ("weather", "visibility", "recommendation", "fov"):
        prior = (parent or {}).get(stage)
        if prior is not None:
            storage.save_stage_result(new_session_id, stage, prior)

    weekly_schedule = scheduler.get_weekly_schedule(fov_result["nights"], max_objects=max_objects)
    storage.save_stage_result(new_session_id, "schedule", weekly_schedule)

    return json.dumps({
        "session_id": new_session_id,
        "revision_of": session_id,
        "schedule_preview": _trim_schedule_for_chat(weekly_schedule),
        "note": "Schedule-only revision — weather/visibility/recommendations reused from the parent session unchanged.",
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


@tool
def search_knowledge_base(query: str, match_count: int = 3) -> str:
    """Searches general astronomy knowledge (object descriptions, terms
    like seeing/transparency/Bortle scale, imaging concepts) — NOT
    observation history. Use for questions like "tell me about the Orion
    Nebula" or "what does seeing mean", where the answer is astronomy
    knowledge rather than something from the user's own sessions. Do NOT
    use this for "why was X recommended in my plan" — that's
    get_session_context instead, since the answer depends on the user's
    actual saved data, not general knowledge. If this returns nothing
    relevant, try web_search next rather than answering from memory."""
    print(f"DEBUG: search_knowledge_base called with query='{query}'")  # remove once confirmed working
    query_embedding = knowledge.embed_text(query)
    results = storage.search_knowledge_base(query_embedding, match_count=match_count)
    print(f"DEBUG: got {len(results) if results else 0} results")
    if not results:
        return "No relevant knowledge base entries found for this query."
    return json.dumps(results, default=str)


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """LAST-RESORT fallback live web search (via Tavily) for questions
    this app has no other way to answer. Use this ONLY when BOTH of the
    following are true: (1) the question is NOT about the user's own
    observation sessions/equipment/schedule — use get_session_context /
    get_recent_sessions for that instead, and (2) you already tried
    search_knowledge_base for an astronomy question and it came back with
    nothing relevant, OR the question isn't an astronomy topic at all
    (e.g. current events, general "what is X", product comparisons).
    Do NOT use this as a first resort, and do NOT use it for anything
    about the user's own plans — those tools have the actual data, this
    tool does not. When you use results from this tool, make clear to the
    user that the answer came from a live web search rather than this
    app's own curated data, since web results can be wrong or outdated."""
    if not TAVILY_API_KEY:
        return (
            "Web search is not configured for this deployment (no TAVILY_API_KEY set) "
            "— cannot fulfill this request. Tell the user you don't have a way to "
            "search the web right now, rather than guessing an answer."
        )
    try:
        client = get_tavily_client()
        response = client.search(query, max_results=max_results)
    except Exception as e:
        return f"Web search failed ({e}). Tell the user the search failed rather than guessing an answer."

    results = response.get("results", [])
    if not results:
        return "No web results found for this query."

    formatted = [
        {"title": r.get("title"), "url": r.get("url"), "content": r.get("content")}
        for r in results
    ]
    return json.dumps(formatted, default=str)


def build_tools() -> list:
    return [
        create_observation_plan, revise_observation_plan, regenerate_schedule,
        get_session_context, get_recent_sessions, search_knowledge_base,
        web_search,
    ]


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
        "saved as a linked revision rather than an unrelated new session. "
        "\n\n"
        "For anything you don't already know how to answer, follow this "
        "order rather than guessing from your own training data: "
        "(1) if it's about the user's own plans, equipment, or schedule, "
        "call get_recent_sessions or get_session_context — never guess or "
        "claim you lack data you haven't actually looked up; "
        "(2) if it's a general astronomy question, call "
        "search_knowledge_base first; "
        "(3) only if neither applies — or search_knowledge_base came back "
        "empty — call web_search as a last resort, and tell the user the "
        "answer came from a live web search rather than this app's own "
        "data, since it can be wrong or outdated. Do not skip straight to "
        "web_search for astronomy topics search_knowledge_base might cover, "
        "and do not use web_search for the user's own session data."
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
    checkpointer=checkpointer,
)


# Once a thread's raw message count passes this, everything except the
# last KEEP_LAST_N_MESSAGES is condensed into one summary and removed
# from graph state. Deliberately small here to make the behavior easy to
# observe while testing — raise both once you trust it (e.g. 20 / 6).
SUMMARIZE_AFTER_N_MESSAGES = 12
KEEP_LAST_N_MESSAGES = 4


def _summarize_and_trim(thread_id: str, conversation_id: str) -> None:
    """
    Keeps one thread's graph state bounded. MemorySaver keeps every
    message in a thread forever by default — dynamic_prompt only
    controls the system message, the full raw history (including big
    tool outputs) still gets resent to Groq every turn regardless. THIS
    is the actual mechanism behind both the 413 and 429 errors earlier:
    token cost per turn was growing, not constant.

    Called at the start of every chat() turn, before the new message is
    added. If history is still short, this is a no-op (one cheap
    get_state call). Once it's long: everything except the most recent
    KEEP_LAST_N_MESSAGES is sent to the LLM for a short summary, that
    summary is persisted to Supabase (conversation_summaries — the table
    that's existed in the schema since the start but had no writer
    until now), and the old messages are deleted from graph state via
    RemoveMessage — not hidden from the prompt, actually removed from
    what gets sent next turn.
    """
    config = {"configurable": {"thread_id": thread_id}}
    current_state = agent.get_state(config)
    messages = current_state.values.get("messages", [])

    if len(messages) <= SUMMARIZE_AFTER_N_MESSAGES:
        return

    to_summarize = messages[:-KEEP_LAST_N_MESSAGES]
    to_keep = messages[-KEEP_LAST_N_MESSAGES:]

    transcript = "\n".join(
        f"{m.type}: {m.content}" for m in to_summarize if isinstance(m.content, str) and m.content
    )
    summary_prompt = (
        "Summarize this astronomy-planning conversation in 3-5 sentences. "
        "Keep any session_ids, locations, or equipment details mentioned "
        "— they may be needed later. Be concise.\n\n" + transcript
    )
    summary = llm.invoke(summary_prompt).content

    storage.save_conversation_summary(conversation_id, summary)

    removals = [RemoveMessage(id=m.id) for m in to_summarize if m.id is not None]
    summary_message = SystemMessage(content=f"[Earlier conversation summary]: {summary}")
    agent.update_state(config, {"messages": removals + [summary_message]})


def chat(user_message: str, user_name: str = "chat_user", thread_id: str = "default") -> str:
    """
    Runs one turn. thread_id identifies the conversation for LangGraph's
    checkpointer (same thread_id = agent remembers prior turns).
    user_name is resolved to a Supabase user_id HERE, from a real Python
    argument — not from the LLM — and cached against thread_id so every
    tool call and dynamic_prompt invocation in this thread can look up
    "who is this" via RunnableConfig instead of asking the model to
    infer or supply it.

    Also persists both sides of the turn to Supabase (messages table) —
    independent of LangGraph's own checkpointed state, so conversation
    history survives even if MemorySaver's in-memory store is ever
    swapped out or the process restarts.
    """
    if thread_id not in _thread_identity_cache:
        user_id = storage.get_or_create_user(user_name)
        _thread_identity_cache[thread_id] = {"user_id": user_id, "user_name": user_name}
    user_id = _thread_identity_cache[thread_id]["user_id"]

    conversation_id = storage.get_or_create_conversation(user_id, thread_id)
    _summarize_and_trim(thread_id, conversation_id)

    config = {"configurable": {"thread_id": thread_id}}

    # retries=2 (3 attempts total): bumped from 1 — with temperature=0.1
    # each retry is a genuinely different sample, so a couple of extra
    # shots meaningfully raises the odds of a well-formed tool call
    # without materially hurting latency on the (rare) failure path.
    reply = _invoke_with_retry(user_message, config, retries=2)

    storage.save_message(conversation_id, "user", user_message)
    storage.save_message(conversation_id, "assistant", reply)

    return reply

def _invoke_with_retry(user_message: str, config: dict, retries: int = 1) -> str:
    """
    Groq's tool-call parser occasionally rejects the model's own malformed
    tool-call syntax (e.g. 'get_recent_sessions [{"limit": 1}]' instead of
    proper JSON args) — a model-side formatting slip, not a bug in our
    tools. Retrying once with a fresh call usually gets a clean response;
    if it fails twice, degrade to a message instead of propagating a raw
    500-style error to the user.
    """
    last_error = None
    for attempt in range(retries + 1):
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_message}]}, config
            )
            return result["messages"][-1].content
        except BadRequestError as e:
            last_error = e
            print(f"WARNING: tool-call validation failed (attempt {attempt + 1}): {e}")
            continue
        except Exception as e:
            last_error = e
            print(f"WARNING: agent.invoke failed (attempt {attempt + 1}): {e}")
            continue

    return (
        "Sorry — I hit an error trying to process that. Could you rephrase "
        "your question, or try again? (If this keeps happening, it's "
        f"likely a transient model issue, not your input. Details: {last_error})"
    )

if __name__ == "__main__":
    print(chat(
        "Plan my observing session for tonight at latitude 33.2, longitude 32.4, "
        "with an 8-inch (200mm) telescope, 1000mm focal length, beginner level.",
        user_name="andrew",
    ))
