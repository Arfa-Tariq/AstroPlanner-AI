"""
AstroPlanner — API Service (FastAPI)

The first non-Colab entrypoint into the orchestrator. Deliberately thin:
every real decision (which tool to call, how to build context, how to
persist state) already lives in orchestrator.py / storage.py — this file
only adapts that to HTTP, matching the guide's own rule: "Assume current
tools are standalone Colab prototypes that will later be migrated into
FastAPI services or Python modules with the same input/output contracts."

Run locally:
    cp .env.example .env      # fill in SUPABASE_URL, SUPABASE_KEY, GROQ_API_KEY, DATABASE_URL
    pip install -r requirements.txt
    uvicorn api:app --reload --port 8000

Then:
    curl -X POST localhost:8000/chat \
      -H "Content-Type: application/json" \
      -d '{"user_name": "andrew", "thread_id": "session-1", "message": "Plan tonight at lat 33.2, lon 32.4, 8-inch, 1000mm, beginner"}'
"""

import os
from typing import Optional

from dotenv import load_dotenv

# Loaded before importing orchestrator, since orchestrator reads several
# env vars (GROQ_API_KEY, SUPABASE_URL/KEY, DATABASE_URL) at import time
# to build the LLM client and checkpointer — those must already be in
# os.environ by the time `import orchestrator` runs.
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import orchestrator
import storage

app = FastAPI(
    title="AstroPlanner AI API",
    description="Conversational orchestrator for observation planning, "
    "backed by weather/visibility/recommendation/FoV/scheduler agents.",
    version="0.1.0",
)

# Wide-open for now — the guide's UI is a separate frontend app (the
# "Dashboard" / "AI Assistant" panel), so CORS needs to allow it during
# development. Tighten to specific origins before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message.")
    user_name: str = Field(
        "chat_user",
        description="Resolved to a Supabase user_id server-side — same "
        "identity-binding pattern as orchestrator.chat(). NOT trusted as "
        "an auth mechanism; swap for real auth before this is public.",
    )
    thread_id: str = Field(
        "default",
        description="Conversation/session key for LangGraph's checkpointer. "
        "One thread_id = one continuous conversation with memory.",
    )


class ChatResponse(BaseModel):
    reply: str


class SessionSummary(BaseModel):
    id: str
    latitude: float
    longitude: float
    generated_at: str
    revision_number: int
    notes: Optional[str] = None
    created_at: str


@app.get("/health")
def health() -> dict:
    """Liveness check — does NOT verify Supabase/Groq connectivity, just
    that the process is up and orchestrator imported without error."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    The one endpoint the frontend's chat panel needs. Mirrors
    orchestrator.chat()'s signature exactly — this is the "same
    input/output contract" the dev guide asks for, just reachable over
    HTTP instead of a direct Python import in a Colab cell.
    """
    try:
        reply = orchestrator.chat(
            user_message=req.message,
            user_name=req.user_name,
            thread_id=req.thread_id,
        )
    except Exception as e:
        # orchestrator.chat() already degrades LLM-side failures into a
        # friendly string via _invoke_with_retry — this except is for
        # everything upstream of that (e.g. Supabase unreachable), which
        # should surface as a real HTTP error rather than a silent 200.
        raise HTTPException(status_code=502, detail=str(e))
    return ChatResponse(reply=reply)


@app.get("/users/{user_name}/sessions", response_model=list[SessionSummary])
def list_sessions(user_name: str, limit: int = 10) -> list[SessionSummary]:
    """
    Powers the Dashboard's "latest observation plan" / "Observation
    History" views without going through the chat agent — the guide's
    dashboard needs to show this on load, before any conversation starts.
    """
    user_id = storage.get_or_create_user(user_name)
    return storage.list_recent_sessions(user_id, limit=limit)


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    """Full session detail (all stages) for the Dashboard's session view.
    Same underlying call the chat agent uses via get_session_context, just
    without going through the LLM — cheaper and structured for a UI."""
    result = storage.get_full_session(session_id)
    if not result or not result.get("session"):
        raise HTTPException(status_code=404, detail=f"No session found with id {session_id}")
    return result
