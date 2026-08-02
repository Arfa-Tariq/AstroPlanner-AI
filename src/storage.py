"""
AstroPlanner — Session Storage (Postgres)

This is the persistence layer behind "Observation Session" from the
architecture doc. Runs against hosted Postgres (Neon or Supabase — both
work identically here, this file only needs a DATABASE_URL connection
string) rather than local SQLite. That switch isn't just "nicer DB" —
Colab wipes local disk on every runtime reset, so a SQLite file would
silently lose all sessions between uses. Hosted Postgres survives that.

Schema is unchanged from the SQLite version — same two tables, same
generic (session_id, tool_name, result_json) shape, same reasoning for
why it's generic rather than typed-per-tool (see the v1 docstring in
git history if you want the full explanation again).

Requires: pip install "psycopg[binary]"
Requires: an env var DATABASE_URL (from Neon or Supabase's connection
string page — looks like postgres://user:pass@host/dbname)
"""

import json
import os
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row

DATABASE_URL = os.environ["DATABASE_URL"]


def get_connection():
    """
    Opens a fresh connection per call — simplest possible approach for
    now. A real backend would use a connection pool (many concurrent
    requests sharing a small set of connections) instead of opening a
    new one every time; not needed yet at "one Colab notebook" scale.
    """
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def init_db() -> None:
    """Creates both tables if they don't exist yet. Safe to call every startup."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id SERIAL PRIMARY KEY,
                user_name TEXT NOT NULL,
                latitude DOUBLE PRECISION NOT NULL,
                longitude DOUBLE PRECISION NOT NULL,
                aperture_mm DOUBLE PRECISION,
                created_at TIMESTAMPTZ NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_results (
                id SERIAL PRIMARY KEY,
                session_id INTEGER NOT NULL REFERENCES sessions(id),
                tool_name TEXT NOT NULL,
                result_json JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
            )
        """)
        conn.commit()


def create_session(user_name: str, latitude: float, longitude: float, aperture_mm: float = None) -> int:
    """Starts a new Observation Session, returns its id. Call once per 'planning event'."""
    with get_connection() as conn:
        row = conn.execute(
            "INSERT INTO sessions (user_name, latitude, longitude, aperture_mm, created_at) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (user_name, latitude, longitude, aperture_mm, datetime.now(timezone.utc)),
        ).fetchone()
        conn.commit()
        return row["id"]


def save_result(session_id: int, tool_name: str, result) -> None:
    """Attaches one tool's output to a session. Called automatically after every tool call."""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO session_results (session_id, tool_name, result_json, created_at) "
            "VALUES (%s, %s, %s, %s)",
            (session_id, tool_name, json.dumps(result, default=str), datetime.now(timezone.utc)),
        )
        conn.commit()


def get_session_results(session_id: int) -> list[dict]:
    """All tool results saved under one session, most recent first."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT tool_name, result_json, created_at FROM session_results "
            "WHERE session_id = %s ORDER BY created_at DESC",
            (session_id,),
        ).fetchall()
        return [
            {"tool_name": r["tool_name"], "result": r["result_json"], "created_at": str(r["created_at"])}
            for r in rows
        ]


def list_recent_sessions(user_name: str, limit: int = 10) -> list[dict]:
    """
    Lightweight session list (no result data) — this is what's safe to put
    directly in context so the model KNOWS past sessions exist, without
    the token cost of including their full contents. If the user asks
    about one of these, the model can then call get_session_results with
    its id.
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, latitude, longitude, aperture_mm, created_at FROM sessions "
            "WHERE user_name = %s ORDER BY created_at DESC LIMIT %s",
            (user_name, limit),
        ).fetchall()
        return [
            {**r, "created_at": str(r["created_at"])}
            for r in rows
        ]


def get_most_recent_session(user_name: str) -> "dict | None":
    sessions = list_recent_sessions(user_name, limit=1)
    return sessions[0] if sessions else None
