-- AstroPlanner AI — Supabase schema
--
-- Design decisions worth understanding, not just running:
--
-- 1. Typed tables per pipeline stage (weather_analyses, visibility_analyses, ...)
--    instead of the old generic (session_id, tool_name, result_json) table.
--    The old shape was flexible but meant every query had to filter by a
--    magic string tool_name and re-parse jsonb blindly. Typed tables let
--    Postgres (and your tool functions) express "give me the latest
--    recommendation for this session" as a real query, and let you add a
--    stage-specific column later (e.g. recommendation_analyses.top_score)
--    without a migration touching every other stage.
--
-- 2. Revisions via (parent_session_id, revision_number) rather than
--    overwriting a row. "Regenerate schedule with 2 hours only" creates a
--    NEW observation_sessions row pointing back at the original via
--    parent_session_id. revision_number just makes "show me v3 of this"
--    human-readable without joining through the whole parent chain.
--
-- 3. jsonb payload columns are still used *within* each typed table —
--    the analysis results themselves are deeply nested (a week of nights,
--    each with a list of scored objects) and don't need to be normalized
--    into rows. Typing lives at the "which stage / which session" level,
--    not inside the payload.
--
-- 4. pgvector lives in the same Postgres instance as everything else.
--    Supabase enables this with one extension call — no second database,
--    no second client, no cross-service latency for a RAG lookup.

create extension if not exists vector;
create extension if not exists "uuid-ossp";

-- ---------------------------------------------------------------------
-- Identity + equipment
-- ---------------------------------------------------------------------

create table if not exists users (
    id uuid primary key default uuid_generate_v4(),
    name text not null,
    created_at timestamptz not null default now()
);

-- Equipment is stored separately from sessions (not just embedded in the
-- session snapshot) so a user's telescope/camera/mount persist across
-- sessions instead of being re-entered every time. A session still
-- captures which equipment_id was used at that moment, so past sessions
-- stay accurate even if the user upgrades their telescope later.
create table if not exists equipment (
    id uuid primary key default uuid_generate_v4(),
    user_id uuid not null references users(id),
    label text not null default 'default',       -- lets a user have >1 rig later
    telescope jsonb not null,                     -- TelescopeSpec.model_dump()
    camera jsonb,                                 -- CameraSpec.model_dump(), nullable
    mount jsonb,                                  -- MountSpec.model_dump(), nullable
    experience_level text not null,
    bortle_scale integer,
    preferences jsonb,                             -- Preferences.model_dump(), nullable
    created_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------
-- Observation sessions — the core entity from the guide
-- ---------------------------------------------------------------------

create table if not exists observation_sessions (
    id uuid primary key default uuid_generate_v4(),
    user_id uuid not null references users(id),
    equipment_id uuid references equipment(id),
    parent_session_id uuid references observation_sessions(id),
    revision_number integer not null default 1,
    latitude double precision not null,
    longitude double precision not null,
    generated_at date not null,                    -- matches WeeklyPlanRequest.generated_at
    notes text,
    created_at timestamptz not null default now()
);

create index if not exists idx_sessions_user on observation_sessions(user_id, created_at desc);
create index if not exists idx_sessions_parent on observation_sessions(parent_session_id);

-- One row per pipeline stage per session. Each stage table is intentionally
-- identical in shape (session_id, result jsonb, created_at) — the only
-- reason they're separate tables rather than one generic table is so a
-- tool can query "the weather for session X" without a string filter, and
-- so each stage can independently gain typed columns later if needed.

create table if not exists weather_analyses (
    id uuid primary key default uuid_generate_v4(),
    session_id uuid not null references observation_sessions(id),
    result jsonb not null,
    created_at timestamptz not null default now()
);

create table if not exists visibility_analyses (
    id uuid primary key default uuid_generate_v4(),
    session_id uuid not null references observation_sessions(id),
    result jsonb not null,
    created_at timestamptz not null default now()
);

create table if not exists recommendation_analyses (
    id uuid primary key default uuid_generate_v4(),
    session_id uuid not null references observation_sessions(id),
    result jsonb not null,
    created_at timestamptz not null default now()
);

create table if not exists fov_analyses (
    id uuid primary key default uuid_generate_v4(),
    session_id uuid not null references observation_sessions(id),
    result jsonb not null,
    created_at timestamptz not null default now()
);

create table if not exists observation_schedules (
    id uuid primary key default uuid_generate_v4(),
    session_id uuid not null references observation_sessions(id),
    result jsonb not null,
    created_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------
-- Conversation memory
-- ---------------------------------------------------------------------

create table if not exists conversations (
    id uuid primary key default uuid_generate_v4(),
    user_id uuid not null references users(id),
    session_id uuid references observation_sessions(id),  -- which session this chat is "attached to"
    thread_id text not null unique,                        -- LangGraph checkpointer key
    created_at timestamptz not null default now()
);

create table if not exists messages (
    id uuid primary key default uuid_generate_v4(),
    conversation_id uuid not null references conversations(id),
    role text not null check (role in ('user', 'assistant', 'tool')),
    content text not null,
    created_at timestamptz not null default now()
);

create table if not exists conversation_summaries (
    id uuid primary key default uuid_generate_v4(),
    conversation_id uuid not null references conversations(id),
    summary text not null,
    created_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------
-- Semantic memory — astronomy knowledge base (pgvector)
-- ---------------------------------------------------------------------

-- 1536 dims matches OpenAI text-embedding-3-small. If you embed with a
-- different model, change the dimension to match — pgvector requires a
-- fixed dimension per column.
create table if not exists knowledge_base (
    id uuid primary key default uuid_generate_v4(),
    content text not null,
    metadata jsonb,                     -- e.g. {"source": "messier_catalog", "object": "M31"}
    embedding vector(1536),
    created_at timestamptz not null default now()
);

create index if not exists idx_knowledge_base_embedding
    on knowledge_base using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- RPC function so the app calls one Supabase function instead of hand
-- writing a vector-search query — the search_knowledge_base tool will
-- call this via supabase.rpc(...).
create or replace function match_knowledge_base (
    query_embedding vector(1536),
    match_count int default 5
)
returns table (id uuid, content text, metadata jsonb, similarity float)
language sql stable
as $$
    select id, content, metadata,
           1 - (embedding <=> query_embedding) as similarity
    from knowledge_base
    order by embedding <=> query_embedding
    limit match_count;
$$;
