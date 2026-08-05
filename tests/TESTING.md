# Testing AstroPlanner AI

Two separate layers, deliberately kept apart:

1. **`tests/`** — pure-function tests for `fov.py`, `scheduler.py`,
   `recommendation.py`. No network, no API keys, no LLM, no Supabase.
   These check the *math* is right.
2. **`chat_cli.py`** — talks to the real `orchestrator.chat()`, i.e. the
   whole stack (Groq + Supabase + the 5-stage pipeline). This checks the
   LLM actually routes to the right tool and the pieces talk to each
   other correctly.

Run them in that order. If layer 1 fails, don't bother debugging layer 2
yet — you'd be chasing an LLM routing issue when the bug is actually in
plain Python math.

## File layout

Drop these into your repo root like this:

```
AstroPlanner-AI/
├── src/                  (already exists)
├── db/                   (already exists)
├── tests/
│   ├── conftest.py
│   ├── test_fov.py
│   ├── test_scheduler.py
│   └── test_recommendation.py
├── chat_cli.py
└── TESTING.md            (this file)
```

## Layer 1 — pure-function tests

```bash
pip install pytest
pip install -r requirements.txt   # pydantic etc. — models.py needs it
pytest tests/ -v
```

No `.env`, no Supabase, no Groq key needed for this layer — `conftest.py`
just puts `src/` on `sys.path` and builds fake `UserProfile` objects.

What's covered:
- `fov.py`: FoV/pixel-scale math, sampling bands, deep-sky fit
  thresholds (too_large / too_small / fits_well / unknown), solar-system
  handling, the no-camera degrade path.
- `scheduler.py`: the mixed time-format parser (`HH:MM`, `MM/DD HH:MM`,
  non-parseable labels), the peak-centered fallback window, overlap
  rejection when two objects' slots collide, solar-system-daytime →
  bonus routing.
- `recommendation.py`: every per-factor scorer in isolation (visibility,
  weather, moon, equipment, light pollution), the preference multiplier,
  and the composite `score_object` — including that an unknown Bortle
  scale correctly *drops* the light-pollution factor instead of treating
  it as worst-case.

What's **not** covered here on purpose: `weather.py` (network calls to
Open-Meteo/7Timer) and `visibility.py` (Skyfield ephemeris + NGC catalog
download). Those are integration-shaped, not pure-function-shaped — if
you want to test them without hitting the network, record one real
response with `responses`/`vcr.py` and replay it, rather than mocking
every field by hand.

## Layer 2 — the actual chatbot

### One-time setup

```bash
cp env.example .env
# fill in SUPABASE_URL, SUPABASE_KEY, GROQ_API_KEY, DATABASE_URL
pip install -r requirements.txt
```

Run everything from the repo root so `./src` is on the path and `.env`
is found.

### A. Chat with it yourself (this is "entering your own input")

```bash
python chat_cli.py
```

You'll get a plain prompt:

```
You: Plan tonight at lat 33.2, lon 32.4, 8-inch, 1000mm, beginner
Assistant: ...
You: Why wasn't NGC1234 recommended?
Assistant: ...
```

Type anything — it's a normal REPL, one line in, one reply out, same
`thread_id` (so it remembers earlier turns) until you `/new`.

Useful in-chat commands:
| Command | What it does |
|---|---|
| `/sessions` | lists your recent observation sessions from Supabase |
| `/session <id>` | dumps the full stored pipeline output for one session |
| `/new` | starts a fresh thread — clears conversation memory |
| `/exit` | quits |

Things worth typing manually to cover what the automated smoke test
only checks loosely:
- `"Plan my session for tonight at lat X, lon Y, <aperture>mm, <focal length>mm, <level>"`
- `"Why wasn't <object name> recommended?"`
- `"Compare today's plan with yesterday's"` (no dedicated tool exists for
  this yet — see what the model does; it should fall back to two
  `get_session_context` calls it reasons over, or ask which sessions)
- `"Regenerate that with only 2 hours" / "only 3 targets"`
- `"Tell me about the Orion Nebula"` (should hit the knowledge base, not
  the session data)
- `"What have I planned recently?"`
- Ask something with no session on file yet (fresh `/new` thread) to
  confirm it doesn't hallucinate a session_id.

### B. Automated smoke test (covers every tool once, hands-off)

```bash
python chat_cli.py --smoke
```

Runs `create_observation_plan → get_session_context → regenerate_schedule
→ revise_observation_plan → get_recent_sessions → search_knowledge_base`
in sequence and prints each reply plus a loose pass/fail. **Read the
replies** — the checks are just keyword/length heuristics, not real
correctness checks. After it runs, verify in Supabase:
- `observation_sessions` gained 2 new rows (the create + the revise),
  and the revised one has `revision_number = 2`.
- `weather_analyses` / `visibility_analyses` / `recommendation_analyses`
  / `fov_analyses` / `observation_schedules` each have a row per session.

### C. Long-conversation / summarization test

```bash
python chat_cli.py --long-convo
```

Sends 13 filler messages on one thread (past the 12-message threshold in
`orchestrator.SUMMARIZE_AFTER_N_MESSAGES`) and checks that
`conversation_summaries` actually got a row — this is the mechanism that
keeps token usage from growing unbounded turn over turn. If it fails,
check `_summarize_and_trim` in `orchestrator.py` for errors first.

### D. Testing the HTTP layer instead of calling Python directly

```bash
uvicorn api:app --reload --port 8000
```

then in another terminal:

```bash
curl -X POST localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_name": "andrew", "thread_id": "http-test-1", "message": "Plan tonight at lat 33.2, lon 32.4, 8-inch, 1000mm, beginner"}'

curl localhost:8000/users/andrew/sessions
curl localhost:8000/sessions/<session_id from above>
```

This exercises `api.py` itself (CORS, request/response models, error
handling) rather than assuming it's a transparent passthrough to
`orchestrator.chat()`.

## What "properly tested" looks like before you trust it

- [ ] `pytest tests/ -v` — all green.
- [ ] `chat_cli.py --smoke` — every step produces a reasonable reply,
      and Supabase rows match expectations (2 sessions, all 5 stage
      tables populated for each).
- [ ] `chat_cli.py --long-convo` — summary row appears.
- [ ] A manual interactive session covering: brand-new plan, a "why"
      question, a regenerate request, a revise request, a knowledge-base
      question, and a question with no prior session (fresh `/new`
      thread) to confirm it doesn't invent data.
- [ ] The HTTP layer (`uvicorn` + `curl`), at least once, so you're not
      only testing the Python-import path.
- [ ] Two different `user_name`s in two different threads never see each
      other's `/sessions` output.
