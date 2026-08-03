"""
AstroPlanner — Local Embeddings (free, no API key)

Uses sentence-transformers' all-MiniLM-L6-v2, run LOCALLY in-process —
no OpenAI/Cohere/etc API call, no cost, no rate limit. Model is ~80MB,
downloaded once and cached by the library itself (~/.cache/torch/...),
so the only real cost is a one-time download + a few seconds of CPU
time per embedding call, both fine for Colab.

Kept separate from storage.py on purpose: storage.py's job is "how do we
persist/query things in Supabase", this file's job is "how do we turn
text into a vector" — a RAG pipeline swaps embedding models far more
often than it swaps databases, so this boundary keeps that swap to one
file.
"""

from typing import Optional

_model = None


def get_model():
    """
    Lazily loads the model once and reuses it — same 'build once' pattern
    as VisibilityEngine's ephemeris and storage.py's Supabase client.
    Loading sentence-transformers takes a few seconds; doing that on
    every embed_text() call would make search_knowledge_base painfully
    slow for no reason.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_text(text: str) -> list[float]:
    """Turns one string into a 384-dim vector. Used both to embed content
    when seeding the knowledge base, and to embed a user's question
    before searching it."""
    model = get_model()
    return model.encode(text, normalize_embeddings=True).tolist()


# ---------------------------------------------------------------------
# Seed data — a small starter set, not a full catalog scrape. Enough to
# prove the retrieval pipeline works end-to-end; expand this (or load
# from a real Messier/NGC descriptions file) once the mechanism is
# confirmed working.
# ---------------------------------------------------------------------

SEED_KNOWLEDGE = [
    {
        "content": (
            "The Orion Nebula (M42) is a diffuse nebula in the constellation Orion, "
            "one of the brightest nebulae visible to the naked eye. It's a stellar "
            "nursery where new stars are actively forming, located about 1,344 "
            "light-years away. Best observed in winter (Northern Hemisphere), it's "
            "an easy target even for small telescopes or binoculars."
        ),
        "metadata": {"source": "seed", "object": "M42", "type": "nebula"},
    },
    {
        "content": (
            "The Andromeda Galaxy (M31) is the nearest large galaxy to the Milky "
            "Way, about 2.5 million light-years away. It's visible to the naked eye "
            "under dark skies as a faint smudge, and is the most distant object "
            "generally visible without optical aid. Best observed in autumn "
            "(Northern Hemisphere)."
        ),
        "metadata": {"source": "seed", "object": "M31", "type": "galaxy"},
    },
    {
        "content": (
            "Astronomical seeing refers to the blurring and twinkling of "
            "astronomical objects caused by turbulence in the Earth's atmosphere. "
            "Poor seeing smears fine detail, which matters most for high-magnification "
            "targets like planets and double stars. Poor seeing does NOT necessarily "
            "mean the sky is cloudy — you can have clear, transparent skies with bad "
            "seeing, or hazy skies with good seeing."
        ),
        "metadata": {"source": "seed", "topic": "seeing"},
    },
    {
        "content": (
            "Sky transparency refers to how clear the atmosphere is — how much light "
            "from faint objects reaches your eye or camera without being scattered or "
            "absorbed by haze, humidity, or light pollution. Poor transparency mainly "
            "hurts faint deep-sky objects (galaxies, nebulae) more than bright ones "
            "like planets or the Moon."
        ),
        "metadata": {"source": "seed", "topic": "transparency"},
    },
    {
        "content": (
            "The Bortle scale rates night-sky darkness from 1 (excellent dark sky, "
            "Milky Way casts shadows) to 9 (inner-city sky, only the Moon and "
            "brightest planets visible). Most suburban skies fall around Bortle "
            "5-7. It's a rough, subjective scale rather than a precise measurement, "
            "but useful for planning which targets are realistically visible from a "
            "given location."
        ),
        "metadata": {"source": "seed", "topic": "bortle_scale"},
    },
]


def seed_knowledge_base() -> int:
    """
    Embeds and inserts SEED_KNOWLEDGE into Supabase. Safe to call more
    than once for testing — it doesn't check for duplicates, so re-running
    will insert copies. Fine for a small seed set; add a dedupe check
    (e.g. on metadata.object) before running this against real content.
    Returns the number of rows inserted.
    """
    import storage

    client = storage.get_client()
    rows = [
        {"content": entry["content"], "metadata": entry["metadata"], "embedding": embed_text(entry["content"])}
        for entry in SEED_KNOWLEDGE
    ]
    client.table("knowledge_base").insert(rows).execute()
    return len(rows)


if __name__ == "__main__":
    n = seed_knowledge_base()
    print(f"Inserted {n} seed knowledge_base rows.")
