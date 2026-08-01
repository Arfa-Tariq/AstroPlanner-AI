"""
AstroPlanner — Celestial Visibility Tool
 
Extracted from notebooks/03_celestial_visibility(v3).ipynb (the v3 version,
not the original — v3 fixes the UTC-noon search-window bug and honest
rise/set pairing, no reason to build on top of the buggy one).
 
Design note — the "build once, reuse" problem:
Unlike weather.py, this notebook has 3 expensive one-time setup steps that
must NOT be redone for every one of the 7 nights:
  1. Downloading/loading the NGC catalog (network + parsing ~14k rows)
  2. Loading the Skyfield ephemeris file (de421.bsp)
  3. Building the Astroplan Observer + FixedTarget list from the filtered
     catalog
 
The notebook already got this half-right (it builds these once, outside
the per-night loop) — this module keeps that shape, just wrapped as a
class so a caller (the orchestrator) can build it ONCE per conversation
and reuse it across multiple tool calls, instead of re-downloading the
catalog and ephemeris on every single question the user asks.
"""
