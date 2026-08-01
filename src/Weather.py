"""
AstroPlanner — Weather Intelligence Tool
 
Extracted from notebooks/02_weather_intelligence.ipynb. This is the first
notebook converted into a plain, importable module rather than a Colab
script — no drive.mount(), no !pip install, no notebook-only state.
 
Design intent: this file has ZERO knowledge of LLMs, chat, or tool-calling.
It only knows how to compute a 7-night sky-quality outlook from a
UserProfile. The orchestrator (built separately) is what turns
get_weekly_sky_conditions into something an LLM can call — this module
just needs to be a clean, predictable function to call.
"""
