"""
AstroPlanner — Orchestrator (v0: one tool, Groq)
 
This is the smallest possible version of the "Conversation Agent" from
the architecture doc. It knows about exactly one tool (weather) so the
loop itself is easy to see clearly. Once this works, adding the other
four tools (visibility, recommendation, fov, scheduler) is just adding
more entries to TOOLS and TOOL_FUNCTIONS — the loop code below does not
change.
 
Requires: pip install groq
Requires: an env var GROQ_API_KEY (get one free at console.groq.com)
"""
