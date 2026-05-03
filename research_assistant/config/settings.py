"""
config/settings.py
Centralised config — loads from .env, exposes typed settings.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Memory ────────────────────────────────────────────────────────────
CHECKPOINT_DB_PATH: str = os.getenv("CHECKPOINT_DB_PATH", "./memory/checkpoints.db")

# ── Search ────────────────────────────────────────────────────────────
MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
SEARCH_TIMEOUT: int = int(os.getenv("SEARCH_TIMEOUT_SECONDS", "10"))

# ── Agent routing constants ───────────────────────────────────────────
AGENT_SEARCH     = "search_agent"
AGENT_SUMMARIZER = "summarizer"
AGENT_ANALYST    = "analyst"
AGENT_CITATION   = "citation_agent"
AGENT_END        = "__end__"

VALID_AGENTS = {AGENT_SEARCH, AGENT_SUMMARIZER, AGENT_ANALYST, AGENT_CITATION, AGENT_END}

# ── Graph limits ──────────────────────────────────────────────────────
MAX_ITERATIONS: int = 10   # prevent infinite supervisor loops
