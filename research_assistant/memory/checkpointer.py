"""
memory/checkpointer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SQLite-backed checkpointing for LangGraph.

What this gives you:
  - Every graph step is checkpointed to SQLite automatically
  - Resuming a session by thread_id loads the full prior state
  - Multi-turn conversations: user can ask follow-ups and the
    agent remembers all prior search results, analysis, etc.
  - Crash recovery: if the process dies mid-run, restart with
    the same thread_id and it picks up where it left off

Key concept — thread_id:
  Each conversation is identified by a thread_id (e.g. "session-abc123").
  Pass {"configurable": {"thread_id": thread_id}} to graph.invoke()
  and LangGraph handles the rest.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from config.settings import CHECKPOINT_DB_PATH


def get_checkpointer() -> SqliteSaver:
    """
    Returns a SqliteSaver instance connected to the configured DB path.
    Creates the directory and DB file if they don't exist.

    Usage:
        checkpointer = get_checkpointer()
        graph = build_graph().compile(checkpointer=checkpointer)

        # First turn
        result = graph.invoke(
            {"query": "What is LangGraph?", ...},
            config={"configurable": {"thread_id": "session-001"}}
        )

        # Follow-up — state is automatically restored
        result = graph.invoke(
            {"query": "Tell me more about its memory system", ...},
            config={"configurable": {"thread_id": "session-001"}}
        )
    """
    db_dir = os.path.dirname(CHECKPOINT_DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    return SqliteSaver(conn=conn)


def list_sessions(checkpointer: SqliteSaver) -> list[str]:
    """
    List all thread_ids that have checkpoints stored.
    Useful for showing a user their conversation history.
    """
    try:
        # SqliteSaver stores checkpoints keyed by (thread_id, checkpoint_id)
        conn = checkpointer.conn
        cursor = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        )
        return [row[0] for row in cursor.fetchall()]
    except Exception:
        return []


def delete_session(checkpointer: SqliteSaver, thread_id: str) -> bool:
    """Delete all checkpoints for a session thread_id."""
    try:
        conn = checkpointer.conn
        conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        conn.commit()
        return True
    except Exception:
        return False
