"""
tools/python_repl.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Safe Python REPL tool — lets the Analyst agent run calculations,
parse data, or do quick numeric reasoning during research.

Safety model:
  - Runs in a restricted exec() environment
  - Blocks imports of os, sys, subprocess, socket (shell escape)
  - 5-second execution timeout via threading
  - Captures stdout; returns result or error string
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import io
import sys
import threading
from contextlib import redirect_stdout
from langchain_core.tools import tool

# Modules the REPL is NOT allowed to import
_BLOCKED = {"os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib"}

EXEC_TIMEOUT = 5  # seconds


def _safe_exec(code: str, result_holder: list) -> None:
    """Execute code in a restricted namespace, capture stdout."""
    restricted_builtins = {
        k: v for k, v in __builtins__.items()
        if k not in ("__import__", "eval", "exec", "compile", "open")
    } if isinstance(__builtins__, dict) else {}

    namespace: dict = {"__builtins__": restricted_builtins}

    # Block dangerous imports via custom __import__
    def safe_import(name, *args, **kwargs):
        base = name.split(".")[0]
        if base in _BLOCKED:
            raise ImportError(f"Import of '{name}' is blocked in the REPL for safety.")
        return __import__(name, *args, **kwargs)

    if isinstance(__builtins__, dict):
        namespace["__builtins__"]["__import__"] = safe_import

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(compile(code, "<repl>", "exec"), namespace)  # noqa: S102
        result_holder.append(buf.getvalue() or "(code ran — no output)")
    except Exception as e:
        result_holder.append(f"Error: {type(e).__name__}: {e}")


@tool
def python_repl(code: str) -> str:
    """
    Execute Python code and return the output.
    Use this for calculations, data parsing, or numeric analysis.
    Dangerous imports (os, sys, subprocess) are blocked.

    Args:
        code: Valid Python code to execute.

    Returns:
        stdout output as a string, or an error message.
    """
    result: list[str] = []
    thread = threading.Thread(target=_safe_exec, args=(code, result), daemon=True)
    thread.start()
    thread.join(timeout=EXEC_TIMEOUT)

    if thread.is_alive():
        return f"Error: Code execution timed out after {EXEC_TIMEOUT}s."
    return result[0] if result else "(no output)"


# ── All REPL tools exported ───────────────────────────────────────────
REPL_TOOLS = [python_repl]
