"""
api.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FastAPI server to expose the LangGraph research assistant to a frontend.
"""
from __future__ import annotations

import os
import uuid
import json
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from graph import build_graph, get_initial_state
from memory.checkpointer import list_sessions, delete_session, get_checkpointer

app = FastAPI(title="Research Assistant API")

# Mount frontend directory for static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Serve the main index.html at the root
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only. Restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the global graph
graph = build_graph(with_memory=True)
checkpointer = get_checkpointer()


class ChatRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None


@app.get("/api/sessions")
def get_sessions():
    """List all saved sessions with their first query as the title."""
    session_ids = list_sessions(checkpointer)
    sessions = []
    for sid in session_ids:
        title = sid  # fallback to session ID
        try:
            config = {"configurable": {"thread_id": sid}}
            state = graph.get_state(config)
            if state and hasattr(state, 'values') and state.values:
                query = state.values.get("query", "")
                if query:
                    title = query
        except Exception:
            pass
        sessions.append({"thread_id": sid, "title": title})
    return {"sessions": sessions}


@app.delete("/api/sessions/{thread_id}")
def delete_session_endpoint(thread_id: str):
    """Delete a session."""
    success = delete_session(checkpointer, thread_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or error deleting")
    return {"status": "deleted"}


@app.get("/api/sessions/{thread_id}")
def get_session_history(thread_id: str):
    """Retrieve the current state of a session."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        # get_state returns a StateSnapshot
        state = graph.get_state(config)
        if not state or not hasattr(state, 'values') or not state.values:
             return {"messages": [], "final_answer": None}
             
        # Extract messages (as strings/dicts for UI)
        messages_data = []
        if "messages" in state.values:
             for msg in state.values["messages"]:
                 messages_data.append({
                     "type": msg.__class__.__name__,
                     "content": msg.content
                 })
                 
        return {
            "query": state.values.get("query", ""),
            "messages": messages_data,
            "final_answer": state.values.get("final_answer"),
            "search_results": state.values.get("search_results", [])
        }
    except Exception as e:
        print(f"Error fetching state: {e}")
        return {"messages": [], "final_answer": None}


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Start or continue a chat session.
    Streams back Server-Sent Events (SSE) representing agent activity,
    followed by the final answer.
    """
    thread_id = request.thread_id or f"session-{uuid.uuid4().hex[:8]}"
    query = request.query

    async def event_generator():
        initial_state = get_initial_state(query)
        config = {"configurable": {"thread_id": thread_id}}
        
        # Send thread_id to client immediately
        yield f"event: thread_id\ndata: {thread_id}\n\n"

        try:
            # We run the synchronous graph.stream in a separate thread if needed,
            # but LangGraph stream is generally fine here if not blocking the whole event loop.
            # Using asyncio.to_thread for better concurrency.
            def run_graph():
                return list(graph.stream(initial_state, config=config, stream_mode="updates"))

            steps = await asyncio.to_thread(run_graph)
            
            final_answer = "No answer produced."
            for step in steps:
                for node_name, update in step.items():
                    if node_name == "__end__":
                        continue

                    # Determine status
                    status_parts = []
                    if update.get("search_results"):
                        status_parts.append(f"{len(update['search_results'])} results found")
                    if update.get("summaries"):
                        status_parts.append("summaries ready")
                    if update.get("analysis"):
                        status_parts.append("analysis complete")
                    if update.get("final_answer"):
                        status_parts.append("final answer ready")
                        final_answer = update["final_answer"]
                    if update.get("next_agent"):
                        status_parts.append(f"routing to {update['next_agent']}")

                    status = " | ".join(status_parts) if status_parts else "processing..."
                    
                    # Send update event
                    event_data = {
                        "agent": node_name,
                        "status": status
                    }
                    yield f"event: update\ndata: {json.dumps(event_data)}\n\n"
                    
                    # Small delay to allow frontend rendering to breathe
                    await asyncio.sleep(0.05)

            # Send final answer
            yield f"event: final_answer\ndata: {json.dumps({'answer': final_answer})}\n\n"
            
        except Exception as e:
            error_msg = f"Error during graph execution: {str(e)}"
            print(error_msg)
            yield f"event: error\ndata: {json.dumps({'error': error_msg})}\n\n"

        yield "event: close\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
