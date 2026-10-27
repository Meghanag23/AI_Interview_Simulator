import os
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from src.agent import create_session, run_agent_turn, _sessions

load_dotenv()

app = FastAPI(
    title="AI Interview Simulator",
    description="Autonomous RAG-powered behavioral interview coach",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Request/Response models ────────────────────────────────────────
class StartSessionRequest(BaseModel):
    candidate_role:  str      # e.g. "Backend SDE"
    target_company:  str      # e.g. "Amazon"
    focus_areas:     List[str] = ["leadership", "delivery"]

class StartSessionResponse(BaseModel):
    session_id:  str
    message:     str

class AnswerRequest(BaseModel):
    session_id:      str
    candidate_answer: str

class AgentResponse(BaseModel):
    session_id:         str
    agent_message:      str
    session_complete:   bool
    questions_asked:    int

# ── Endpoints ──────────────────────────────────────────────────────
@app.post("/session/start", response_model=StartSessionResponse)
async def start_session(request: StartSessionRequest):
    """
    Start a new interview session.
    Returns a session_id to use in subsequent requests.
    """
    session_id = create_session(
        candidate_role=request.candidate_role,
        target_company=request.target_company,
        focus_areas=request.focus_areas
    )

    # Trigger first question immediately
    first_response = run_agent_turn(
        session_id=session_id,
        candidate_input="START"
    )

    return StartSessionResponse(
        session_id=session_id,
        message=first_response
    )

@app.post("/session/answer", response_model=AgentResponse)
async def submit_answer(request: AnswerRequest):
    """
    Submit a candidate answer. Agent evaluates it and
    retrieves the next question adaptively.
    """
    session = _sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    response_text = run_agent_turn(
        session_id=request.session_id,
        candidate_input=request.candidate_answer
    )

    # Refresh session state after agent runs
    session = _sessions.get(request.session_id)

    return AgentResponse(
        session_id=request.session_id,
        agent_message=response_text,
        session_complete=session.is_complete,
        questions_asked=len(session.questions_asked)
    )

@app.get("/session/{session_id}/report")
async def get_report(session_id: str):
    """Get the final performance report for a completed session."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.build_final_report()

@app.get("/health")
async def health_check():
    return {"status": "running", "version": "1.0.0"}