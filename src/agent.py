# src/agent.py
# Direct interview controller.
# This version avoids OpenAI/LangChain so the FastAPI demo works without OpenAI quota.

import uuid
from typing import Dict, Optional

from src.retriever import RAGRetriever, RetrievedQuestion
from src.evaluator import evaluate_answer
from src.session import SessionState, QuestionRecord


# Global session storage
_sessions: Dict[str, SessionState] = {}

# Stores the current active question for each session
_current_questions: Dict[str, RetrievedQuestion] = {}

# Initialize retriever once
rag_retriever = RAGRetriever()


def create_session(
    candidate_role: str,
    target_company: str,
    focus_areas: list
) -> str:
    """
    Create a new interview session and return its session_id.
    """
    session_id = str(uuid.uuid4())[:8]

    _sessions[session_id] = SessionState(
        session_id=session_id,
        candidate_role=candidate_role,
        target_company=target_company,
        focus_areas=focus_areas
    )

    return session_id


def build_retrieval_query(session: SessionState) -> str:
    """
    Build a query for RAG retrieval based on current session state.
    """
    focus_text = " ".join(session.focus_areas)

    if hasattr(session, "weakest_topic"):
        weakest_topic = session.weakest_topic()
    elif hasattr(session, "get_weakest_topic"):
        weakest_topic = session.get_weakest_topic()
    else:
        weakest_topic = None
        
    if weakest_topic:
        return (
            f"{session.candidate_role} {session.target_company} "
            f"{weakest_topic} behavioral interview question"
        )

    return (
        f"{session.candidate_role} {session.target_company} "
        f"{focus_text} behavioral interview question"
    )


def infer_role_filter(candidate_role: str) -> str:
    """
    Convert candidate role text into a role metadata filter.
    """
    role_lower = candidate_role.lower()

    if "backend" in role_lower:
        return "backend"
    if "frontend" in role_lower:
        return "frontend"
    if "fullstack" in role_lower or "full stack" in role_lower:
        return "fullstack"

    return "sde"

def get_session_difficulty(session: SessionState) -> str:
    """
    Support both possible SessionState field names.
    """
    if hasattr(session, "difficulty"):
        return session.difficulty
    if hasattr(session, "current_difficulty"):
        return session.current_difficulty
    return "medium"

def select_next_question(session: SessionState) -> Optional[RetrievedQuestion]:
    """
    Select next question using RAG retrieval.
    """
    query = build_retrieval_query(session)
    role_filter = infer_role_filter(session.candidate_role)

    results = rag_retriever.retrieve(
        query=query,
        n_results=3,
        difficulty=get_session_difficulty(session),
        role=role_filter,
        exclude_ids=session.questions_asked
    )

    if not results:
        results = rag_retriever.retrieve(
            query=query,
            n_results=3,
            difficulty=session.difficulty,
            role=None,
            exclude_ids=session.questions_asked
        )

    if not results:
        results = rag_retriever.retrieve(
            query=query,
            n_results=3,
            difficulty=None,
            role=None,
            exclude_ids=session.questions_asked
        )

    if not results:
        return None

    return results[0]


def run_agent_turn(session_id: str, candidate_input: str) -> str:
    """
    Run one turn of the interview.

    START:
      retrieve first question and ask it

    Candidate answer:
      evaluate answer
      update session
      adapt difficulty
      retrieve next question
    """
    session = _sessions.get(session_id)

    if not session:
        return "Session not found. Please start a new session."

    if session.is_complete:
        return format_final_report(session)

    # First turn
    if candidate_input.strip().upper() == "START":
        question = select_next_question(session)

        if not question:
            return "Could not retrieve a question. Please check questions.json and ChromaDB index."

        _current_questions[session_id] = question

        return (
            f"Welcome to your AI Interview Simulator session.\n\n"
            f"Role: {session.candidate_role}\n"
            f"Company: {session.target_company}\n"
            f"Current difficulty: {get_session_difficulty(session)}\n\n"
            f"Question 1:\n{question.question_text}"
        )

    # Candidate answer turn
    current_question = _current_questions.get(session_id)

    if not current_question:
        question = select_next_question(session)

        if not question:
            return "No active question found and no new question could be retrieved."

        _current_questions[session_id] = question
        return f"Let us start with this question:\n\n{question.question_text}"

    score = evaluate_answer(
        question_text=current_question.question_text,
        ideal_hints=current_question.ideal_hints,
        candidate_answer=candidate_input,
        difficulty=current_question.difficulty
    )

    record = QuestionRecord(
        question_id=current_question.question_id,
        question_text=current_question.question_text,
        topic=current_question.topic,
        difficulty=current_question.difficulty,
        candidate_answer=candidate_input,
        star_scores={
            "situation": score.situation,
            "task": score.task,
            "action": score.action,
            "result": score.result,
            "overall": score.overall,
            "feedback": score.feedback
        },
        weak_component=score.weakness
    )

    if hasattr(session, "update_after_evaluation"):
        session.update_after_evaluation(record)
    elif hasattr(session, "update"):
        session.update(record)
    else:
        raise AttributeError("SessionState has no update method.")

    feedback = (
        f"STAR Evaluation:\n"
        f"Situation: {score.situation}/10\n"
        f"Task: {score.task}/10\n"
        f"Action: {score.action}/10\n"
        f"Result: {score.result}/10\n"
        f"Overall: {score.overall}/10\n\n"
        f"Feedback: {score.feedback}\n"
        f"Weakest component: {score.weakness}\n\n"
        f"Updated difficulty: {get_session_difficulty(session)}\n"
        f"Questions completed: {len(session.questions_asked)}/{session.max_questions}\n"
    )

    if session.is_complete:
        return feedback + "\n" + format_final_report(session)

    next_question = select_next_question(session)

    if not next_question:
        return feedback + "\nNo more questions could be retrieved."

    _current_questions[session_id] = next_question

    next_question_number = len(session.questions_asked) + 1

    return (
        feedback
        + f"\nNext Question {next_question_number}:\n"
        + next_question.question_text
    )


def run_turn(session_id: str, candidate_input: str) -> str:
    """
    Compatibility alias.
    """
    return run_agent_turn(session_id, candidate_input)


def format_final_report(session: SessionState) -> str:
    """
    Format final report as readable text.
    """
    report = session.build_final_report()

    return (
        f"\nFinal Interview Report\n"
        f"Session ID: {report['session_id']}\n"
        f"Role: {report['candidate_role']}\n"
        f"Company: {report['target_company']}\n"
        f"Questions answered: {report['questions_answered']}\n"
        f"Average score: {report['average_score']}/10\n"
        f"Weakest topic: {report['weakest_topic']}\n"
        f"Weakest STAR component: {report['weakest_component']}\n"
        f"Top improvement tip: {report['top_tip']}\n"
    )