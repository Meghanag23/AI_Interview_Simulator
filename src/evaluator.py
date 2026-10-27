import os
import json
from dataclasses import dataclass
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

WEIGHTS = {
    "situation": 0.15,
    "task": 0.20,
    "action": 0.40,
    "result": 0.25
}


@dataclass
class STARScore:
    situation: int
    task: int
    action: int
    result: int
    overall: float
    feedback: str
    weakness: str


def extract_json(text: str) -> dict:
    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned.replace("```", "").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1:
        cleaned = cleaned[start:end + 1]

    return json.loads(cleaned)


def evaluate_answer(
    question_text: str,
    ideal_hints: str,
    candidate_answer: str,
    difficulty: str = "medium"
) -> STARScore:
    prompt = f"""
You are an expert behavioral interview evaluator trained on the STAR framework.

Evaluate the candidate answer using the STAR framework and the retrieved rubric.

QUESTION:
{question_text}

RETRIEVED RUBRIC:
{ideal_hints}

CANDIDATE ANSWER:
{candidate_answer}

DIFFICULTY:
{difficulty}

IMPORTANT SCORING INSTRUCTIONS:
- Score Situation based on whether the candidate explains the context/background.
- Score Task based on whether the candidate explains their responsibility or goal.
- Score Action based on whether the candidate explains specific steps they personally took.
- Score Result based on whether the candidate explains the outcome, impact, or measurable result.
- Do not give 1 unless the component is completely missing.
- If a component is present but not perfect, use 5-7.
- If a component is clear and specific, use 7-9.
- Use the retrieved rubric to guide feedback, but do not use rubric phrases as the weakness name.

SCORING SCALE:
1-3 = missing or extremely vague
4-5 = present but weak
6-7 = acceptable with some detail
8-9 = strong and specific
10 = exceptional with measurable impact

Return ONLY valid JSON with this exact structure:
{{
  "situation": 1,
  "task": 1,
  "action": 1,
  "result": 1,
  "feedback": "Two specific coaching sentences.",
  "weakness": "situation"
}}

The weakness value MUST be exactly one of:
"situation", "task", "action", "result"
"""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        data = extract_json(response.text)

        situation = max(1, min(10, int(data["situation"])))
        task = max(1, min(10, int(data["task"])))
        action = max(1, min(10, int(data["action"])))
        result = max(1, min(10, int(data["result"])))

        valid_weaknesses = {"situation", "task", "action", "result"}
        weakness = str(data.get("weakness", "result")).lower().strip()

        if weakness not in valid_weaknesses:
            score_map = {
                "situation": situation,
                "task": task,
                "action": action,
                "result": result
            }
            weakness = min(score_map, key=score_map.get)
    
        overall = (
            situation * WEIGHTS["situation"] +
            task * WEIGHTS["task"] +
            action * WEIGHTS["action"] +
            result * WEIGHTS["result"]
        )

        return STARScore(
            situation=situation,
            task=task,
            action=action,
            result=result,
            overall=round(overall, 2),
            feedback=data.get("feedback", ""),
            weakness=weakness
        )

    except Exception as error:
        print(f"Gemini evaluation failed: {error}")

        return STARScore(
            situation=5,
            task=5,
            action=5,
            result=5,
            overall=5.0,
            feedback="Evaluation fallback used because Gemini API was unavailable.",
            weakness="unknown"
        )


if __name__ == "__main__":
    score = evaluate_answer(
        question_text="Tell me about a time you resolved a conflict between two engineers.",
        ideal_hints="Active listening demonstrated; objective tie-breaker used; team cohesion maintained; quantified outcome stated",
        candidate_answer="At my previous company, two engineers disagreed about REST vs gRPC. I organized a design review, compared benchmark data, and helped the team choose gRPC because it reduced p99 latency by 35 percent. We launched on time.",
        difficulty="medium"
    )

    print(score)