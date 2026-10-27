from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class QuestionRecord:
    """One asked question and its evaluation result."""
    question_id:    str
    question_text:  str
    topic:          str
    difficulty:     str
    candidate_answer: str
    star_scores:    Dict         # {situation, task, action, result, overall}
    weak_component: str
    timestamp:      str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SessionState:
    """
    Complete state of one interview session.
    Maintained across the entire agent loop.
    """
    session_id:         str
    candidate_role:     str          # e.g. "backend SDE"
    target_company:     str          # e.g. "Amazon"
    focus_areas:        List[str]    # e.g. ["leadership", "system_design"]

    # Current state
    current_difficulty: str = "medium"
    questions_asked:    List[str] = field(default_factory=list)
    question_records:   List[QuestionRecord] = field(default_factory=list)

    # Performance tracking
    scores_by_topic:    Dict[str, List[float]] = field(default_factory=dict)
    weak_components:    List[str] = field(default_factory=list)

    # Session limits
    max_questions:      int = 6
    is_complete:        bool = False

    def update_after_evaluation(self, record: QuestionRecord):
        """Update session state after each STAR evaluation."""
        self.questions_asked.append(record.question_id)
        self.question_records.append(record)
        self.weak_components.append(record.weak_component)

        # Track score by topic
        topic = record.topic
        if topic not in self.scores_by_topic:
            self.scores_by_topic[topic] = []
        self.scores_by_topic[topic].append(
            record.star_scores["overall"]
        )

        # Adaptive difficulty
        self._update_difficulty(record.star_scores["overall"])

        # Check if session is done
        if len(self.questions_asked) >= self.max_questions:
            self.is_complete = True

    def _update_difficulty(self, overall_score: float):
        """
        Adaptive difficulty logic:
        Score >= 8.0  → increase difficulty
        Score 5.0-7.9 → stay the same
        Score < 5.0   → decrease difficulty
        """
        if overall_score >= 8.0:
            if self.current_difficulty == "easy":
                self.current_difficulty = "medium"
            elif self.current_difficulty == "medium":
                self.current_difficulty = "hard"

        elif overall_score < 5.0:
            if self.current_difficulty == "hard":
                self.current_difficulty = "medium"
            elif self.current_difficulty == "medium":
                self.current_difficulty = "easy"

    def get_weakest_topic(self) -> Optional[str]:
        """Return topic with lowest average score."""
        if not self.scores_by_topic:
            return None
        return min(
            self.scores_by_topic,
            key=lambda t: sum(self.scores_by_topic[t]) / len(self.scores_by_topic[t])
        )

    def get_most_common_weak_component(self) -> Optional[str]:
        """Return the STAR component that was weakest most often."""
        if not self.weak_components:
            return None
        return max(set(self.weak_components), key=self.weak_components.count)

    def get_average_score(self) -> float:
        """Return overall average score across all questions."""
        all_scores = [
            r.star_scores["overall"]
            for r in self.question_records
        ]
        return round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0

    def build_final_report(self) -> Dict:
        """Generate the end-of-session performance report."""
        return {
            "session_id": self.session_id,
            "candidate_role": self.candidate_role,
            "target_company": self.target_company,
            "total_questions": len(self.question_records),
            "average_score": self.get_average_score(),
            "final_difficulty": self.current_difficulty,
            "scores_by_topic": {
                topic: round(sum(scores)/len(scores), 2)
                for topic, scores in self.scores_by_topic.items()
            },
            "weakest_topic": self.get_weakest_topic(),
            "most_common_weak_component": self.get_most_common_weak_component(),
            "recommendations": self._generate_recommendations(),
            "question_breakdown": [
                {
                    "question": r.question_text,
                    "topic": r.topic,
                    "scores": r.star_scores,
                    "feedback": r.star_scores.get("feedback", "")
                }
                for r in self.question_records
            ]
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate 3 specific recommendations based on performance."""
        recs = []
        weak_topic = self.get_weakest_topic()
        weak_component = self.get_most_common_weak_component()

        if weak_topic:
            recs.append(
                f"Practice more {weak_topic} questions — "
                f"your average score in this area was "
                f"{round(sum(self.scores_by_topic[weak_topic])/len(self.scores_by_topic[weak_topic]),1)}/10"
            )

        if weak_component == "result":
            recs.append(
                "Always end your answers with a quantified result — "
                "specific numbers, percentages, or timeframes make your "
                "impact concrete and memorable."
            )
        elif weak_component == "task":
            recs.append(
                "Be more explicit about YOUR specific responsibility. "
                "Say 'I was responsible for...' rather than 'we worked on...'"
            )
        elif weak_component == "action":
            recs.append(
                "Give more detail on the specific steps YOU took. "
                "Walk through your exact decision process."
            )
        elif weak_component == "situation":
            recs.append(
                "Set the context more clearly at the start — "
                "specify the company, team size, timeline, and what was at stake."
            )

        avg = self.get_average_score()
        if avg >= 8.0:
            recs.append("Strong overall performance. Focus on consistency "
                       "and adding more quantified results.")
        elif avg >= 6.0:
            recs.append("Good foundation. Record yourself answering and "
                       "practice until answers feel natural.")
        else:
            recs.append("Use the STAR framework as a checklist for every answer: "
                       "Situation → Task → Action → Result.")

        return recs