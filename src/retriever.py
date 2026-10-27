import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

from src.database import (
    get_chroma_client, get_or_create_collection,
    embed_text, tokenize_for_bm25,
    build_bm25_index
)

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@dataclass
class RetrievedQuestion:
    """
    Represents one retrieved question with all metadata.
    This is what the agent receives.
    """
    question_id:  str
    question_text: str
    topic:        str
    difficulty:   str
    role:         str
    ideal_hints:  str
    follow_up:    str
    score:        float          # similarity score from retrieval
    source:       str            # "rag" or "bm25"


class RAGRetriever:
    """
    Dense vector retrieval using ChromaDB + OpenAI embeddings.
    Semantic search — understands meaning, not just keywords.
    """

    def __init__(self, collection_name: str = "interview_questions"):
        self.client     = get_chroma_client()
        self.collection = get_or_create_collection(
            self.client, collection_name
        )

    def retrieve(
        self,
        query: str,
        n_results: int = 3,
        difficulty: Optional[str] = None,
        role: Optional[str] = None,
        exclude_ids: Optional[List[str]] = None
    ) -> List[RetrievedQuestion]:
        """
        Retrieve top-n most semantically similar questions.

        Args:
            query:       What to search for (candidate's role description
                         or topic they want to practice)
            n_results:   How many questions to return
            difficulty:  Filter by "easy", "medium", or "hard"
            role:        Filter by role string (e.g. "sde", "backend")
            exclude_ids: Question IDs already asked this session
        """

        # Embed the query
        query_vector = embed_text(query)

        # Build metadata filter
        # ChromaDB supports exact metadata filters like $eq.
        # It does not support $contains in this version, so role is filtered manually later.
        if difficulty:
            where = {"difficulty": {"$eq": difficulty}}
        else:
            where = None

        # Retrieve more than needed if we need to exclude IDs
        fetch_n = n_results + len(exclude_ids or []) + 10
        fetch_n = min(fetch_n, self.collection.count())
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=fetch_n,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"ChromaDB query error: {e}")
            return []

        # Parse results and filter excluded IDs
        retrieved = []
        ids        = results["ids"][0]
        documents  = results["documents"][0]
        metadatas  = results["metadatas"][0]
        distances  = results["distances"][0]

        # Role is stored as comma-separated metadata like "sde,backend,fullstack"
        # Since ChromaDB does not support $contains here, filter role manually.
        for qid, doc, meta, dist in zip(ids, documents, metadatas, distances):
            if exclude_ids and qid in exclude_ids:
                continue
            if role:
                role_metadata = meta.get("role", "")
                role_list = [r.strip().lower() for r in role_metadata.split(",")]
                if role.lower() not in role_list:
                    continue
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 - (distance/2)
            similarity = round(1 - (dist / 2), 4)

            retrieved.append(RetrievedQuestion(
                question_id=qid,
                question_text=doc,
                topic=meta.get("topic", ""),
                difficulty=meta.get("difficulty", "medium"),
                role=meta.get("role", ""),
                ideal_hints=meta.get("ideal_hints", ""),
                follow_up=meta.get("follow_up", ""),
                score=similarity,
                source="rag"
            ))

            if len(retrieved) >= n_results:
                break

        return retrieved


class BM25Retriever:
    """
    Sparse keyword retrieval using BM25Okapi.
    Strong on exact-term matches, zero semantic understanding.
    This is the BASELINE for your comparison experiment.
    """

    def __init__(self, questions_path: str = "data/questions.json"):
        self.bm25, self.questions = build_bm25_index(questions_path)
        self.questions_map = {
            q["question_id"]: q for q in self.questions
        }

    def retrieve(
        self,
        query: str,
        n_results: int = 3,
        difficulty: Optional[str] = None,
        role: Optional[str] = None,
        exclude_ids: Optional[List[str]] = None
    ) -> List[RetrievedQuestion]:
        """
        Retrieve top-n questions by BM25 keyword matching.
        """
        query_tokens = tokenize_for_bm25(query)
        scores       = self.bm25.get_scores(query_tokens)

        # Get ranked indices (highest score first)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        retrieved = []
        for idx in ranked_indices:
            q    = self.questions[idx]
            qid  = q["question_id"]
            score = round(float(scores[idx]), 4)

            # Apply filters manually (BM25 doesn't have metadata filtering)
            if difficulty and q.get("difficulty") != difficulty:
                continue
            if role and role not in q.get("role", []):
                continue
            if exclude_ids and qid in exclude_ids:
                continue
            if score == 0.0:
                continue

            retrieved.append(RetrievedQuestion(
                question_id=qid,
                question_text=q["question_text"],
                topic=q.get("topic", ""),
                difficulty=q.get("difficulty", "medium"),
                role=",".join(q.get("role", [])),
                ideal_hints=q.get("ideal_hints", ""),
                follow_up=q.get("follow_up", ""),
                score=score,
                source="bm25"
            ))

            if len(retrieved) >= n_results:
                break

        return retrieved
    
if __name__ == "__main__":
    print("=== Testing RAG Retriever ===")

    rag = RAGRetriever()

    rag_results = rag.retrieve(
        query="handling disagreement between engineers",
        n_results=3,
        difficulty="medium",
        role="sde"
    )

    if not rag_results:
        print("No RAG results found.")
    else:
        for result in rag_results:
            print(
                f"[{result.score}] {result.question_id} | "
                f"{result.topic} | {result.difficulty} | "
                f"{result.question_text}"
            )

    print("\n=== Testing BM25 Retriever ===")

    bm25 = BM25Retriever()

    bm25_results = bm25.retrieve(
        query="handling disagreement between engineers",
        n_results=3,
        difficulty="medium",
        role="sde"
    )

    if not bm25_results:
        print("No BM25 results found.")
    else:
        for result in bm25_results:
            print(
                f"[{result.score}] {result.question_id} | "
                f"{result.topic} | {result.difficulty} | "
                f"{result.question_text}"
            )