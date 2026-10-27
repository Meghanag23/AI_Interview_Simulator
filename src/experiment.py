import json
import time
from typing import List, Dict
from dataclasses import dataclass, asdict

from src.retriever import RAGRetriever, BM25Retriever
from src.database import build_bm25_index

# ── Ground truth test set ──────────────────────────────────────────
# 30 test queries with manually labeled relevant questions
TEST_QUERIES = [
    {
        "query_id": "q001",
        "query": "conflict resolution between engineers",
        "relevant_ids": ["q_sde_leadership_001"],
        "difficulty": "medium"
    },
    {
        "query_id": "q002",
        "query": "leadership without authority influencing team",
        "relevant_ids": ["q_sde_leadership_002"],
        "difficulty": "hard"
    },
    {
        "query_id": "q003",
        "query": "handling disagreement on technical approach",
        "relevant_ids": ["q_sde_leadership_001", "q_sde_leadership_002"],
        "difficulty": "medium"
    },
    {
        "query_id": "q004",
        "query": "learning from failure and mistakes",
        "relevant_ids": ["q_sde_failure_001"],
        "difficulty": "medium"
    },
    {
        "query_id": "q005",
        "query": "delivering project under pressure tight deadline",
        "relevant_ids": ["q_sde_delivery_001"],
        "difficulty": "medium"
    },
    # Add 25 more queries covering all topics in your database
]

@dataclass
class ExperimentResult:
    query_id:       str
    query_text:     str
    method:         str          # "rag" or "bm25"
    retrieved_ids:  List[str]
    relevant_ids:   List[str]
    precision_at_3: float
    mrr:            float
    latency_ms:     float


def precision_at_k(retrieved: List[str], relevant: List[str], k: int = 3) -> float:
    """
    Fraction of top-k retrieved results that are relevant.
    P@3 = |relevant ∩ top-3| / 3
    """
    top_k = retrieved[:k]
    hits  = sum(1 for r in top_k if r in relevant)
    return round(hits / k, 4)


def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """
    1 / rank of the first relevant result.
    MRR = 1.0 if first result is relevant.
    MRR = 0.5 if second result is relevant.
    MRR = 0.0 if no relevant result found.
    """
    for i, rid in enumerate(retrieved):
        if rid in relevant:
            return round(1 / (i + 1), 4)
    return 0.0


def run_experiment(questions_path: str = "data/questions.json") -> Dict:
    """
    Run the full comparison experiment.
    Returns summary statistics and per-query results.
    """
    rag     = RAGRetriever()
    bm25    = BM25Retriever(questions_path)
    results = []

    print("Running comparison experiment: RAG vs BM25")
    print(f"Total queries: {len(TEST_QUERIES)}")
    print("-" * 50)

    for tq in TEST_QUERIES:
        # RAG retrieval
        start = time.time()
        rag_results = rag.retrieve(query=tq["query"], n_results=3)
        rag_latency = (time.time() - start) * 1000

        rag_ids = [r.question_id for r in rag_results]
        rag_p3  = precision_at_k(rag_ids, tq["relevant_ids"])
        rag_mrr = mean_reciprocal_rank(rag_ids, tq["relevant_ids"])

        results.append(ExperimentResult(
            query_id=tq["query_id"],
            query_text=tq["query"],
            method="rag",
            retrieved_ids=rag_ids,
            relevant_ids=tq["relevant_ids"],
            precision_at_3=rag_p3,
            mrr=rag_mrr,
            latency_ms=round(rag_latency, 2)
        ))

        # BM25 retrieval
        start = time.time()
        bm25_results = bm25.retrieve(query=tq["query"], n_results=3)
        bm25_latency = (time.time() - start) * 1000

        bm25_ids = [r.question_id for r in bm25_results]
        bm25_p3  = precision_at_k(bm25_ids, tq["relevant_ids"])
        bm25_mrr = mean_reciprocal_rank(bm25_ids, tq["relevant_ids"])

        results.append(ExperimentResult(
            query_id=tq["query_id"],
            query_text=tq["query"],
            method="bm25",
            retrieved_ids=bm25_ids,
            relevant_ids=tq["relevant_ids"],
            precision_at_3=bm25_p3,
            mrr=bm25_mrr,
            latency_ms=round(bm25_latency, 2)
        ))

        # Print per-query comparison
        print(f"Query: '{tq['query']}'")
        print(f"  RAG  → P@3: {rag_p3:.2f}  MRR: {rag_mrr:.2f}  ({rag_latency:.0f}ms)")
        print(f"  BM25 → P@3: {bm25_p3:.2f}  MRR: {bm25_mrr:.2f}  ({bm25_latency:.0f}ms)")
        print()

    # Calculate summary statistics
    rag_results_only  = [r for r in results if r.method == "rag"]
    bm25_results_only = [r for r in results if r.method == "bm25"]

    summary = {
        "total_queries": len(TEST_QUERIES),
        "rag": {
            "mean_precision_at_3": round(
                sum(r.precision_at_3 for r in rag_results_only) / len(rag_results_only), 4
            ),
            "mean_mrr": round(
                sum(r.mrr for r in rag_results_only) / len(rag_results_only), 4
            ),
            "mean_latency_ms": round(
                sum(r.latency_ms for r in rag_results_only) / len(rag_results_only), 1
            )
        },
        "bm25": {
            "mean_precision_at_3": round(
                sum(r.precision_at_3 for r in bm25_results_only) / len(bm25_results_only), 4
            ),
            "mean_mrr": round(
                sum(r.mrr for r in bm25_results_only) / len(bm25_results_only), 4
            ),
            "mean_latency_ms": round(
                sum(r.latency_ms for r in bm25_results_only) / len(bm25_results_only), 1
            )
        },
        "per_query_results": [asdict(r) for r in results]
    }

    # Print final summary
    print("=" * 50)
    print("EXPERIMENT SUMMARY")
    print("=" * 50)
    print(f"RAG  → Mean P@3: {summary['rag']['mean_precision_at_3']}  "
          f"Mean MRR: {summary['rag']['mean_mrr']}  "
          f"Avg latency: {summary['rag']['mean_latency_ms']}ms")
    print(f"BM25 → Mean P@3: {summary['bm25']['mean_precision_at_3']}  "
          f"Mean MRR: {summary['bm25']['mean_mrr']}  "
          f"Avg latency: {summary['bm25']['mean_latency_ms']}ms")

    # Save results
    with open("experiment_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nResults saved to experiment_results.json")

    return summary


if __name__ == "__main__":
    run_experiment()