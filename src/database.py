import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ── ChromaDB client ────────────────────────────────────────────────
def get_chroma_client():
    return chromadb.PersistentClient(
        path=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    )

def get_or_create_collection(client, collection_name: str = "interview_questions"):
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

# ── Embedding function ─────────────────────────────────────────────
def embed_text(text: str) -> List[float]:
    """
    Convert one text into a dense vector using local SentenceTransformer.
    This is free and does not require OpenAI API quota.
    """
    embedding = embedding_model.encode(
        text,
        normalize_embeddings=True
    )
    return embedding.tolist()

def embed_batch(texts):
    """
    Embed question texts locally using SentenceTransformer.
    This is free and does not require OpenAI API quota.
    """
    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    )
    return embeddings.tolist()

# ── BM25 tokenizer ─────────────────────────────────────────────────
STOPWORDS = {
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for",
    "of", "with", "you", "your", "about", "how", "did", "tell", "me",
    "describe", "what", "when", "where", "was", "that", "this", "is",
    "are", "were", "have", "had", "been", "would", "could", "should",
    "time", "situation", "example"
}

def tokenize_for_bm25(text: str) -> List[str]:
    """
    Clean, lowercase, and remove stopwords.
    Returns list of meaningful tokens.
    """
    text = re.sub(r"[^a-z0-9\s]", "", text.lower())
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return tokens

# ── Index builder ──────────────────────────────────────────────────
def build_chroma_index(questions_path: str = "data/questions.json"):
    """
    Read questions.json, embed every question_text,
    and store in ChromaDB with full metadata.
    """
    print("Loading questions...")
    with open(questions_path, "r") as f:
        questions = json.load(f)

    chroma = get_chroma_client()
    collection = get_or_create_collection(chroma)

    # Check if already indexed
    existing = collection.count()
    if existing == len(questions):
        print(f"ChromaDB already has {existing} questions. Skipping.")
        return collection

    print(f"Embedding {len(questions)} questions...")
    # Embed in batch of 20 to avoid rate limits
    batch_size = 20
    all_embeddings = []

    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        texts = [q["question_text"] for q in batch]
        embeddings = embed_batch(texts)
        all_embeddings.extend(embeddings)
        print(f"  Embedded {min(i+batch_size, len(questions))}/{len(questions)}")

    # Prepare ChromaDB format
    ids        = [q["question_id"] for q in questions]
    documents  = [q["question_text"] for q in questions]
    metadatas  = []

    for q in questions:
        metadatas.append({
            "topic":        q["topic"],
            "difficulty":   q["difficulty"],
            "role":         ",".join(q["role"]),
            "company":      ",".join(q["company"]),
            "ideal_hints":  q["ideal_hints"],
            "follow_up":    q["follow_up"]
        })

    # Add to ChromaDB
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=all_embeddings,
        metadatas=metadatas
    )

    print(f"ChromaDB index built: {collection.count()} questions stored.")
    return collection


def build_bm25_index(questions_path: str = "data/questions.json"):
    """
    Build in-memory BM25 index from questions.json.
    Returns (bm25_model, questions_list) tuple.
    """
    with open(questions_path, "r") as f:
        questions = json.load(f)

    tokenized_corpus = [
        tokenize_for_bm25(q["question_text"])
        for q in questions
    ]

    bm25 = BM25Okapi(tokenized_corpus)
    print(f"BM25 index built: {len(questions)} questions indexed.")
    return bm25, questions


# ── Run this file directly to build both indexes ───────────────────
if __name__ == "__main__":
    print("=== Building ChromaDB Index ===")
    build_chroma_index()

    print("\n=== Building BM25 Index ===")
    bm25_model, questions = build_bm25_index()
    print("Both indexes ready.")