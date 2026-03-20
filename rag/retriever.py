"""
Retrieves relevant chunks from SQLite and generates answers via OpenAI.
Uses numpy cosine similarity — no SQLite extensions needed.
"""
import hashlib
import os
import re
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.llm import generate_answer
from bot.history import get_cache, get_history, set_cache

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "vectors.db")
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    model = get_model()
    query_vec = model.encode(query).astype(np.float32)

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, source, content, embedding FROM chunks").fetchall()
    conn.close()

    if not rows:
        return []

    scored = []
    for row_id, source, content, emb_blob in rows:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        score = cosine_similarity(query_vec, emb)
        scored.append({"id": row_id, "source": source, "content": content, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _clean_excerpt(content: str, max_len: int = 120) -> str:
    """Return a clean, readable excerpt from a raw chunk."""
    # Strip markdown headings
    text = re.sub(r"#+\s+[^\n]+\n?", "", content)
    # Flatten whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # If the chunk was cut mid-word at the start, skip to the next word boundary
    if text and text[0].islower():
        space = text.find(" ")
        text = text[space + 1:].lstrip() if space != -1 else text
    return text[:max_len]


async def answer_query(query: str, user_id: int | None = None) -> str:
    query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()

    cached = get_cache(query_hash)
    if cached:
        return cached + "\n\n_(cached result)_"

    chunks = retrieve(query)
    if not chunks:
        return "I couldn't find any relevant information in the knowledge base."

    context = "\n\n".join(
        f"[{c['source']}]\n{c['content']}" for c in chunks
    )

    history = get_history(user_id) if user_id is not None else None
    answer = await generate_answer(query, context, history=history)

    # Build source snippets
    snippet_lines = []
    for c in chunks:
        excerpt = _clean_excerpt(c["content"])
        snippet_lines.append(f"• [{c['source']}]: {excerpt}...")
    snippets = "\n".join(snippet_lines)

    result = f"{answer}\n\nSources:\n{snippets}"
    set_cache(query_hash, result)
    return result
