"""
Ingests knowledge base documents:
  - Reads Markdown/text files from knowledge_base/
  - Splits into chunks
  - Embeds using sentence-transformers
  - Stores embeddings as BLOBs in plain SQLite
"""
import os
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge_base")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "vectors.db")
CHUNK_SIZE = 300  # characters
CHUNK_OVERLAP = 50
MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents(directory: str) -> list[dict]:
    docs = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith((".md", ".txt")):
            path = os.path.join(directory, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            docs.append({"source": filename, "content": content})
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            source  TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
        """
    )
    conn.commit()


def ingest() -> None:
    print("Loading documents...")
    docs = load_documents(KNOWLEDGE_BASE_DIR)
    if not docs:
        print("No documents found in knowledge_base/")
        return

    model = SentenceTransformer(MODEL_NAME)
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    # Clear existing data before re-ingesting
    conn.execute("DELETE FROM chunks")

    for doc in docs:
        chunks = chunk_text(doc["content"])
        for chunk in chunks:
            embedding = model.encode(chunk).astype(np.float32)
            conn.execute(
                "INSERT INTO chunks (source, content, embedding) VALUES (?, ?, ?)",
                (doc["source"], chunk, embedding.tobytes()),
            )
        print(f"  Ingested {len(chunks)} chunks from {doc['source']}")

    conn.commit()
    conn.close()
    print("Ingestion complete.")


if __name__ == "__main__":
    ingest()
