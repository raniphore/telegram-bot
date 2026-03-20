"""
Per-user interaction history and MD5 query cache using SQLite.
"""
import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bot_state.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   INTEGER NOT NULL,
            role      TEXT NOT NULL,
            content   TEXT NOT NULL,
            response  TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_cache (
            hash      TEXT PRIMARY KEY,
            result    TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def save_interaction(user_id: int, role: str, content: str, response: str) -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT INTO user_history (user_id, role, content, response, timestamp) VALUES (?, ?, ?, ?, ?)",
        (user_id, role, content, response, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_history(user_id: int, limit: int = 3) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT role, content, response FROM user_history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    # Return in chronological order
    return [{"role": r, "content": c, "response": resp} for r, c, resp in reversed(rows)]


def get_cache(query_hash: str) -> str | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT result FROM query_cache WHERE hash = ?", (query_hash,)
    ).fetchone()
    conn.close()
    return row[0] if row else None


def set_cache(query_hash: str, result: str) -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO query_cache (hash, result, timestamp) VALUES (?, ?, ?)",
        (query_hash, result, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
