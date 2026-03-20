"""
Tests for bot/history.py — user history and query cache.
Uses a temporary SQLite DB so it never touches data/bot_state.db.
"""
import os
import tempfile
import pytest

# Patch the DB path before importing the module
import bot.history as history_mod


@pytest.fixture(autouse=True)
def tmp_db(monkeypatch, tmp_path):
    db_file = str(tmp_path / "test_state.db")
    monkeypatch.setattr(history_mod, "DB_PATH", db_file)
    yield db_file


# ── history ──────────────────────────────────────────────────────────────────

def test_get_history_empty():
    assert history_mod.get_history(user_id=1) == []


def test_save_and_get_history_single():
    history_mod.save_interaction(1, "query", "what are the hours?", "9-5 Mon-Fri")
    result = history_mod.get_history(user_id=1)
    assert len(result) == 1
    assert result[0]["role"] == "query"
    assert result[0]["content"] == "what are the hours?"
    assert result[0]["response"] == "9-5 Mon-Fri"


def test_get_history_returns_chronological_order():
    for i in range(5):
        history_mod.save_interaction(1, "query", f"q{i}", f"r{i}")
    result = history_mod.get_history(user_id=1, limit=3)
    # Most recent 3, returned oldest-first
    assert [r["content"] for r in result] == ["q2", "q3", "q4"]


def test_get_history_isolated_per_user():
    history_mod.save_interaction(1, "query", "user1 question", "ans1")
    history_mod.save_interaction(2, "query", "user2 question", "ans2")
    assert len(history_mod.get_history(user_id=1)) == 1
    assert len(history_mod.get_history(user_id=2)) == 1
    assert history_mod.get_history(user_id=1)[0]["content"] == "user1 question"


def test_save_image_interaction():
    history_mod.save_interaction(42, "image", "photo", "A cat sitting on a table")
    result = history_mod.get_history(user_id=42)
    assert result[0]["role"] == "image"
    assert result[0]["content"] == "photo"


# ── cache ─────────────────────────────────────────────────────────────────────

def test_cache_miss():
    assert history_mod.get_cache("nonexistent_hash") is None


def test_cache_set_and_get():
    history_mod.set_cache("abc123", "cached answer")
    assert history_mod.get_cache("abc123") == "cached answer"


def test_cache_overwrite():
    history_mod.set_cache("h1", "first")
    history_mod.set_cache("h1", "updated")
    assert history_mod.get_cache("h1") == "updated"


def test_cache_keys_are_independent():
    history_mod.set_cache("k1", "val1")
    history_mod.set_cache("k2", "val2")
    assert history_mod.get_cache("k1") == "val1"
    assert history_mod.get_cache("k2") == "val2"
