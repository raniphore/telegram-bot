"""
Tests for rag/retriever.py — excerpt cleaning, caching, and answer_query flow.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag.retriever import _clean_excerpt, answer_query


# ── _clean_excerpt ────────────────────────────────────────────────────────────

def test_clean_excerpt_strips_markdown_headings():
    content = "# Company Policy\n\n## Working Hours\nStandard working hours are 9 to 5."
    result = _clean_excerpt(content)
    assert "#" not in result
    assert result.startswith("Standard working hours")


def test_clean_excerpt_skips_leading_partial_word():
    content = "te work requires a stable internet connection and availability."
    result = _clean_excerpt(content)
    assert result.startswith("work requires")


def test_clean_excerpt_preserves_normal_start():
    content = "Remote work requires a stable internet connection."
    result = _clean_excerpt(content)
    assert result.startswith("Remote work")


def test_clean_excerpt_respects_max_len():
    content = "A" * 200
    assert len(_clean_excerpt(content, max_len=50)) <= 50


def test_clean_excerpt_collapses_whitespace():
    content = "## Header\n\nSome    text   here."
    result = _clean_excerpt(content)
    assert "  " not in result


def test_clean_excerpt_empty_string():
    assert _clean_excerpt("") == ""


def test_clean_excerpt_only_heading():
    # Content that is just a heading — should return empty after strip
    result = _clean_excerpt("## Section Title\n")
    assert result == ""


# ── answer_query ──────────────────────────────────────────────────────────────

FAKE_CHUNKS = [
    {"id": 1, "source": "policy.md", "content": "Standard working hours are 9 to 5.", "score": 0.9},
    {"id": 2, "source": "faq.md", "content": "Flexible hours available on request.", "score": 0.7},
]


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch, tmp_path):
    import bot.history as history_mod
    monkeypatch.setattr(history_mod, "DB_PATH", str(tmp_path / "test.db"))


@pytest.mark.asyncio
async def test_answer_query_returns_answer_and_sources():
    with patch("rag.retriever.retrieve", return_value=FAKE_CHUNKS), \
         patch("rag.retriever.generate_answer", new=AsyncMock(return_value="Working hours are 9-5.")), \
         patch("rag.retriever.get_cache", return_value=None), \
         patch("rag.retriever.set_cache") as mock_set, \
         patch("rag.retriever.get_history", return_value=[]):

        result = await answer_query("what are the hours?", user_id=1)

    assert "Working hours are 9-5." in result
    assert "policy.md" in result
    assert "faq.md" in result
    assert "Sources:" in result
    mock_set.assert_called_once()


@pytest.mark.asyncio
async def test_answer_query_uses_cache_on_hit():
    with patch("rag.retriever.get_cache", return_value="Cached answer.\n\nSources:\n• [x.md]: ...") as mock_get, \
         patch("rag.retriever.retrieve") as mock_retrieve:

        result = await answer_query("what are the hours?")

    mock_retrieve.assert_not_called()
    assert "cached result" in result


@pytest.mark.asyncio
async def test_answer_query_no_chunks():
    with patch("rag.retriever.retrieve", return_value=[]), \
         patch("rag.retriever.get_cache", return_value=None):

        result = await answer_query("unknown question")

    assert "couldn't find" in result.lower()


@pytest.mark.asyncio
async def test_answer_query_passes_history_to_llm():
    fake_history = [{"role": "query", "content": "previous q", "response": "previous a"}]
    with patch("rag.retriever.retrieve", return_value=FAKE_CHUNKS), \
         patch("rag.retriever.generate_answer", new=AsyncMock(return_value="Answer.")) as mock_gen, \
         patch("rag.retriever.get_cache", return_value=None), \
         patch("rag.retriever.set_cache"), \
         patch("rag.retriever.get_history", return_value=fake_history):

        await answer_query("follow-up question", user_id=99)

    _, kwargs = mock_gen.call_args
    assert kwargs.get("history") == fake_history


@pytest.mark.asyncio
async def test_answer_query_md5_hash_deterministic():
    """Same query text (case/whitespace normalised) should hit same cache key."""
    captured_hashes = []

    def fake_set_cache(h, result):
        captured_hashes.append(h)

    with patch("rag.retriever.retrieve", return_value=FAKE_CHUNKS), \
         patch("rag.retriever.generate_answer", new=AsyncMock(return_value="Ans.")), \
         patch("rag.retriever.get_cache", return_value=None), \
         patch("rag.retriever.set_cache", side_effect=fake_set_cache), \
         patch("rag.retriever.get_history", return_value=[]):

        await answer_query("Working Hours?")
        await answer_query("working hours?")  # different casing

    assert captured_hashes[0] == captured_hashes[1]
