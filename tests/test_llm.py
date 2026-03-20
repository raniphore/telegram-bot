"""
Tests for rag/llm.py — generate_answer and generate_summary.
OpenAI client is always mocked; no real API calls.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_openai_response(text: str):
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture()
def mock_openai(monkeypatch):
    """Patch AsyncOpenAI so every test gets a fresh mock client."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock()
    with patch("rag.llm.AsyncOpenAI", return_value=mock_client):
        yield mock_client


# ── generate_answer ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_answer_basic(mock_openai):
    from rag.llm import generate_answer
    mock_openai.chat.completions.create.return_value = _make_openai_response("  The answer.  ")

    result = await generate_answer("what time?", "context text")

    assert result == "The answer."
    mock_openai.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_generate_answer_includes_context_in_message(mock_openai):
    from rag.llm import generate_answer
    mock_openai.chat.completions.create.return_value = _make_openai_response("ok")

    await generate_answer("my question", "my context")

    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    user_message = call_kwargs["messages"][-1]["content"]
    assert "my context" in user_message
    assert "my question" in user_message


@pytest.mark.asyncio
async def test_generate_answer_prepends_history(mock_openai):
    from rag.llm import generate_answer
    mock_openai.chat.completions.create.return_value = _make_openai_response("ok")

    history = [
        {"role": "query", "content": "prev question", "response": "prev answer"},
    ]
    await generate_answer("new question", "ctx", history=history)

    messages = mock_openai.chat.completions.create.call_args.kwargs["messages"]
    # system + 1 user history + 1 assistant history + 1 final user = 4
    assert len(messages) == 4
    assert messages[1] == {"role": "user", "content": "prev question"}
    assert messages[2] == {"role": "assistant", "content": "prev answer"}


@pytest.mark.asyncio
async def test_generate_answer_no_history(mock_openai):
    from rag.llm import generate_answer
    mock_openai.chat.completions.create.return_value = _make_openai_response("ok")

    await generate_answer("q", "ctx", history=None)

    messages = mock_openai.chat.completions.create.call_args.kwargs["messages"]
    # system + final user only = 2
    assert len(messages) == 2


@pytest.mark.asyncio
async def test_generate_answer_empty_history(mock_openai):
    from rag.llm import generate_answer
    mock_openai.chat.completions.create.return_value = _make_openai_response("ok")

    await generate_answer("q", "ctx", history=[])

    messages = mock_openai.chat.completions.create.call_args.kwargs["messages"]
    assert len(messages) == 2  # same as no history


# ── generate_summary ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_summary_returns_stripped_text(mock_openai):
    from rag.llm import generate_summary
    mock_openai.chat.completions.create.return_value = _make_openai_response("  Summary here.  ")

    result = await generate_summary("some history text")

    assert result == "Summary here."


@pytest.mark.asyncio
async def test_generate_summary_passes_history_as_user_message(mock_openai):
    from rag.llm import generate_summary
    mock_openai.chat.completions.create.return_value = _make_openai_response("ok")

    await generate_summary("USER asked X\nBOT replied Y")

    messages = mock_openai.chat.completions.create.call_args.kwargs["messages"]
    user_msg = next(m for m in messages if m["role"] == "user")
    assert "USER asked X" in user_msg["content"]
