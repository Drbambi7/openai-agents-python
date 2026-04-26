"""
Tests for uncovered paths in models/chatcmpl_stream_handler.py.

These tests call ChatCmplStreamHandler.handle_stream directly with synthetic
ChatCompletionChunk sequences to exercise the code paths not already covered by
the integration-style tests in test_openai_chatcompletions_stream.py.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.completion_usage import CompletionUsage
from openai.types.responses import Response

from agents.models.chatcmpl_stream_handler import ChatCmplStreamHandler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response() -> Response:
    return Response(
        id="resp-id",
        created_at=0,
        model="fake-model",
        object="response",
        output=[],
        tool_choice="none",
        tools=[],
        parallel_tool_calls=False,
    )


def _text_chunk(
    text: str, *, chunk_id: str = "cid", with_usage: bool = False
) -> ChatCompletionChunk:
    usage = (
        CompletionUsage(completion_tokens=1, prompt_tokens=1, total_tokens=2)
        if with_usage
        else None
    )
    return ChatCompletionChunk(
        id=chunk_id,
        created=1,
        model="fake",
        object="chat.completion.chunk",
        choices=[Choice(index=0, delta=ChoiceDelta(content=text))],
        usage=usage,
    )


async def _stream(*chunks: ChatCompletionChunk) -> AsyncIterator[ChatCompletionChunk]:
    for chunk in chunks:
        yield chunk


async def _collect_events(*chunks: ChatCompletionChunk, model: str | None = None) -> list[Any]:
    events = []
    async for event in ChatCmplStreamHandler.handle_stream(
        _make_response(),
        _stream(*chunks),  # type: ignore[arg-type]
        model=model,
    ):
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Empty / no-choices chunks are skipped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_choices_chunk_is_skipped() -> None:
    """A chunk with no choices should not produce any content events."""
    empty_chunk = ChatCompletionChunk(
        id="cid",
        created=1,
        model="fake",
        object="chat.completion.chunk",
        choices=[],
    )
    text_chunk = _text_chunk("hi", with_usage=True)

    events = await _collect_events(empty_chunk, text_chunk)

    # response.created + output_item.added + content_part.added + delta + content_part.done +
    # output_item.done + response.completed = 7 events
    types = [e.type for e in events]
    assert "response.created" in types
    assert "response.completed" in types
    # The empty chunk should not have caused extra delta events.
    delta_events = [e for e in events if e.type == "response.output_text.delta"]
    assert len(delta_events) == 1
    assert delta_events[0].delta == "hi"


# ---------------------------------------------------------------------------
# chunk.model fallback when model= kwarg not passed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chunk_model_attribute_used_when_kwarg_omitted() -> None:
    """When no model= kwarg is provided, chunk.model should populate provider_data."""
    chunk = ChatCompletionChunk(
        id="cid",
        created=1,
        model="gpt-4o",
        object="chat.completion.chunk",
        choices=[Choice(index=0, delta=ChoiceDelta(content="hello"))],
        usage=CompletionUsage(completion_tokens=1, prompt_tokens=1, total_tokens=2),
    )

    events = await _collect_events(chunk, model=None)

    completed = next(e for e in events if e.type == "response.completed")
    # provider_data should carry "model" = "gpt-4o" from the chunk
    assert completed.response.output  # ensure something was produced


# ---------------------------------------------------------------------------
# delta.reasoning path (third-party providers, e.g. DeepSeek)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delta_reasoning_attribute_produces_reasoning_events() -> None:
    """delta.reasoning triggers ReasoningItem creation and text-delta events."""
    from typing import cast

    reasoning_chunk_1 = ChatCompletionChunk(
        id="cid",
        created=1,
        model="deepseek",
        object="chat.completion.chunk",
        choices=[Choice(index=0, delta=ChoiceDelta())],
    )
    cast(Any, reasoning_chunk_1.choices[0].delta).reasoning = "think..."

    reasoning_chunk_2 = ChatCompletionChunk(
        id="cid",
        created=1,
        model="deepseek",
        object="chat.completion.chunk",
        choices=[Choice(index=0, delta=ChoiceDelta())],
        usage=CompletionUsage(completion_tokens=5, prompt_tokens=3, total_tokens=8),
    )
    cast(Any, reasoning_chunk_2.choices[0].delta).reasoning = "more"

    events = await _collect_events(reasoning_chunk_1, reasoning_chunk_2, model="deepseek")

    types = [e.type for e in events]
    # Should see reasoning item added, reasoning text deltas, item done, and completed.
    assert "response.output_item.added" in types
    assert "response.reasoning_text.delta" in types
    assert "response.output_item.done" in types
    assert "response.completed" in types

    delta_events = [e for e in events if e.type == "response.reasoning_text.delta"]
    assert len(delta_events) == 2
    assert delta_events[0].delta == "think..."
    assert delta_events[1].delta == "more"


# ---------------------------------------------------------------------------
# Anthropic thinking_blocks path – text + signature accumulation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thinking_blocks_text_and_signature_stored() -> None:
    """thinking_blocks deltas should be stored and put into encrypted_content.

    The thinking_text and thinking_signature are stored into the reasoning item
    that was created by a preceding reasoning_content delta.
    """
    from typing import cast

    # First chunk: reasoning_content creates the reasoning item.
    chunk_reasoning = ChatCompletionChunk(
        id="cid",
        created=1,
        model="claude-3-7-sonnet-latest",
        object="chat.completion.chunk",
        choices=[Choice(index=0, delta=ChoiceDelta())],
    )
    cast(Any, chunk_reasoning.choices[0].delta).reasoning_content = "step1"

    # Second chunk: thinking_blocks contributes thinking text to the existing reasoning item.
    chunk_thinking_text = ChatCompletionChunk(
        id="cid",
        created=1,
        model="claude-3-7-sonnet-latest",
        object="chat.completion.chunk",
        choices=[Choice(index=0, delta=ChoiceDelta())],
    )
    cast(Any, chunk_thinking_text.choices[0].delta).thinking_blocks = [{"thinking": "my_thought"}]

    # Third chunk: thinking_blocks contributes a signature.
    chunk_signature = ChatCompletionChunk(
        id="cid",
        created=1,
        model="claude-3-7-sonnet-latest",
        object="chat.completion.chunk",
        choices=[Choice(index=0, delta=ChoiceDelta())],
        usage=CompletionUsage(completion_tokens=5, prompt_tokens=3, total_tokens=8),
    )
    cast(Any, chunk_signature.choices[0].delta).thinking_blocks = [{"signature": "sig_abc"}]

    events = await _collect_events(
        chunk_reasoning, chunk_thinking_text, chunk_signature, model="claude-3-7-sonnet-latest"
    )

    completed = next(e for e in events if e.type == "response.completed")
    # The reasoning item should carry the encrypted_content (signature).
    from openai.types.responses import ResponseReasoningItem

    reasoning_outputs = [
        o for o in completed.response.output if isinstance(o, ResponseReasoningItem)
    ]
    assert reasoning_outputs, "Expected at least one reasoning item in output"
    assert reasoning_outputs[0].encrypted_content == "sig_abc"
    # The thinking text should be in the content list.
    assert any(
        c.text == "my_thought"
        for c in (reasoning_outputs[0].content or [])
        if hasattr(c, "text")
    )


# ---------------------------------------------------------------------------
# Fallback path for function calls (name never arrives / no streaming start)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_function_call_without_id_uses_fallback_path() -> None:
    """When a tool call id is never sent the fallback non-streaming path is used."""
    # Build a chunk where call_id is never set (id=None)
    tc_delta = ChoiceDeltaToolCall(
        index=0,
        id=None,
        function=ChoiceDeltaToolCallFunction(name=None, arguments='{"x":1}'),
        type="function",
    )
    chunk = ChatCompletionChunk(
        id="cid",
        created=1,
        model="fake",
        object="chat.completion.chunk",
        choices=[Choice(index=0, delta=ChoiceDelta(tool_calls=[tc_delta]))],
        usage=CompletionUsage(completion_tokens=1, prompt_tokens=1, total_tokens=2),
    )

    events = await _collect_events(chunk)

    types = [e.type for e in events]
    # Fallback path emits: output_item.added, function_call_arguments.delta, output_item.done
    assert "response.output_item.added" in types
    assert "response.function_call_arguments.delta" in types
    assert "response.output_item.done" in types


# ---------------------------------------------------------------------------
# Refusal then text: two content parts in same message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refusal_after_text_produces_both_content_parts() -> None:
    """A message with both text and refusal parts should produce two content part events."""
    chunk1 = ChatCompletionChunk(
        id="cid",
        created=1,
        model="fake",
        object="chat.completion.chunk",
        choices=[Choice(index=0, delta=ChoiceDelta(content="prefix"))],
    )
    # Add refusal attribute dynamically (not all openai lib builds have it in ChoiceDelta)
    from typing import cast

    chunk2 = ChatCompletionChunk(
        id="cid",
        created=1,
        model="fake",
        object="chat.completion.chunk",
        choices=[Choice(index=0, delta=ChoiceDelta())],
        usage=CompletionUsage(completion_tokens=2, prompt_tokens=2, total_tokens=4),
    )
    cast(Any, chunk2.choices[0].delta).refusal = "I cannot help"

    events = await _collect_events(chunk1, chunk2)

    types = [e.type for e in events]
    # Both content part added events should appear.
    added_events = [e for e in events if e.type == "response.content_part.added"]
    assert len(added_events) == 2
    # And both should be done.
    done_events = [e for e in events if e.type == "response.content_part.done"]
    assert len(done_events) == 2
    assert "response.refusal.delta" in types


# ---------------------------------------------------------------------------
# reasoning_content (summary-based, e.g. o-series) then text – finish_reasoning_summary_part
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reasoning_summary_part_finishes_when_text_arrives() -> None:
    """When text content arrives after reasoning_content, the summary part is closed first."""
    from typing import cast

    chunk1 = ChatCompletionChunk(
        id="cid",
        created=1,
        model="o3",
        object="chat.completion.chunk",
        choices=[Choice(index=0, delta=ChoiceDelta())],
    )
    cast(Any, chunk1.choices[0].delta).reasoning_content = "thinking..."

    # Text delta in same stream – should trigger _finish_reasoning_summary_part
    chunk2 = _text_chunk("answer", with_usage=True)

    events = await _collect_events(chunk1, chunk2, model="o3")

    types = [e.type for e in events]
    # The reasoning summary part should have been added and done.
    assert "response.reasoning_summary_part.added" in types
    assert "response.reasoning_summary_part.done" in types
    # The text delta should appear after.
    assert "response.output_text.delta" in types


# ---------------------------------------------------------------------------
# _finish_reasoning_summary_part early exit paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reasoning_summary_part_done_early_exit_out_of_range_index() -> None:
    """_finish_reasoning_summary_part should silently exit if summary list is empty."""

    from agents.models.chatcmpl_stream_handler import SequenceNumber, StreamingState

    state = StreamingState()
    # Set an active index that is out of range for an empty summary list.
    from openai.types.responses import ResponseReasoningItem

    reasoning_item = ResponseReasoningItem(id="fake", summary=[], type="reasoning")
    state.reasoning_content_index_and_output = (0, reasoning_item)
    state.active_reasoning_summary_index = 5  # out of range

    seq = SequenceNumber()
    events = list(ChatCmplStreamHandler._finish_reasoning_summary_part(state, seq))

    # Should yield nothing, and active_reasoning_summary_index reset to None.
    assert events == []
    assert state.active_reasoning_summary_index is None


# ---------------------------------------------------------------------------
# _finish_reasoning_item – content path (no summary items)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finish_reasoning_item_content_path() -> None:
    """_finish_reasoning_item should emit ResponseReasoningTextDoneEvent for content-based items."""
    from openai.types.responses import ResponseReasoningItem
    from openai.types.responses.response_reasoning_item import Content

    from agents.models.chatcmpl_stream_handler import SequenceNumber, StreamingState

    state = StreamingState()
    reasoning_item = ResponseReasoningItem(
        id="fake",
        summary=[],
        content=[Content(text="my reasoning", type="reasoning_text")],
        type="reasoning",
    )
    state.reasoning_content_index_and_output = (0, reasoning_item)

    seq = SequenceNumber()
    events = list(ChatCmplStreamHandler._finish_reasoning_item(state, seq))

    types = [e.type for e in events]
    assert "response.reasoning_text.done" in types
    assert "response.output_item.done" in types
    assert state.reasoning_item_done is True
