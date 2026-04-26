"""
Tests for run_internal/approvals.py helper functions.
"""

from __future__ import annotations

from typing import Any

from openai.types.responses import ResponseFunctionToolCall

from agents import Agent
from agents.items import ToolCallOutputItem
from agents.run_internal.approvals import (
    _build_function_tool_call_for_approval_error,  # noqa: PLC2701
    append_approval_error_output,
    append_input_items_excluding_approvals,
    approvals_from_step,
    filter_tool_approvals,
)

from .utils.factories import make_tool_approval_item, make_tool_call

# ---------------------------------------------------------------------------
# filter_tool_approvals
# ---------------------------------------------------------------------------


def test_filter_tool_approvals_keeps_only_approval_items() -> None:
    agent = Agent(name="test")
    approval = make_tool_approval_item(agent, call_id="c1", name="tool_a")

    # Mix approval items with non-approval items (plain strings, dicts).
    mixed: list[Any] = [approval, "not_an_approval", {"type": "function_call_output"}]

    result = filter_tool_approvals(mixed)

    assert result == [approval]


def test_filter_tool_approvals_empty_list() -> None:
    assert filter_tool_approvals([]) == []


def test_filter_tool_approvals_no_approvals() -> None:
    result = filter_tool_approvals(["a", 42, None])
    assert result == []


# ---------------------------------------------------------------------------
# approvals_from_step
# ---------------------------------------------------------------------------


def test_approvals_from_step_returns_empty_when_no_interruptions() -> None:
    class StepWithoutInterruptions:
        pass

    result = approvals_from_step(StepWithoutInterruptions())
    assert result == []


def test_approvals_from_step_returns_empty_when_interruptions_none() -> None:
    class StepWithNoneInterruptions:
        interruptions = None

    result = approvals_from_step(StepWithNoneInterruptions())
    assert result == []


def test_approvals_from_step_filters_approvals_from_interruptions() -> None:
    agent = Agent(name="test")
    approval = make_tool_approval_item(agent, call_id="c2", name="tool_b")

    class StepWithInterruptions:
        interruptions = [approval, "something_else"]

    result = approvals_from_step(StepWithInterruptions())
    assert result == [approval]


# ---------------------------------------------------------------------------
# append_approval_error_output
# ---------------------------------------------------------------------------


def test_append_approval_error_output_with_response_function_tool_call() -> None:
    """When tool_call is already a ResponseFunctionToolCall it must be reused directly."""
    agent = Agent(name="test")
    tool_call = make_tool_call(call_id="c3", name="my_tool")
    generated: list[Any] = []

    append_approval_error_output(
        generated_items=generated,
        agent=agent,
        tool_call=tool_call,
        tool_name="my_tool",
        call_id="c3",
        message="rejected",
    )

    assert len(generated) == 1
    item = generated[0]
    assert isinstance(item, ToolCallOutputItem)
    assert item.output == "rejected"


def test_append_approval_error_output_with_dict_tool_call_no_namespace() -> None:
    agent = Agent(name="test")
    tool_call_dict: dict[str, Any] = {"type": "function_call", "name": "my_tool", "call_id": "c4"}
    generated: list[Any] = []

    append_approval_error_output(
        generated_items=generated,
        agent=agent,
        tool_call=tool_call_dict,
        tool_name="my_tool",
        call_id="c4",
        message="denied",
    )

    assert len(generated) == 1
    assert generated[0].output == "denied"


def test_append_approval_error_output_with_dict_tool_call_with_namespace() -> None:
    """Namespace should be propagated when the dict contains a namespace key."""
    agent = Agent(name="test")
    tool_call_dict: dict[str, Any] = {
        "type": "function_call",
        "name": "my_tool",
        "call_id": "c5",
        "namespace": "billing",
    }
    generated: list[Any] = []

    append_approval_error_output(
        generated_items=generated,
        agent=agent,
        tool_call=tool_call_dict,
        tool_name="my_tool",
        call_id="c5",
        message="denied",
    )

    assert len(generated) == 1
    assert generated[0].output == "denied"


def test_build_function_tool_call_for_approval_error_dict_with_namespace() -> None:
    """_build_function_tool_call_for_approval_error propagates namespace from dict."""
    result = _build_function_tool_call_for_approval_error(
        {"type": "function_call", "name": "my_tool", "call_id": "c5", "namespace": "billing"},
        "my_tool",
        "c5",
    )
    assert isinstance(result, ResponseFunctionToolCall)
    assert getattr(result, "namespace", None) == "billing"


def test_build_function_tool_call_for_approval_error_dict_without_namespace() -> None:
    """_build_function_tool_call_for_approval_error works when dict has no namespace."""
    result = _build_function_tool_call_for_approval_error(
        {"type": "function_call", "name": "my_tool", "call_id": "c5"},
        "my_tool",
        "c5",
    )
    assert isinstance(result, ResponseFunctionToolCall)
    assert getattr(result, "namespace", None) is None


def test_append_approval_error_output_with_object_tool_call_with_namespace() -> None:
    """Namespace attribute on an arbitrary object should be propagated."""
    agent = Agent(name="test")

    class FakeToolCall:
        namespace = "shipping"

    generated: list[Any] = []

    append_approval_error_output(
        generated_items=generated,
        agent=agent,
        tool_call=FakeToolCall(),
        tool_name="ship_order",
        call_id="c6",
        message="denied",
    )

    assert len(generated) == 1
    assert generated[0].output == "denied"


def test_build_function_tool_call_for_approval_error_object_with_namespace() -> None:
    """_build_function_tool_call_for_approval_error propagates namespace from object attr."""

    class FakeToolCall:
        namespace = "shipping"

    result = _build_function_tool_call_for_approval_error(FakeToolCall(), "ship_order", "c6")
    assert isinstance(result, ResponseFunctionToolCall)
    assert getattr(result, "namespace", None) == "shipping"


def test_build_function_tool_call_for_approval_error_object_without_namespace() -> None:
    """Objects without namespace attr should produce ResponseFunctionToolCall with no namespace."""

    class FakeToolCallNoNS:
        pass

    result = _build_function_tool_call_for_approval_error(FakeToolCallNoNS(), "order_action", "c7")
    assert isinstance(result, ResponseFunctionToolCall)
    assert getattr(result, "namespace", None) is None


def test_append_approval_error_output_none_call_id() -> None:
    """call_id=None should fall back to 'unknown' in the synthesized function tool call."""
    agent = Agent(name="test")

    class FakeToolCall:
        pass

    generated: list[Any] = []

    append_approval_error_output(
        generated_items=generated,
        agent=agent,
        tool_call=FakeToolCall(),
        tool_name="some_tool",
        call_id=None,
        message="timeout",
    )

    assert len(generated) == 1
    # raw_item is the output dict; call_id there should be "unknown"
    raw = generated[0].raw_item
    assert raw["call_id"] == "unknown"


# ---------------------------------------------------------------------------
# append_input_items_excluding_approvals
# ---------------------------------------------------------------------------


def test_append_input_items_excluding_approvals_skips_approval_items() -> None:
    """ToolApprovalItem instances must be silently skipped."""
    agent = Agent(name="test")
    approval = make_tool_approval_item(agent, call_id="c8", name="tool_x")

    base: list[Any] = []
    append_input_items_excluding_approvals(base, [approval])
    assert base == []


def test_append_input_items_excluding_approvals_includes_non_approval_items() -> None:
    """Non-approval RunItems should be converted and appended."""
    agent = Agent(name="test")
    from agents.items import MessageOutputItem

    from .test_responses import get_text_message

    msg_item = MessageOutputItem(
        agent=agent,
        raw_item=get_text_message("hello"),
    )

    base: list[Any] = []
    append_input_items_excluding_approvals(base, [msg_item])
    # The converted dict should have been appended.
    assert len(base) == 1


def test_append_input_items_excluding_approvals_mixes_items() -> None:
    """Mixed lists: approval items skipped, others appended."""
    agent = Agent(name="test")
    from agents.items import MessageOutputItem

    from .test_responses import get_text_message

    approval = make_tool_approval_item(agent, call_id="c9", name="tool_y")
    msg_item = MessageOutputItem(
        agent=agent,
        raw_item=get_text_message("world"),
    )

    base: list[Any] = []
    append_input_items_excluding_approvals(base, [approval, msg_item])
    assert len(base) == 1
