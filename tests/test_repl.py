import pytest

from agents import Agent, run_demo_loop

from .fake_model import FakeModel
from .test_responses import get_text_input_item, get_text_message


@pytest.mark.asyncio
async def test_run_demo_loop_conversation(monkeypatch, capsys):
    model = FakeModel()
    model.add_multiple_turn_outputs([[get_text_message("hello")], [get_text_message("good")]])

    agent = Agent(name="test", model=model)

    inputs = iter(["Hi", "How are you?", "quit"])
    monkeypatch.setattr("builtins.input", lambda _=" > ": next(inputs))

    await run_demo_loop(agent, stream=False)

    output = capsys.readouterr().out
    assert "hello" in output
    assert "good" in output
    assert model.last_turn_args["input"] == [
        get_text_input_item("Hi"),
        get_text_message("hello").model_dump(exclude_unset=True),
        get_text_input_item("How are you?"),
    ]


@pytest.mark.asyncio
async def test_run_demo_loop_exits_on_eof(monkeypatch, capsys):
    """EOFError during input() should terminate the loop gracefully."""
    model = FakeModel()
    agent = Agent(name="test", model=model)

    def raise_eof(_prompt: str = " > ") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)

    await run_demo_loop(agent, stream=False)

    # No model calls should have been made.
    assert model.first_turn_args is None


@pytest.mark.asyncio
async def test_run_demo_loop_exits_on_keyboard_interrupt(monkeypatch, capsys):
    """KeyboardInterrupt during input() should terminate the loop gracefully."""
    model = FakeModel()
    agent = Agent(name="test", model=model)

    def raise_interrupt(_prompt: str = " > ") -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", raise_interrupt)

    await run_demo_loop(agent, stream=False)

    assert model.first_turn_args is None


@pytest.mark.asyncio
async def test_run_demo_loop_skips_empty_input(monkeypatch, capsys):
    """Empty string input should be ignored and not sent to the model."""
    model = FakeModel()
    model.add_multiple_turn_outputs([[get_text_message("response")]])
    agent = Agent(name="test", model=model)

    # First input is empty (should be skipped), second is real, third exits.
    inputs = iter(["", "Hello", "exit"])
    monkeypatch.setattr("builtins.input", lambda _=" > ": next(inputs))

    await run_demo_loop(agent, stream=False)

    output = capsys.readouterr().out
    assert "response" in output
    # The model should have been called exactly once (for "Hello").
    assert model.last_turn_args["input"] == [get_text_input_item("Hello")]


@pytest.mark.asyncio
async def test_run_demo_loop_streaming_prints_text_deltas(monkeypatch, capsys):
    """In streaming mode, text delta content should be printed."""
    model = FakeModel()
    model.add_multiple_turn_outputs([[get_text_message("streamed response")]])
    agent = Agent(name="test", model=model)

    inputs = iter(["Say something", "exit"])
    monkeypatch.setattr("builtins.input", lambda _=" > ": next(inputs))

    await run_demo_loop(agent, stream=True)

    output = capsys.readouterr().out
    assert "streamed response" in output


@pytest.mark.asyncio
async def test_run_demo_loop_streaming_prints_tool_call_notification(monkeypatch, capsys):
    """In streaming mode, tool call items should print a notification."""
    from agents import function_tool

    @function_tool
    def my_tool() -> str:
        return "tool result"

    model = FakeModel()
    from .test_responses import get_function_tool_call

    # Return a tool call then a follow-up message to complete the turn.
    model.add_multiple_turn_outputs(
        [
            [
                get_function_tool_call("my_tool", "{}", "call_1"),
            ],
            [get_text_message("done")],
        ]
    )
    agent = Agent(name="test", model=model, tools=[my_tool])

    inputs = iter(["Use a tool", "exit"])
    monkeypatch.setattr("builtins.input", lambda _=" > ": next(inputs))

    await run_demo_loop(agent, stream=True)

    output = capsys.readouterr().out
    assert "[tool called]" in output


@pytest.mark.asyncio
async def test_run_demo_loop_streaming_exit_keyword_stops_loop(monkeypatch, capsys):
    """Typing 'exit' in streaming mode should stop the loop."""
    model = FakeModel()
    agent = Agent(name="test", model=model)

    inputs = iter(["exit"])
    monkeypatch.setattr("builtins.input", lambda _=" > ": next(inputs))

    await run_demo_loop(agent, stream=True)

    assert model.first_turn_args is None
