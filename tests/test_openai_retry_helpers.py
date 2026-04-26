"""
Tests for the internal helpers inside models/_openai_retry.py.
These cover the helper functions that extract retry metadata from exceptions.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from agents.models._openai_retry import (
    _get_error_code,  # noqa: PLC2701
    _get_header_value,  # noqa: PLC2701
    _get_status_code,  # noqa: PLC2701
    _header_lookup,  # noqa: PLC2701
    _parse_retry_after,  # noqa: PLC2701
    _parse_retry_after_ms,  # noqa: PLC2701
    get_openai_retry_advice,
)
from agents.retry import ModelRetryAdviceRequest

# ---------------------------------------------------------------------------
# _header_lookup
# ---------------------------------------------------------------------------


def test_header_lookup_httpx_headers_present() -> None:
    headers = httpx.Headers({"retry-after": "10"})
    assert _header_lookup(headers, "retry-after") == "10"


def test_header_lookup_httpx_headers_missing() -> None:
    headers = httpx.Headers({"x-other": "value"})
    assert _header_lookup(headers, "retry-after") is None


def test_header_lookup_mapping_case_insensitive() -> None:
    """Mapping headers should be matched case-insensitively."""
    headers: dict[str, str] = {"Retry-After": "5"}
    assert _header_lookup(headers, "retry-after") == "5"


def test_header_lookup_mapping_missing() -> None:
    headers: dict[str, str] = {"x-foo": "bar"}
    assert _header_lookup(headers, "retry-after") is None


def test_header_lookup_none_returns_none() -> None:
    assert _header_lookup(None, "retry-after") is None


# ---------------------------------------------------------------------------
# _get_header_value – walks the error chain and reads response_headers attr
# ---------------------------------------------------------------------------


def test_get_header_value_reads_response_headers_attribute() -> None:
    """_get_header_value should fall back to a 'response_headers' attribute."""

    class FakeError(Exception):
        response_headers = {"retry-after": "3"}

    error = FakeError("oops")
    assert _get_header_value(error, "retry-after") == "3"


def test_get_header_value_reads_headers_attribute() -> None:
    """_get_header_value should read a 'headers' attribute on the exception."""

    class FakeError(Exception):
        headers = {"retry-after": "7"}

    error = FakeError("oops")
    assert _get_header_value(error, "retry-after") == "7"


def test_get_header_value_returns_none_when_no_header_found() -> None:
    class FakeError(Exception):
        pass

    assert _get_header_value(FakeError("no header here"), "retry-after") is None


# ---------------------------------------------------------------------------
# _parse_retry_after_ms
# ---------------------------------------------------------------------------


def test_parse_retry_after_ms_none_returns_none() -> None:
    assert _parse_retry_after_ms(None) is None


def test_parse_retry_after_ms_valid() -> None:
    result = _parse_retry_after_ms("2000")
    assert result == pytest.approx(2.0)


def test_parse_retry_after_ms_invalid_string_returns_none() -> None:
    assert _parse_retry_after_ms("not-a-number") is None


def test_parse_retry_after_ms_negative_returns_none() -> None:
    assert _parse_retry_after_ms("-500") is None


# ---------------------------------------------------------------------------
# _parse_retry_after
# ---------------------------------------------------------------------------


def test_parse_retry_after_none_returns_none() -> None:
    assert _parse_retry_after(None) is None


def test_parse_retry_after_numeric_string() -> None:
    assert _parse_retry_after("30") == pytest.approx(30.0)


def test_parse_retry_after_negative_numeric_returns_none() -> None:
    assert _parse_retry_after("-1") is None


def test_parse_retry_after_http_date_future() -> None:
    """An RFC 2822 date in the future should return a positive delay."""
    future = time.time() + 3600
    import email.utils

    http_date = email.utils.formatdate(future, usegmt=True)
    result = _parse_retry_after(http_date)
    assert result is not None
    assert result > 0


def test_parse_retry_after_invalid_string_returns_none() -> None:
    assert _parse_retry_after("totally-invalid-date") is None


# ---------------------------------------------------------------------------
# _get_status_code – fallback attributes
# ---------------------------------------------------------------------------


def test_get_status_code_from_attribute() -> None:
    """Should return status_code from a non-APIStatusError with that attribute."""

    class FakeError(Exception):
        status_code = 503

    assert _get_status_code(FakeError("boom")) == 503


def test_get_status_code_from_status_attribute() -> None:
    """Should fall back to a 'status' attribute if 'status_code' is absent."""

    class FakeError(Exception):
        status = 429

    assert _get_status_code(FakeError("rate-limited")) == 429


def test_get_status_code_returns_none_when_absent() -> None:
    class FakeError(Exception):
        pass

    assert _get_status_code(FakeError("no code")) is None


# ---------------------------------------------------------------------------
# _get_error_code – body.code fallback
# ---------------------------------------------------------------------------


def test_get_error_code_from_flat_body_code() -> None:
    """Should read body.get('code') when body has no nested 'error' key."""

    class FakeError(Exception):
        body = {"code": "rate_limit_exceeded"}

    assert _get_error_code(FakeError("err")) == "rate_limit_exceeded"


def test_get_error_code_from_nested_body_error() -> None:
    """Should read body['error']['code'] when nested."""

    class FakeError(Exception):
        body = {"error": {"code": "quota_exceeded"}}

    assert _get_error_code(FakeError("err")) == "quota_exceeded"


def test_get_error_code_returns_none_when_absent() -> None:
    class FakeError(Exception):
        pass

    assert _get_error_code(FakeError("no code")) is None


# ---------------------------------------------------------------------------
# get_openai_retry_advice – specific branches
# ---------------------------------------------------------------------------


def _make_advice_request(error: Exception, **kwargs: Any) -> ModelRetryAdviceRequest:
    return ModelRetryAdviceRequest(
        error=error,
        attempt=1,
        stream=False,
        previous_response_id=kwargs.get("previous_response_id"),
        conversation_id=kwargs.get("conversation_id"),
    )


def test_get_openai_retry_advice_unsafe_to_replay_flag() -> None:
    """Errors marked unsafe_to_replay must return suggested=False."""

    class UnsafeError(Exception):
        unsafe_to_replay = True

    request = _make_advice_request(UnsafeError("unsafe"))
    advice = get_openai_retry_advice(request)
    assert advice is not None
    assert advice.suggested is False
    assert advice.replay_safety == "unsafe"


def test_get_openai_retry_advice_websocket_safety_message() -> None:
    """Errors with the WebSocket not-retry message must return suggested=False."""
    error = Exception(
        "the request may have been accepted, so the sdk will not automatically "
        "retry this websocket request."
    )
    request = _make_advice_request(error)
    advice = get_openai_retry_advice(request)
    assert advice is not None
    assert advice.suggested is False
    assert advice.replay_safety == "unsafe"


def test_get_openai_retry_advice_x_should_retry_false_header() -> None:
    """x-should-retry: false header must suppress retry."""
    request_obj = httpx.Request("POST", "https://api.openai.com")
    response = httpx.Response(
        500,
        request=request_obj,
        headers={"x-should-retry": "false"},
        json={"error": {"code": "server_error", "message": "err"}},
    )
    error = APIStatusError("err", response=response, body={})
    request = _make_advice_request(error)
    advice = get_openai_retry_advice(request)
    assert advice is not None
    assert advice.suggested is False


def test_get_openai_retry_advice_x_should_retry_true_header() -> None:
    """x-should-retry: true header must force retry."""
    request_obj = httpx.Request("POST", "https://api.openai.com")
    response = httpx.Response(
        200,
        request=request_obj,
        headers={"x-should-retry": "true"},
        json={},
    )
    error = APIStatusError("err", response=response, body={})
    request = _make_advice_request(error)
    advice = get_openai_retry_advice(request)
    assert advice is not None
    assert advice.suggested is True


def test_get_openai_retry_advice_retry_after_only() -> None:
    """When only retry-after is set but no retriable status, advice should still be produced."""
    request_obj = httpx.Request("POST", "https://api.openai.com")
    response = httpx.Response(
        200,
        request=request_obj,
        headers={"retry-after": "5"},
        json={},
    )
    error = APIStatusError("err", response=response, body={})
    request = _make_advice_request(error)
    advice = get_openai_retry_advice(request)
    # retry_after should be parsed; advice may or may not be suggested but must not be None
    assert advice is not None
    assert advice.retry_after is not None


def test_get_openai_retry_advice_network_error_is_retryable() -> None:
    """APIConnectionError should always be suggested for retry."""
    error = APIConnectionError(
        message="connection dropped",
        request=httpx.Request("POST", "https://api.openai.com"),
    )
    request = _make_advice_request(error)
    advice = get_openai_retry_advice(request)
    assert advice is not None
    assert advice.suggested is True


def test_get_openai_retry_advice_timeout_error_is_retryable() -> None:
    """APITimeoutError should always be suggested for retry."""
    error = APITimeoutError(
        request=httpx.Request("POST", "https://api.openai.com"),
    )
    request = _make_advice_request(error)
    advice = get_openai_retry_advice(request)
    assert advice is not None
    assert advice.suggested is True


def test_get_openai_retry_advice_non_retriable_status_returns_none() -> None:
    """A plain 4xx status with no special headers should return None (no advice)."""
    request_obj = httpx.Request("POST", "https://api.openai.com")
    response = httpx.Response(400, request=request_obj, json={"error": {}})
    error = APIStatusError("bad request", response=response, body={})
    request = _make_advice_request(error)
    advice = get_openai_retry_advice(request)
    assert advice is None


def test_get_openai_retry_advice_stateful_request_sets_replay_safety() -> None:
    """Retriable 429 with a stateful previous_response_id should carry replay_safety='safe'."""
    request_obj = httpx.Request("POST", "https://api.openai.com")
    response = httpx.Response(
        429,
        request=request_obj,
        json={"error": {"code": "rate_limit", "message": "rate limited"}},
    )
    error = APIStatusError("rate limited", response=response, body={})
    advice_request = _make_advice_request(error, previous_response_id="resp_prev")
    advice = get_openai_retry_advice(advice_request)
    assert advice is not None
    assert advice.suggested is True
    assert advice.replay_safety == "safe"
