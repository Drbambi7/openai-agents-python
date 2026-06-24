import threading
import time
from collections import defaultdict
from functools import wraps

from flask import jsonify, request

from config import config

_rate_store: dict[str, list[float]] = defaultdict(list)
_rate_lock = threading.Lock()
_request_counter = 0
_CLEANUP_EVERY = 500


def _cleanup(now: float) -> None:
    stale = [ip for ip, ts in _rate_store.items()
             if not any(now - t < config.RATE_LIMIT_WINDOW for t in ts)]
    for ip in stale:
        del _rate_store[ip]


def rate_limit(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        global _request_counter
        ip = request.remote_addr or "unknown"
        now = time.time()
        with _rate_lock:
            _request_counter += 1
            if _request_counter % _CLEANUP_EVERY == 0:
                _cleanup(now)
            calls = [t for t in _rate_store[ip] if now - t < config.RATE_LIMIT_WINDOW]
            if len(calls) >= config.RATE_LIMIT_CALLS:
                return jsonify({"error": "יותר מדי בקשות. נסה שוב בעוד דקה."}), 429
            calls.append(now)
            _rate_store[ip] = calls
        return fn(*args, **kwargs)
    return wrapper


def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    )
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Server"] = "server"
    if config.HTTPS_ENABLED:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


def extract_text() -> tuple[str, tuple | None]:
    """Validate request and extract text field. Returns (text, None) or ('', error_tuple)."""
    if not request.is_json:
        return "", (jsonify({"error": "Content-Type חייב להיות application/json."}), 415)
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return "", (jsonify({"error": "שדה 'text' נדרש."}), 400)
    if len(text) > config.MAX_TEXT_LEN:
        return "", (
            jsonify({"error": f"הטקסט ארוך מדי. מקסימום {config.MAX_TEXT_LEN} תווים."}),
            400,
        )
    return text, None
