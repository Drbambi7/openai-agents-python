"""
Flask app with two AI engines for person profiling.

Access Engine  – extracts a structured profile from free text.
Mirror Engine  – detects emotions and psychological tone from free text.

Run:
    pip install flask openai
    export OPENAI_API_KEY=sk-...
    python app.py
"""

import json
import logging
import os
import threading
import time
from collections import defaultdict
from functools import wraps

from flask import Flask, jsonify, request
from openai import OpenAI

# ────────────────────────────────────────────────
# Startup validation – fail fast if key is missing
# ────────────────────────────────────────────────

_api_key = os.environ.get("OPENAI_API_KEY", "")
if not _api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Export it before starting the server:\n"
        "  export OPENAI_API_KEY=sk-..."
    )

# ────────────────────────────────────────────────
# App + logging
# ────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Block bodies larger than 64 KB – prevents memory exhaustion.
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

client = OpenAI(api_key=_api_key, timeout=30.0)

# ────────────────────────────────────────────────
# Security headers on every response
# ────────────────────────────────────────────────

@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = "default-src 'none'"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    # הסתר את גרסת Flask/Werkzeug/Python מהתוקף
    response.headers["Server"] = "server"
    # הפעל HSTS רק בפרודקשן עם HTTPS
    if os.environ.get("HTTPS_ENABLED", "false").lower() == "true":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# ────────────────────────────────────────────────
# Simple in-memory rate limiter
# max_calls per window_seconds per IP
# ────────────────────────────────────────────────

_rate_store: dict[str, list[float]] = defaultdict(list)
_rate_lock = threading.Lock()
MAX_CALLS = 10        # בקשות
WINDOW_SEC = 60       # לדקה
CLEANUP_EVERY = 500   # נקה זיכרון כל N בקשות
_request_counter = 0


def _cleanup_old_entries(now: float) -> None:
    """Remove IPs with no recent requests to prevent unbounded memory growth."""
    stale = [ip for ip, ts in _rate_store.items() if not any(now - t < WINDOW_SEC for t in ts)]
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
            if _request_counter % CLEANUP_EVERY == 0:
                _cleanup_old_entries(now)
            calls = [t for t in _rate_store[ip] if now - t < WINDOW_SEC]
            if len(calls) >= MAX_CALLS:
                return jsonify({"error": "יותר מדי בקשות. נסה שוב בעוד דקה."}), 429
            calls.append(now)
            _rate_store[ip] = calls
        return fn(*args, **kwargs)
    return wrapper


# ────────────────────────────────────────────────
# Input helpers
# ────────────────────────────────────────────────

MAX_TEXT_LEN = 2_000  # תווים


def _extract_text() -> tuple[str, tuple | None]:
    """Return (text, None) on success or ("", error_response) on failure."""
    if not request.is_json:
        return "", (jsonify({"error": "Content-Type חייב להיות application/json."}), 415)

    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()

    if not text:
        return "", (jsonify({"error": "שדה 'text' נדרש."}), 400)

    if len(text) > MAX_TEXT_LEN:
        return "", (
            jsonify({"error": f"הטקסט ארוך מדי. מקסימום {MAX_TEXT_LEN} תווים."}),
            400,
        )

    return text, None


# ────────────────────────────────────────────────
# Access Engine – ניתוח פרופיל אישי
# ────────────────────────────────────────────────

def analyze_person(input_text: str) -> dict:
    """Extract a structured personal profile from free-form text."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "אתה מנתח פרופילים. חלץ מהטקסט: שם, גיל, מקצוע, תחומי עניין, "
                    "ותכונות אישיות בולטות. החזר תמיד JSON תקני עם המפתחות: "
                    "name, age, profession, interests, traits."
                ),
            },
            {"role": "user", "content": input_text},
        ],
        response_format={"type": "json_object"},
        max_tokens=300,
    )
    return json.loads(response.choices[0].message.content)


# ────────────────────────────────────────────────
# Mirror Engine – ניתוח רגשות
# ────────────────────────────────────────────────

def analyze_emotions(input_text: str) -> dict:
    """Detect emotions and psychological tone from free-form text."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "אתה מנתח רגשות ומצב פסיכולוגי. זהה מהטקסט: רגש דומיננטי, "
                    "עוצמת הרגש (1-10), מצב רוח כללי, ורמת מתח. "
                    "החזר תמיד JSON תקני עם המפתחות: "
                    "dominant_emotion, intensity, mood, stress_level, summary."
                ),
            },
            {"role": "user", "content": input_text},
        ],
        response_format={"type": "json_object"},
        max_tokens=300,
    )
    return json.loads(response.choices[0].message.content)


# ────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
@rate_limit
def analyze():
    """Access Engine endpoint – returns a personal profile."""
    text, err = _extract_text()
    if err:
        return err
    try:
        return jsonify({"profile": analyze_person(text)})
    except Exception:
        logger.exception("analyze_person failed")
        return jsonify({"error": "שגיאה פנימית. נסה שוב מאוחר יותר."}), 500


@app.route("/emotions", methods=["POST"])
@rate_limit
def emotions():
    """Mirror Engine endpoint – returns emotional analysis."""
    text, err = _extract_text()
    if err:
        return err
    try:
        return jsonify({"emotions": analyze_emotions(text)})
    except Exception:
        logger.exception("analyze_emotions failed")
        return jsonify({"error": "שגיאה פנימית. נסה שוב מאוחר יותר."}), 500


@app.route("/profile", methods=["POST"])
@rate_limit
def full_profile():
    """Combined endpoint – returns both profile and emotional analysis."""
    text, err = _extract_text()
    if err:
        return err
    try:
        return jsonify({
            "profile": analyze_person(text),
            "emotions": analyze_emotions(text),
        })
    except Exception:
        logger.exception("full_profile failed")
        return jsonify({"error": "שגיאה פנימית. נסה שוב מאוחר יותר."}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # debug=False חובה בפרודקשן – debug=True מאפשר הרצת קוד שרירותי מהדפדפן.
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug, port=5000)
