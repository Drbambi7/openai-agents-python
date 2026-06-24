"""
Profile AI — Flask application with five analysis engines.

Engines:
  Access Engine   – personal profile (personality, values, strengths)
  Mirror Engine   – emotional state and underlying needs
  Language Engine – speech patterns and communication style
  Shadow Engine   – red flags, manipulation detection, trust score
  Summary Engine  – executive summary with interaction tips

Run:
    pip install flask openai pydantic
    export OPENAI_API_KEY=sk-...
    python app.py
"""

import json
import logging
import sys

from flask import Flask, jsonify, render_template, request
from openai import OpenAI

from config import config
from engines import (
    run_access_engine,
    run_mirror_engine,
    run_language_engine,
    run_shadow_engine,
    run_summary_engine,
)
from security import add_security_headers, extract_text, rate_limit

# ── Startup validation ─────────────────────────────────────────────────────────

if not config.OPENAI_API_KEY:
    sys.exit(
        "ERROR: OPENAI_API_KEY is not set.\n"
        "Export it before starting:\n"
        "  export OPENAI_API_KEY=sk-..."
    )

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_BODY_BYTES
app.after_request(add_security_headers)

openai_client = OpenAI(api_key=config.OPENAI_API_KEY, timeout=config.OPENAI_TIMEOUT)

# ── UI ─────────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# ── API v1 ─────────────────────────────────────────────────────────────────────

@app.route("/api/v1/analyze", methods=["POST"])
@rate_limit
def analyze():
    text, err = extract_text()
    if err:
        return err
    try:
        profile = run_access_engine(text, openai_client)
        return jsonify({"profile": profile.model_dump()})
    except Exception:
        logger.exception("access engine failed")
        return jsonify({"error": "שגיאה פנימית. נסה שוב מאוחר יותר."}), 500


@app.route("/api/v1/emotions", methods=["POST"])
@rate_limit
def emotions():
    text, err = extract_text()
    if err:
        return err
    try:
        result = run_mirror_engine(text, openai_client)
        return jsonify({"emotions": result.model_dump()})
    except Exception:
        logger.exception("mirror engine failed")
        return jsonify({"error": "שגיאה פנימית. נסה שוב מאוחר יותר."}), 500


@app.route("/api/v1/language", methods=["POST"])
@rate_limit
def language():
    text, err = extract_text()
    if err:
        return err
    try:
        result = run_language_engine(text, openai_client)
        return jsonify({"language": result.model_dump()})
    except Exception:
        logger.exception("language engine failed")
        return jsonify({"error": "שגיאה פנימית. נסה שוב מאוחר יותר."}), 500


@app.route("/api/v1/shadow", methods=["POST"])
@rate_limit
def shadow():
    text, err = extract_text()
    if err:
        return err
    try:
        result = run_shadow_engine(text, openai_client)
        return jsonify({"shadow": result.model_dump()})
    except Exception:
        logger.exception("shadow engine failed")
        return jsonify({"error": "שגיאה פנימית. נסה שוב מאוחר יותר."}), 500


@app.route("/api/v1/profile", methods=["POST"])
@rate_limit
def full_profile():
    """Run all five engines and return a combined analysis."""
    text, err = extract_text()
    if err:
        return err
    try:
        profile   = run_access_engine(text, openai_client)
        emo       = run_mirror_engine(text, openai_client)
        lang      = run_language_engine(text, openai_client)
        shadow_r  = run_shadow_engine(text, openai_client)

        combined = (
            f"PROFILE:\n{json.dumps(profile.model_dump(), ensure_ascii=False)}\n\n"
            f"EMOTIONS:\n{json.dumps(emo.model_dump(), ensure_ascii=False)}\n\n"
            f"LANGUAGE:\n{json.dumps(lang.model_dump(), ensure_ascii=False)}\n\n"
            f"SHADOW:\n{json.dumps(shadow_r.model_dump(), ensure_ascii=False)}"
        )
        summary = run_summary_engine(combined, openai_client)

        return jsonify({
            "profile":  profile.model_dump(),
            "emotions": emo.model_dump(),
            "language": lang.model_dump(),
            "shadow":   shadow_r.model_dump(),
            "summary":  summary.model_dump(),
        })
    except Exception:
        logger.exception("full profile failed")
        return jsonify({"error": "שגיאה פנימית. נסה שוב מאוחר יותר."}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": config.OPENAI_MODEL})


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=config.FLASK_DEBUG, port=config.PORT)
