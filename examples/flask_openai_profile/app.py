"""
Flask app with two AI engines for person profiling.

Access Engine  – extracts a structured profile from free text.
Mirror Engine  – detects emotions and psychological tone from free text.

Run:
    pip install flask openai
    export OPENAI_API_KEY=sk-...
    python app.py
"""

import os

from flask import Flask, jsonify, request
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ────────────────────────────────────────────────
# Access Engine – ניתוח פרופיל אישי
# ────────────────────────────────────────────────

def analyze_person(input_text: str) -> str:
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
    return response.choices[0].message.content


# ────────────────────────────────────────────────
# Mirror Engine – ניתוח רגשות
# ────────────────────────────────────────────────

def analyze_emotions(input_text: str) -> str:
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
    return response.choices[0].message.content


# ────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    """Access Engine endpoint – returns a personal profile."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "שדה 'text' נדרש."}), 400
    try:
        import json
        profile = json.loads(analyze_person(text))
        return jsonify({"profile": profile})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/emotions", methods=["POST"])
def emotions():
    """Mirror Engine endpoint – returns emotional analysis."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "שדה 'text' נדרש."}), 400
    try:
        import json
        analysis = json.loads(analyze_emotions(text))
        return jsonify({"emotions": analysis})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/profile", methods=["POST"])
def full_profile():
    """Combined endpoint – returns both profile and emotional analysis."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "שדה 'text' נדרש."}), 400
    try:
        import json
        profile = json.loads(analyze_person(text))
        emotions_data = json.loads(analyze_emotions(text))
        return jsonify({"profile": profile, "emotions": emotions_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
