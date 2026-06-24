from openai import OpenAI

from config import config
from models import EmotionalProfile

_SYSTEM = """
אתה מומחה לניתוח רגשי ופסיכולוגי. נתח את הטקסט הבא וזהה את המצב הרגשי של האדם.
היה מדויק ואמפתי. אל תנחש מעבר למה שניתן להסיק מהטקסט.
החזר JSON תקני בלבד:
{
  "dominant_emotion": string,
  "secondary_emotions": [string],
  "emotional_intensity": integer (1-10),
  "mood": string,
  "stress_level": "נמוך"|"בינוני"|"גבוה"|"קריטי",
  "anxiety_indicators": [string],
  "emotional_stability": string,
  "underlying_needs": [string],
  "summary": string
}
"""


def run_mirror_engine(text: str, client: OpenAI) -> EmotionalProfile:
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        max_tokens=500,
    )
    return EmotionalProfile.model_validate_json(response.choices[0].message.content)
