from openai import OpenAI

from config import config
from models import PersonProfile

_SYSTEM = """
אתה מנתח פרופילים פסיכולוגיים מקצועי. בהינתן טקסט, חלץ פרופיל מפורט של האדם.
הסק רק ממה שמוזכר במפורש או עולה בבירור מהטקסט — אל תמציא.
כאשר אין מידע מספיק לשדה מסוים, החזר null.
החזר JSON תקני בלבד התואם לסכמה הבאה:
{
  "name": string|null,
  "age_estimate": string|null,
  "gender": string|null,
  "profession": string|null,
  "education_level": string|null,
  "personality_type": string|null,
  "core_values": [string],
  "life_goals": [string],
  "interests": [string],
  "strengths": [string],
  "weaknesses": [string],
  "communication_style": string|null
}
"""


def run_access_engine(text: str, client: OpenAI) -> PersonProfile:
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        max_tokens=600,
    )
    return PersonProfile.model_validate_json(response.choices[0].message.content)
