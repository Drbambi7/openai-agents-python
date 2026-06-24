from openai import OpenAI

from config import config
from models import LanguageProfile

_SYSTEM = """
אתה בלשן ומומחה לתקשורת. נתח את סגנון השפה של הכותב/הנאמר בטקסט.
התמקד בדפוסים לשוניים, לא בתוכן הסמנטי.
החזר JSON תקני בלבד:
{
  "formality_level": integer (1-10),
  "assertiveness": integer (1-10),
  "vocabulary_richness": "בסיסי"|"ממוצע"|"עשיר"|"מומחה",
  "dominant_tone": string,
  "communication_patterns": [string],
  "honesty_indicators": "גבוה"|"ממוצע"|"נמוך",
  "deflection_signs": [string],
  "key_phrases": [string]
}
"""


def run_language_engine(text: str, client: OpenAI) -> LanguageProfile:
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        max_tokens=500,
    )
    return LanguageProfile.model_validate_json(response.choices[0].message.content)
