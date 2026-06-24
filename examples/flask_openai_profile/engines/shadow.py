from openai import OpenAI

from config import config
from models import ShadowProfile

_SYSTEM = """
אתה מומחה לפסיכולוגיה חברתית וזיהוי דפוסים מניפולטיביים.
נתח את הטקסט וזהה סימני אזהרה, דפוסי מניפולציה, ונטיות אישיות בעייתיות.
היה אובייקטיבי. ציונים של 0 פירושם "ללא עדות", 10 פירושם "עדות חזקה מאוד".
אם אין עדות לדגלים אדומים, החזר רשימות ריקות וציונים נמוכים.
החזר JSON תקני בלבד:
{
  "red_flags": [string],
  "manipulation_tactics": [string],
  "narcissism_score": integer (0-10),
  "control_tendency": integer (0-10),
  "empathy_level": integer (0-10),
  "trust_recommendation": "גבוהה"|"זהירות"|"נמוכה",
  "risk_summary": string
}
"""


def run_shadow_engine(text: str, client: OpenAI) -> ShadowProfile:
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        max_tokens=500,
    )
    return ShadowProfile.model_validate_json(response.choices[0].message.content)
