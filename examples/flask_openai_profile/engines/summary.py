from openai import OpenAI

from config import config
from models import ComprehensiveSummary

_SYSTEM = """
אתה יועץ אסטרטגי המתמחה בהערכת אנשים. קיבלת ניתוחים ממספר מנועים.
צור סיכום מקצועי, ממוקד ומעשי שעוזר להבין עם מי מתעסקים.
overall_score: 1=לא אמין בכלל, 10=אמין לחלוטין.
החזר JSON תקני בלבד:
{
  "headline": string,
  "executive_summary": string,
  "key_insights": [string],
  "interaction_tips": [string],
  "watch_out_for": [string],
  "overall_score": integer (1-10)
}
"""


def run_summary_engine(combined_analysis: str, client: OpenAI) -> ComprehensiveSummary:
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": combined_analysis},
        ],
        response_format={"type": "json_object"},
        max_tokens=600,
    )
    return ComprehensiveSummary.model_validate_json(response.choices[0].message.content)
