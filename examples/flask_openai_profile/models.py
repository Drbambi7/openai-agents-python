from pydantic import BaseModel, Field


# ── Access Engine ──────────────────────────────────────────────────────────────

class PersonProfile(BaseModel):
    name: str | None = Field(None, description="שם האדם אם מוזכר")
    age_estimate: str | None = Field(None, description="הערכת גיל (טווח)")
    gender: str | None = Field(None, description="מגדר אם ניתן להסיק")
    profession: str | None = Field(None, description="מקצוע / תחום עיסוק")
    education_level: str | None = Field(None, description="רמת השכלה משוערת")
    personality_type: str | None = Field(None, description="סוג אישיות (אנליטי/יצירתי/חברותי/מוביל)")
    core_values: list[str] = Field(default_factory=list, description="ערכי ליבה")
    life_goals: list[str] = Field(default_factory=list, description="מטרות חיים")
    interests: list[str] = Field(default_factory=list, description="תחומי עניין")
    strengths: list[str] = Field(default_factory=list, description="חוזקות")
    weaknesses: list[str] = Field(default_factory=list, description="נקודות חולשה")
    communication_style: str | None = Field(None, description="סגנון תקשורת")


# ── Mirror Engine ──────────────────────────────────────────────────────────────

class EmotionalProfile(BaseModel):
    dominant_emotion: str = Field(..., description="רגש דומיננטי")
    secondary_emotions: list[str] = Field(default_factory=list, description="רגשות משניים")
    emotional_intensity: int = Field(..., ge=1, le=10, description="עוצמת רגש 1-10")
    mood: str = Field(..., description="מצב רוח כללי")
    stress_level: str = Field(..., description="רמת מתח: נמוך/בינוני/גבוה/קריטי")
    anxiety_indicators: list[str] = Field(default_factory=list, description="סימני חרדה")
    emotional_stability: str = Field(..., description="יציבות רגשית")
    underlying_needs: list[str] = Field(default_factory=list, description="צרכים רגשיים בסיסיים")
    summary: str = Field(..., description="סיכום קצר")


# ── Language Engine ────────────────────────────────────────────────────────────

class LanguageProfile(BaseModel):
    formality_level: int = Field(..., ge=1, le=10, description="רמת פורמליות 1-10")
    assertiveness: int = Field(..., ge=1, le=10, description="נחרצות 1-10")
    vocabulary_richness: str = Field(..., description="עושר אוצר מילים: בסיסי/ממוצע/עשיר/מומחה")
    dominant_tone: str = Field(..., description="טון דומיננטי")
    communication_patterns: list[str] = Field(default_factory=list, description="דפוסי תקשורת בולטים")
    honesty_indicators: str = Field(..., description="אינדיקטורי כנות: גבוה/ממוצע/נמוך")
    deflection_signs: list[str] = Field(default_factory=list, description="סימני הסטה / הימנעות")
    key_phrases: list[str] = Field(default_factory=list, description="ביטויים מאפיינים")


# ── Shadow Engine ──────────────────────────────────────────────────────────────

class ShadowProfile(BaseModel):
    red_flags: list[str] = Field(default_factory=list, description="דגלים אדומים")
    manipulation_tactics: list[str] = Field(default_factory=list, description="טקטיקות מניפולציה")
    narcissism_score: int = Field(..., ge=0, le=10, description="מדד נרקיסיזם 0-10")
    control_tendency: int = Field(..., ge=0, le=10, description="נטייה לשליטה 0-10")
    empathy_level: int = Field(..., ge=0, le=10, description="רמת אמפתיה 0-10")
    trust_recommendation: str = Field(..., description="המלצת אמון: גבוהה/זהירות/נמוכה")
    risk_summary: str = Field(..., description="סיכום סיכון")


# ── Summary Engine ─────────────────────────────────────────────────────────────

class ComprehensiveSummary(BaseModel):
    headline: str = Field(..., description="כותרת תמציתית לאדם")
    executive_summary: str = Field(..., description="סיכום מנהלים 2-3 משפטים")
    key_insights: list[str] = Field(default_factory=list, description="תובנות מפתח")
    interaction_tips: list[str] = Field(default_factory=list, description="טיפים לתקשורת עם האדם")
    watch_out_for: list[str] = Field(default_factory=list, description="על מה להיזהר")
    overall_score: int = Field(..., ge=1, le=10, description="ציון אמינות כללי 1-10")
