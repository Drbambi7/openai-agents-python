import os


class Config:
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o")
    OPENAI_TIMEOUT: float = 45.0
    MAX_TEXT_LEN: int = 3_000
    MAX_BODY_BYTES: int = 64 * 1024
    RATE_LIMIT_CALLS: int = 15
    RATE_LIMIT_WINDOW: int = 60
    HTTPS_ENABLED: bool = os.environ.get("HTTPS_ENABLED", "false").lower() == "true"
    FLASK_DEBUG: bool = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    PORT: int = int(os.environ.get("PORT", 5000))


config = Config()
