"""Global configuration and path management using Pydantic Settings."""
from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
EVALS_DIR = ROOT_DIR / "evals"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# Ensure output directory exists (fallback to /tmp on read-only serverless filesystems)
try:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
except (OSError, PermissionError):
    OUTPUTS_DIR = Path("/tmp/outputs")
    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Service & Database
    ENVIRONMENT: str = "development"
    DATABASE_URL: str = "sqlite:///./chronexa.db"
    DEMO_MODE: bool = False
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # Rate Limiting & Networking
    RATE_LIMIT_PER_MINUTE: int = 60
    CORS_ALLOWED_ORIGINS: Optional[str] = None
    FASTEMBED_CACHE_DIR: Optional[str] = None

    # Auth & Security
    JWT_SECRET_KEY: str = "chronexa-insecure-dev-secret-key-change-in-prod"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440  # 24 hours

    # Runtime Modes
    CHRONEXA_OFFLINE: str = "0"
    INTEGRATION_MODE: str = "mock"  # "mock" or "live"

    # LLM Providers & Keys
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "qwen/qwen3.8-27b"

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"

    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-5-haiku-20241022"

    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-flash-latest"

    CLAUDE_MODEL: str = "haiku"

    # External Integrations (Fallback environment credentials)
    SLACK_BOT_TOKEN: Optional[str] = None
    GMAIL_CREDENTIALS: Optional[str] = None
    CALENDAR_CREDENTIALS: Optional[str] = None


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
