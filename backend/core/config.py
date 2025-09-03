"""
Centralized configuration using environment variables.
"""
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    app_env: str = os.getenv("APP_ENV", "dev")
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    cors_origins: list[str] = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")

    # Optional database (pgvector)
    database_url: str | None = os.getenv("DATABASE_URL")

    # Optional LLM
    llm_provider: str | None = os.getenv("LLM_PROVIDER")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")

    # NEW: default summary length (words)
    summary_words_default: int = int(os.getenv("SUMMARY_WORDS_DEFAULT", "350"))

settings = Settings()
