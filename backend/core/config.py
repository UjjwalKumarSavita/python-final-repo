"""
Centralized configuration using environment variables.
Keeps function signatures stable across modules.
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

    # Future (Milestone 2+): swap in Postgres/pgvector
    database_url: str | None = os.getenv("DATABASE_URL")

    # Future (Milestone 5): LangSmith / MCP
    langsmith_api_key: str | None = os.getenv("LANGSMITH_API_KEY")
    langsmith_project: str | None = os.getenv("LANGSMITH_PROJECT")

settings = Settings()
