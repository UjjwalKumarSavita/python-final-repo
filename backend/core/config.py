# """
# Centralized configuration using environment variables.
# """
# from pydantic import BaseModel
# from dotenv import load_dotenv
# import os

# load_dotenv()

# class Settings(BaseModel):
#     app_env: str = os.getenv("APP_ENV", "dev")
#     api_host: str = os.getenv("API_HOST", "127.0.0.1")
#     api_port: int = int(os.getenv("API_PORT", "8000"))
#     cors_origins: list[str] = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")

#     # Optional database (pgvector)
#     database_url: str | None = os.getenv("DATABASE_URL")

#     # Optional LLM
#     llm_provider: str | None = os.getenv("LLM_PROVIDER")
#     openai_api_key: str | None = os.getenv("OPENAI_API_KEY")

#     # NEW: default summary length (words)
#     summary_words_default: int = int(os.getenv("SUMMARY_WORDS_DEFAULT", "350"))

# settings = Settings()


# backend/core/config.py
from __future__ import annotations
from typing import List, Union
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "dev"
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # Can be "*" OR a comma-separated string OR a JSON-like list in env
    cors_origins: Union[str, List[str]] = "*"

    summary_words_default: int = 350

    # Vector store
    use_pgvector: int = 0
    database_url: str | None = None
    pgvector_dim: int = 384

    # LLM (optional)
    llm_provider: str | None = None
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    openai_api_version: str | None = None

    def parsed_cors(self) -> List[str]:
        v = self.cors_origins
        if v is None or v == "*" or (isinstance(v, list) and v == ["*"]):
            return ["*"]
        if isinstance(v, (list, tuple, set)):
            return [str(o) for o in v]
        # string case: "http://a.com, http://b.com"
        return [o.strip() for o in str(v).split(",") if o.strip()]

settings = Settings()




# # # backend/core/config.py
# # from __future__ import annotations
# # from pydantic_settings import BaseSettings, SettingsConfigDict
# # from typing import List

# # class Settings(BaseSettings):
# #     model_config = SettingsConfigDict(
# #         env_file=".env",
# #         env_file_encoding="utf-8",
# #         extra="ignore",
# #     )

# #     app_env: str = "dev"
# #     api_host: str = "127.0.0.1"
# #     api_port: int = 8000

# #     # Comma-separated list or "*" for all
# #     cors_origins: str = "*"

# #     summary_words_default: int = 350

# #     # Vector store
# #     use_pgvector: int = 0
# #     database_url: str | None = None
# #     pgvector_dim: int = 384

# #     # LLM (optional)
# #     llm_provider: str | None = None
# #     openai_api_key: str | None = None
# #     openai_model: str = "gpt-4o-mini"
# #     openai_base_url: str | None = None
# #     openai_api_version: str | None = None

# #     def parsed_cors(self) -> List[str]:
# #         if not self.cors_origins or self.cors_origins == "*":
# #             return ["*"]
# #         return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

# # settings = Settings()
