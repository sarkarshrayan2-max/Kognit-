from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    app_name: str = "KOGNIT"
    environment: str = "development"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    qdrant_collection: str = "ecs_knowledge_base"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "kognit_admin"
    postgres_password: str = "kognit_password"
    postgres_db: str = "kognit_db"

    redis_host: str = "localhost"
    redis_port: int = 6379

    groq_api_key: str | None = None

    groq_model: str = "qwen/qwen3.6-27b"

    tavily_api_key: str | None = None

    dense_model: str = "BAAI/bge-large-en-v1.5"

    sparse_model: str = "Qdrant/bm25"

    reranker_model: str = "BAAI/bge-reranker-large"


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()