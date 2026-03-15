# 환경변수 로드 (Pydantic Settings)
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _csv_env(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or default


class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini-2025-08-07")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ai-pm-knowledge")
    PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")
    AI_CORS_ALLOW_ORIGINS = _csv_env("AI_CORS_ALLOW_ORIGINS", ["*"])
    REQUIRE_OPENAI_API_KEY = _bool_env("REQUIRE_OPENAI_API_KEY", True)
    REQUIRE_PINECONE_FOR_RAG = _bool_env("REQUIRE_PINECONE_FOR_RAG", False)

    RAG_TOP_K = _int_env("RAG_TOP_K", 4)
    RAG_MAX_CONTEXT_CHARS = _int_env("RAG_MAX_CONTEXT_CHARS", 2400)
    RAG_MAX_DOC_CHARS = _int_env("RAG_MAX_DOC_CHARS", 600)
    RAG_PHASE_FILTER_ENABLED = _bool_env("RAG_PHASE_FILTER_ENABLED", True)


settings = Settings()
