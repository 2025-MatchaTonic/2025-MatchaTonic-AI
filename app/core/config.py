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


class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ai-pm-knowledge")
    PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")

    RAG_TOP_K = _int_env("RAG_TOP_K", 4)
    RAG_MAX_CONTEXT_CHARS = _int_env("RAG_MAX_CONTEXT_CHARS", 2400)
    RAG_MAX_DOC_CHARS = _int_env("RAG_MAX_DOC_CHARS", 600)
    RAG_PHASE_FILTER_ENABLED = _bool_env("RAG_PHASE_FILTER_ENABLED", True)


settings = Settings()
