import logging
from time import perf_counter
from types import SimpleNamespace

from langchain_openai import ChatOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

LLM_MODEL = settings.OPENAI_MODEL
LLM_RESPONSE_CACHE_MAX_ITEMS = 128
LLM_RESPONSE_CACHE: dict[tuple[object, ...], str] = {}

conversation_llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0.7,
    openai_api_key=settings.OPENAI_API_KEY,
    timeout=settings.OPENAI_TIMEOUT_SECONDS,
    max_retries=settings.OPENAI_MAX_RETRIES,
)

structured_llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0.2,
    openai_api_key=settings.OPENAI_API_KEY,
    timeout=settings.OPENAI_TIMEOUT_SECONDS,
    max_retries=settings.OPENAI_MAX_RETRIES,
)


def invoke_llm(
    llm: ChatOpenAI,
    prompt: str,
    *,
    label: str,
    cache_key: tuple[object, ...] | None = None,
    **kwargs,
):
    if cache_key is not None and cache_key in LLM_RESPONSE_CACHE:
        logger.info("%s cache hit", label)
        return SimpleNamespace(content=LLM_RESPONSE_CACHE[cache_key])

    started_at = perf_counter()
    try:
        response = llm.invoke(prompt, **kwargs)
    except Exception:
        logger.exception("%s failed in %.2fs", label, perf_counter() - started_at)
        return None
    if cache_key is not None:
        if len(LLM_RESPONSE_CACHE) >= LLM_RESPONSE_CACHE_MAX_ITEMS:
            oldest_key = next(iter(LLM_RESPONSE_CACHE))
            LLM_RESPONSE_CACHE.pop(oldest_key, None)
        LLM_RESPONSE_CACHE[cache_key] = str(getattr(response, "content", ""))
    logger.info("%s completed in %.2fs", label, perf_counter() - started_at)
    return response
