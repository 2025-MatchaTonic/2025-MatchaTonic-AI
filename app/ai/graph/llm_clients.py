import logging
from time import perf_counter

from langchain_openai import ChatOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

LLM_MODEL = settings.OPENAI_MODEL

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


def invoke_llm(llm: ChatOpenAI, prompt: str, *, label: str, **kwargs):
    started_at = perf_counter()
    try:
        response = llm.invoke(prompt, **kwargs)
    except Exception:
        logger.exception("%s failed in %.2fs", label, perf_counter() - started_at)
        return None
    logger.info("%s completed in %.2fs", label, perf_counter() - started_at)
    return response
