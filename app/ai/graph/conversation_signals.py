import json
import logging
import re
from json import JSONDecodeError

from app.ai.graph.collected_data import is_meta_conversation
from app.ai.graph.conversation_heuristics import (
    GUIDANCE_SIGNAL_PATTERNS,
    HELP_REQUEST_KEYWORDS,
    META_CONVERSATION_PATTERNS,
    SUMMARY_REQUEST_KEYWORDS,
)
from app.ai.graph.llm_clients import invoke_llm as _invoke_llm, structured_llm
from app.ai.graph.text_support import clean_text as _clean_text, strip_mates_mention
from app.core.config import settings

logger = logging.getLogger(__name__)

SIGNAL_CACHE_MAX_ITEMS = 128
SIGNAL_CONFIDENCE_THRESHOLD = 0.8
SIGNAL_CLASSIFICATION_CACHE: dict[str, tuple[str, float]] = {}

NEXT_STEP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:다음|그다음)(?:\s*으로)?\s*(?:할\s*일|액션|단계|스텝)", re.IGNORECASE
    ),
    re.compile(r"(?:뭐|무엇)을?\s*(?:하면|해야)", re.IGNORECASE),
    re.compile(r"(?:next\s*step|next\s*steps|action\s*items?)", re.IGNORECASE),
)
SUMMARY_STRONG_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b요약\b", re.IGNORECASE),
    re.compile(
        r"(?:지금\s*모인\s*정보|확정된\s*사항|확정된\s*정보|정리된\s*상황|정해진\s*사항)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:현재\s*확정|현재까지|세션\s*요약)", re.IGNORECASE),
)
SUMMARY_EXCLUSION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:우선순위|priority)\s*정리", re.IGNORECASE),
    re.compile(r"(?:다음|그다음)\s*할\s*일", re.IGNORECASE),
    re.compile(r"(?:action\s*items?|todo|to-do|체크리스트|로드맵)", re.IGNORECASE),
    re.compile(r"(?:실행\s*계획|진행\s*순서|mvp\s*기준)", re.IGNORECASE),
)
HELP_REQUEST_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:추천|예시|후보).*(?:해줘|주세요|주라)?", re.IGNORECASE),
    re.compile(r"(?:어떻게|왜|이유|설명).*(?:알려|말해|정리)", re.IGNORECASE),
    re.compile(r"(?:도와|가이드).*(?:줘|주세요|주라)?", re.IGNORECASE),
)


def _cache_key(message: str) -> str:
    return _clean_text(strip_mates_mention(message)).lower()


def _store_signal(cache_key: str, label: str, confidence: float) -> None:
    if not cache_key:
        return
    SIGNAL_CLASSIFICATION_CACHE[cache_key] = (label, confidence)
    if len(SIGNAL_CLASSIFICATION_CACHE) > SIGNAL_CACHE_MAX_ITEMS:
        oldest_key = next(iter(SIGNAL_CLASSIFICATION_CACHE))
        SIGNAL_CLASSIFICATION_CACHE.pop(oldest_key, None)


def _shortcut_signal_label(message: str) -> str | None:
    normalized = _clean_text(strip_mates_mention(message))
    if not normalized:
        return None

    if is_meta_conversation(normalized) or any(
        pattern.match(normalized) for pattern in META_CONVERSATION_PATTERNS
    ):
        return "meta_conversation"
    if any(pattern.search(normalized) for pattern in NEXT_STEP_PATTERNS):
        return "next_step_request"
    if any(pattern.search(normalized) for pattern in SUMMARY_EXCLUSION_PATTERNS):
        return "other"
    if any(pattern.search(normalized) for pattern in SUMMARY_STRONG_PATTERNS):
        return "summary_request"
    return None


def _fallback_signal_label(message: str) -> str | None:
    normalized = _clean_text(strip_mates_mention(message))
    if not normalized:
        return None

    if any(keyword in normalized for keyword in SUMMARY_REQUEST_KEYWORDS):
        return "summary_request"
    if any(pattern.search(normalized) for pattern in GUIDANCE_SIGNAL_PATTERNS):
        return "guidance_signal"
    if any(pattern.search(normalized) for pattern in HELP_REQUEST_PATTERNS):
        return "help_request"
    if any(keyword in normalized for keyword in HELP_REQUEST_KEYWORDS):
        return "help_request"
    return None


def _should_try_llm_signal_classification(message: str) -> bool:
    if not settings.OPENAI_API_KEY:
        return False

    normalized = _clean_text(strip_mates_mention(message))
    if not normalized:
        return False
    if len(normalized) > 240:
        return False
    return False


def _llm_signal_label(message: str) -> tuple[str | None, float]:
    if not _should_try_llm_signal_classification(message):
        return None, 0.0

    cache_key = _cache_key(message)
    cached = SIGNAL_CLASSIFICATION_CACHE.get(cache_key)
    if cached:
        return cached

    normalized = _clean_text(strip_mates_mention(message))
    prompt = f"""
    You classify one Korean project chat message.

    Labels:
    - summary_request: asking to summarize confirmed or collected information.
    - next_step_request: asking what to do next, what to decide next, or action items.
    - help_request: asking for explanation, recommendation, examples, or guidance in general.
    - guidance_signal: expressing uncertainty or asking the assistant to help decide.
    - meta_conversation: reacting to the assistant's wording, saying it is wrong, odd, or confusing.
    - other: none of the above.

    Rules:
    - "우선순위 정리", "다음 할 일 정리", "실행 계획" are next_step_request, not summary_request.
    - "정리해줘" alone is not enough for summary_request.
    - Be conservative. Use other if uncertain.

    Message: {normalized}

    Output JSON:
    {{
      "label": "summary_request | next_step_request | help_request | guidance_signal | meta_conversation | other",
      "confidence": 0.0
    }}
    """

    response = _invoke_llm(
        structured_llm,
        prompt,
        label="conversation_signal_classifier llm",
        response_format={"type": "json_object"},
    )
    if response is None:
        return None, 0.0

    try:
        raw_result = json.loads(response.content)
    except JSONDecodeError:
        logger.warning(
            "failed to parse conversation signal classifier JSON: raw_output=%s",
            getattr(response, "content", ""),
        )
        return None, 0.0

    label = str(raw_result.get("label") or "").strip()
    if label not in {
        "summary_request",
        "next_step_request",
        "help_request",
        "guidance_signal",
        "meta_conversation",
        "other",
    }:
        return None, 0.0

    try:
        confidence = float(raw_result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))
    _store_signal(cache_key, label, confidence)
    return label, confidence


def classify_signal(message: str) -> str:
    cache_key = _cache_key(message)
    cached = SIGNAL_CLASSIFICATION_CACHE.get(cache_key)
    if cached and cached[1] >= SIGNAL_CONFIDENCE_THRESHOLD:
        return cached[0]

    shortcut_label = _shortcut_signal_label(message)
    if shortcut_label:
        _store_signal(cache_key, shortcut_label, 1.0)
        return shortcut_label

    llm_label, llm_confidence = _llm_signal_label(message)
    if llm_label and llm_confidence >= SIGNAL_CONFIDENCE_THRESHOLD:
        _store_signal(cache_key, llm_label, llm_confidence)
        return llm_label

    fallback_label = _fallback_signal_label(message)
    if fallback_label:
        _store_signal(cache_key, fallback_label, 0.6)
        return fallback_label

    return "other"


def is_summary_request(message: str) -> bool:
    return classify_signal(message) == "summary_request"


def is_next_step_request(message: str) -> bool:
    return classify_signal(message) == "next_step_request"


def is_help_request(message: str) -> bool:
    return classify_signal(message) in {"help_request", "guidance_signal"}


def is_guidance_signal(message: str) -> bool:
    return classify_signal(message) == "guidance_signal"


def is_meta_conversation_message(message: str) -> bool:
    return classify_signal(message) == "meta_conversation"
