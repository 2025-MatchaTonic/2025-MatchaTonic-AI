import json
import logging
import re
from json import JSONDecodeError

from app.ai.graph.conversation_heuristics import BUTTON_ONLY_PATTERN
from app.ai.graph.llm_clients import invoke_llm as _invoke_llm, structured_llm
from app.ai.graph.text_support import clean_text as _clean_text
from app.core.config import settings

logger = logging.getLogger(__name__)

INITIAL_BUTTON_TOKENS = {
    "",
    "yes",
    "y",
    "no",
    "n",
    "네",
    "예",
    "응",
    "ㅇㅇ",
    "아니오",
    "아니요",
    "ㄴㄴ",
    "있음",
    "없음",
    "있어요",
    "없어요",
    "주제있음",
    "주제없음",
}

TOPIC_PRESENCE_CACHE_MAX_ITEMS = 128
TOPIC_PRESENCE_CONFIDENCE_THRESHOLD = 0.8
TOPIC_PRESENCE_CLASSIFICATION_CACHE: dict[str, tuple[str, float]] = {}
TOPIC_PRESENCE_POSITIVE_TOKENS = {
    "yes",
    "y",
    "네",
    "예",
    "응",
    "ㅇㅇ",
    "있음",
    "있어요",
    "주제있음",
}
TOPIC_PRESENCE_NEGATIVE_TOKENS = {
    "no",
    "n",
    "노",
    "아니오",
    "아니요",
    "ㄴㄴ",
    "없음",
    "없어요",
    "주제없음",
}
TOPIC_PRESENCE_POSITIVE_PHRASES = (
    "주제가 있습니다",
    "주제 있습니다",
    "정해진 주제가 있습니다",
    "생각해둔 주제가 있습니다",
    "이미 주제가 있습니다",
)
TOPIC_PRESENCE_NEGATIVE_PHRASES = (
    "주제가 없습니다",
    "주제 없습니다",
    "정해진 주제가 없습니다",
    "아직 주제가 없습니다",
    "주제는 아직 없습니다",
)
TOPIC_PRESENCE_EXPLICIT_FACT_PATTERN = re.compile(
    r"(?:주제|프로젝트|아이디어)\s*(?:은|는|:)\s*.+",
    re.IGNORECASE,
)
TOPIC_PRESENCE_LLM_TRIGGER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:주제|아이디어).*(?:있|없|미정|정했|정한|생각)", re.IGNORECASE),
    re.compile(r"(?:정한\s*건\s*없|정한건\s*없|못\s*정했|아직\s*못\s*정했)", re.IGNORECASE),
    re.compile(r"(?:같이\s*정하고\s*싶|정하고\s*싶|정해\s*줘|도와\s*줘)", re.IGNORECASE),
    re.compile(r"(?:없고|없어서).*(?:같이\s*정|정해|도와)", re.IGNORECASE),
    re.compile(r"(?:이미|벌써).*(?:있|정했|생각해둔)", re.IGNORECASE),
)


def _normalize_button_token(value: object) -> str:
    lowered = str(value or "").strip().lower()
    return BUTTON_ONLY_PATTERN.sub("", lowered)


def _topic_presence_cache_key(message: object) -> str:
    cleaned = _clean_text(message).strip().lower()
    if cleaned:
        return cleaned
    return _normalize_button_token(message)


def _store_topic_presence_classification(cache_key: str, label: str, confidence: float) -> None:
    if not cache_key:
        return
    TOPIC_PRESENCE_CLASSIFICATION_CACHE[cache_key] = (label, confidence)
    if len(TOPIC_PRESENCE_CLASSIFICATION_CACHE) > TOPIC_PRESENCE_CACHE_MAX_ITEMS:
        oldest_key = next(iter(TOPIC_PRESENCE_CLASSIFICATION_CACHE))
        TOPIC_PRESENCE_CLASSIFICATION_CACHE.pop(oldest_key, None)


def _rule_based_topic_presence_label(message: object) -> str | None:
    normalized = _normalize_button_token(message)
    if not normalized:
        return None
    if normalized in TOPIC_PRESENCE_POSITIVE_TOKENS:
        return "has_topic"
    if normalized in TOPIC_PRESENCE_NEGATIVE_TOKENS:
        return "no_topic"

    cleaned = _clean_text(message).lower()
    if not cleaned:
        return None
    if TOPIC_PRESENCE_EXPLICIT_FACT_PATTERN.search(cleaned):
        return "not_topic_presence"
    if any(phrase in cleaned for phrase in TOPIC_PRESENCE_POSITIVE_PHRASES):
        return "has_topic"
    if any(phrase in cleaned for phrase in TOPIC_PRESENCE_NEGATIVE_PHRASES):
        return "no_topic"
    if "주제" not in cleaned and "프로젝트" not in cleaned:
        return None

    if len(cleaned) <= 25 and any(keyword in cleaned for keyword in ("없", "미정", "아직")):
        return "no_topic"
    if len(cleaned) <= 25 and any(
        keyword in cleaned for keyword in ("있", "정해", "정했", "생각해")
    ):
        return "has_topic"
    return None


def _should_try_llm_topic_presence_classification(message: object) -> bool:
    if not settings.OPENAI_API_KEY:
        return False

    cleaned = _clean_text(message)
    if not cleaned:
        return False
    if TOPIC_PRESENCE_EXPLICIT_FACT_PATTERN.search(cleaned):
        return False
    if len(cleaned) > 120:
        return False

    normalized = _normalize_button_token(message)
    if normalized in TOPIC_PRESENCE_POSITIVE_TOKENS | TOPIC_PRESENCE_NEGATIVE_TOKENS:
        return False

    return any(pattern.search(cleaned) for pattern in TOPIC_PRESENCE_LLM_TRIGGER_PATTERNS)


def _llm_topic_presence_label(message: object) -> tuple[str | None, float]:
    if not _should_try_llm_topic_presence_classification(message):
        return None, 0.0

    cache_key = _topic_presence_cache_key(message)
    cached = TOPIC_PRESENCE_CLASSIFICATION_CACHE.get(cache_key)
    if cached:
        return cached

    cleaned = _clean_text(message)
    prompt = f"""
    You classify one Korean chat message.

    Decide whether the message is only answering this question:
    "Do you already have a project topic in mind?"

    Labels:
    - has_topic: the user says they already have a topic or idea.
    - no_topic: the user says they do not have one yet, it is undecided, or they want help choosing.
    - not_topic_presence: the user is giving the actual topic/title, asking something else, or talking about another issue.
    - unclear: you cannot tell confidently.

    Rules:
    - If the message includes the actual topic content, choose not_topic_presence.
    - If the user says they have not decided yet and wants help choosing, choose no_topic.
    - Be conservative. Do not choose has_topic or no_topic unless the meaning is explicit.

    Message: {cleaned}

    Output JSON:
    {{
      "label": "has_topic | no_topic | not_topic_presence | unclear",
      "confidence": 0.0
    }}
    """

    response = _invoke_llm(
        structured_llm,
        prompt,
        label="topic_presence_classifier llm",
        response_format={"type": "json_object"},
    )
    if response is None:
        return None, 0.0

    try:
        raw_result = json.loads(response.content)
    except JSONDecodeError:
        logger.warning(
            "failed to parse topic presence classifier JSON: raw_output=%s",
            getattr(response, "content", ""),
        )
        return None, 0.0

    label = str(raw_result.get("label") or "").strip()
    if label not in {"has_topic", "no_topic", "not_topic_presence", "unclear"}:
        return None, 0.0

    try:
        confidence = float(raw_result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))
    _store_topic_presence_classification(cache_key, label, confidence)
    return label, confidence


def _classify_topic_presence_message(message: object) -> str:
    cache_key = _topic_presence_cache_key(message)
    cached = TOPIC_PRESENCE_CLASSIFICATION_CACHE.get(cache_key)
    if cached and cached[1] >= TOPIC_PRESENCE_CONFIDENCE_THRESHOLD:
        return cached[0]

    rule_based_label = _rule_based_topic_presence_label(message)
    if rule_based_label:
        _store_topic_presence_classification(cache_key, rule_based_label, 1.0)
        return rule_based_label

    llm_label, llm_confidence = _llm_topic_presence_label(message)
    if llm_label and llm_confidence >= TOPIC_PRESENCE_CONFIDENCE_THRESHOLD:
        logger.info(
            "topic presence classified with llm message=%r label=%s confidence=%.2f",
            _clean_text(message),
            llm_label,
            llm_confidence,
        )
        return llm_label

    return "not_topic_presence"


def _matches_topic_presence_button_message(message: object) -> bool:
    label = _classify_topic_presence_message(message)
    return label in {"has_topic", "no_topic"}


def _is_topic_presence_negative_message(message: object) -> bool:
    return _classify_topic_presence_message(message) == "no_topic"


def _matches_initial_button_message(action: str, message: object) -> bool:
    normalized = _normalize_button_token(message)
    if not normalized:
        return True
    if normalized in INITIAL_BUTTON_TOKENS:
        return True

    cleaned = _clean_text(message).lower()
    if action in {"BTN_YES", "BTN_GO_DEF"}:
        return (
            ("주제" in cleaned or "프로젝트" in cleaned)
            and any(keyword in cleaned for keyword in ("있", "정해", "정했", "생각해"))
        )
    if action == "BTN_NO":
        return (
            ("주제" in cleaned or "프로젝트" in cleaned)
            and any(keyword in cleaned for keyword in ("없", "미정", "아직"))
        )
    return False
