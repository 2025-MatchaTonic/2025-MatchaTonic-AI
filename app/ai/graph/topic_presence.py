import re
import logging

logger = logging.getLogger(__name__)

_MATES_MENTION_PATTERN = re.compile(r"@mates\b", re.IGNORECASE)
_BTN_NO_PATTERN = re.compile(r"^btn[_\-\s]*no$", re.IGNORECASE)
_BTN_YES_PATTERN = re.compile(r"^btn[_\-\s]*(?:yes|go[_\-\s]*def)$", re.IGNORECASE)
# "아니오, 아직 주제가 없습니다" 같이 부정으로 시작하는 문장형 표현
_NEGATIVE_SENTENCE_PATTERN = re.compile(
    r"^\s*(?:아니오|아니요|아니|ㄴㄴ|없어요?|주제\s*없|.*주제(?:가|는)?\s*없)",
    re.IGNORECASE,
)
_POSITIVE_SENTENCE_PATTERN = re.compile(
    r"^\s*(?:네|예|응|ㅇㅇ|있어요?|주제\s*있|.*주제(?:가|는)?\s*있)",
    re.IGNORECASE,
)

_BUTTON_NORMALIZE_PATTERN = re.compile(r"[\s\.\,\!\?]+")

INITIAL_BUTTON_TOKENS: frozenset[str] = frozenset({
    "",
    "yes", "y", "no", "n",
    "btnyes", "btnno", "btngodef",
    "네", "예", "응", "ㅇㅇ",
    "아니오", "아니요", "ㄴㄴ",
    "있음", "없음", "있어요", "없어요",
    "주제있음", "주제없음",
})

TOPIC_PRESENCE_POSITIVE_TOKENS: frozenset[str] = frozenset({
    "yes", "y", "네", "예", "응", "ㅇㅇ", "있음", "있어요", "주제있음",
    "btnyes", "btngodef",
})

TOPIC_PRESENCE_NEGATIVE_TOKENS: frozenset[str] = frozenset({
    "no", "n", "노", "아니오", "아니요", "ㄴㄴ", "없음", "없어요", "주제없음",
    "btnno",
})


def _normalize_button_token(value: object) -> str:
    lowered = _MATES_MENTION_PATTERN.sub(" ", str(value or "")).strip().lower()
    return _BUTTON_NORMALIZE_PATTERN.sub("", lowered)


def _matches_topic_presence_button_message(message: object) -> bool:
    normalized = _normalize_button_token(message)
    cleaned = _MATES_MENTION_PATTERN.sub(" ", str(message or "")).strip()
    return (
        normalized in TOPIC_PRESENCE_POSITIVE_TOKENS | TOPIC_PRESENCE_NEGATIVE_TOKENS
        or bool(_BTN_NO_PATTERN.match(cleaned))
        or bool(_BTN_YES_PATTERN.match(cleaned))
        or bool(_NEGATIVE_SENTENCE_PATTERN.match(cleaned))
        or bool(_POSITIVE_SENTENCE_PATTERN.match(cleaned))
    )


def _is_topic_presence_negative_message(message: object) -> bool:
    normalized = _normalize_button_token(message)
    if normalized in TOPIC_PRESENCE_NEGATIVE_TOKENS:
        return True
    # 단순 토큰이 아닌 문장형 부정 표현도 처리 ("아니오, 아직 주제가 없습니다" 등)
    cleaned = _MATES_MENTION_PATTERN.sub(" ", str(message or "")).strip()
    return bool(_BTN_NO_PATTERN.match(cleaned)) or bool(_NEGATIVE_SENTENCE_PATTERN.match(cleaned))


def _matches_initial_button_message(action: str, message: object) -> bool:
    normalized = _normalize_button_token(message)
    return not normalized or normalized in INITIAL_BUTTON_TOKENS
