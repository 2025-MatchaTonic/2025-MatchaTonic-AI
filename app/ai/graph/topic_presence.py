import re
import logging

logger = logging.getLogger(__name__)

# "아니오, 아직 주제가 없습니다" 같이 부정으로 시작하는 문장형 표현
_NEGATIVE_SENTENCE_PATTERN = re.compile(
    r"^\s*(?:아니오|아니요|ㄴㄴ|없어요?|주제\s*없)\b",
    re.IGNORECASE,
)

_BUTTON_NORMALIZE_PATTERN = re.compile(r"[\s\.\,\!\?]+")

INITIAL_BUTTON_TOKENS: frozenset[str] = frozenset({
    "",
    "yes", "y", "no", "n",
    "네", "예", "응", "ㅇㅇ",
    "아니오", "아니요", "ㄴㄴ",
    "있음", "없음", "있어요", "없어요",
    "주제있음", "주제없음",
})

TOPIC_PRESENCE_POSITIVE_TOKENS: frozenset[str] = frozenset({
    "yes", "y", "네", "예", "응", "ㅇㅇ", "있음", "있어요", "주제있음",
})

TOPIC_PRESENCE_NEGATIVE_TOKENS: frozenset[str] = frozenset({
    "no", "n", "노", "아니오", "아니요", "ㄴㄴ", "없음", "없어요", "주제없음",
})


def _normalize_button_token(value: object) -> str:
    lowered = str(value or "").strip().lower()
    return _BUTTON_NORMALIZE_PATTERN.sub("", lowered)


def _matches_topic_presence_button_message(message: object) -> bool:
    normalized = _normalize_button_token(message)
    return normalized in TOPIC_PRESENCE_POSITIVE_TOKENS | TOPIC_PRESENCE_NEGATIVE_TOKENS


def _is_topic_presence_negative_message(message: object) -> bool:
    normalized = _normalize_button_token(message)
    if normalized in TOPIC_PRESENCE_NEGATIVE_TOKENS:
        return True
    # 단순 토큰이 아닌 문장형 부정 표현도 처리 ("아니오, 아직 주제가 없습니다" 등)
    return bool(_NEGATIVE_SENTENCE_PATTERN.match(str(message or "").strip()))


def _matches_initial_button_message(action: str, message: object) -> bool:
    normalized = _normalize_button_token(message)
    return not normalized or normalized in INITIAL_BUTTON_TOKENS
