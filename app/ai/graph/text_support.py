import re

from app.core.config import settings

MATES_MENTION_PATTERN = re.compile(r"@mates\b", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
AI_RESPONSE_MAX_CHARS = max(80, int(settings.AI_RESPONSE_MAX_CHARS))

PLAIN_LANGUAGE_RULES = """
- 참고 레퍼런스의 전문용어를 그대로 복붙하지 말고, 초보 팀도 이해할 수 있는 쉬운 한국어로 풀어서 설명하세요.
- 꼭 필요한 전문용어를 써야 하면 한 번만 쓰고, 바로 뒤에 괄호나 짧은 설명으로 뜻을 덧붙이세요.
- 문장은 짧고 분명하게 쓰고, 현업자가 아닌 사람도 바로 이해할 수 있는 표현을 우선하세요.
""".strip()


def clean_text(value: object) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def strip_mates_mention(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    stripped = MATES_MENTION_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", stripped).strip()


def truncate_message(message: str, max_chars: int = AI_RESPONSE_MAX_CHARS) -> str:
    text = str(message or "").strip()
    if len(text) <= max_chars:
        return text
    truncated = text[: max_chars - 1].rstrip()
    if not truncated:
        return text[:max_chars]
    if truncated[-1] in {".", "!", "?", "…"}:
        return truncated
    return f"{truncated}…"
