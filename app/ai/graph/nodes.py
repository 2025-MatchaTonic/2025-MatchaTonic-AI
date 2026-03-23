# app/ai/graph/nodes.py
import json
import logging
import re
from time import perf_counter
from json import JSONDecodeError

from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from app.ai.schemas.llm_outputs import GatherLLMResponse
from app.ai.graph.collected_data import (
    build_collected_data_json_example,
    merge_collected_data,
)
from app.ai.graph.state import AgentState, TurnPolicy
from app.core.config import settings
from app.rag.retriever import get_rag_context

# NOTE
# The `collected_data` structure was recently extended with a new key
# "roles" (담당 역할).  All of the helper functions imported from
# `collected_data.py` are dynamic, so new fields are included
# automatically in prompts, examples and merging logic.  We also add
# some extra safety checks below so that nodes remain robust as the
# schema evolves.

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

RAG_EMPTY_CONTEXT = "(관련 레퍼런스를 찾지 못했습니다.)"
FAST_EXPLORE_REPLY = "좋아요. 최근 일주일 동안 '이거 좀 불편하다' 싶었던 순간이 있었나요?"
SHORT_MESSAGE_PATTERN = re.compile(r"[^a-z0-9가-힣]+")
MATES_MENTION_PATTERN = re.compile(r"@mates\b", re.IGNORECASE)
FAST_TOPIC_EXISTS_REPLY = (
    "좋아요. 이미 생각해둔 주제가 있다면 지금 한두 줄로 보내주세요. "
    "입력해주신 내용을 바탕으로 프로젝트 주제를 짧게 정리해서 collected data에 반영하겠습니다."
)
BUTTON_ONLY_PATTERN = re.compile(r"[\s\.\,\!\?]+")
TRAILING_TOPIC_ENDINGS_PATTERN = re.compile(
    r"(이에요|예요|입니다|이요|요|입니다요|하고 싶어요|하려고 해요|생각 중이에요|같아요)$"
)
QUESTION_LINE_ENDING_PATTERN = re.compile(
    r"(\?|？)\s*$|"
    r"(인가요|있나요|없나요|어떤가요|뭔가요|무엇인가요|왜인가요|어떨까요|할까요|볼까요|나요|까요)\s*$"
)
TRIVIAL_MESSAGES = {
    "",
    "hi",
    "hello",
    "hey",
    "yo",
    "안녕",
    "안녕하세요",
    "하이",
    "헬로",
    "ㅎㅇ",
    "ㅁㅌ",
    "mates",
}
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
GATHER_FIELD_GUIDE = {
    "title": {
        "label": "프로젝트 제목",
        "question": "프로젝트 제목을 한 줄로 어떻게 정리하면 될까요?",
    },
    "goal": {
        "label": "프로젝트 목표",
        "question": "이 프로젝트로 팀이 최종적으로 만들고 싶은 결과를 한 줄로 말하면 무엇인가요?",
    },
    "teamSize": {
        "label": "팀 인원",
        "question": "현재 이 프로젝트를 함께하는 팀원은 몇 명인가요?",
    },
    "roles": {
        "label": "역할",
        "question": "팀원 역할 분담은 어떻게 가져갈 생각인가요? 아직 미정이면 필요한 역할만 말해도 됩니다.",
    },
    "dueDate": {
        "label": "마감일",
        "question": "중간발표나 최종제출 기준으로 생각하는 마감일은 언제인가요?",
    },
    "deliverables": {
        "label": "산출물",
        "question": "최종적으로 제출하거나 보여줘야 하는 산출물은 무엇인가요?",
    },
}
UNSUPPORTED_GATHER_TOPICS = {
    "targetUser": {
        "label": "혜택 대상",
        "instruction": (
            "최근 대화의 초점은 혜택 대상입니다. 사용자의 답변이나 질문에 먼저 반응하되 "
            "이 정보는 collected_data의 기존 키로 억지로 저장하지 마세요."
        ),
    },
    "importance": {
        "label": "중요한 이유",
        "instruction": (
            "최근 대화의 초점은 문제의 중요성입니다. 먼저 이유를 설명하거나 정리해 주고, "
            "이 정보는 collected_data의 기존 키로 억지로 저장하지 마세요."
        ),
    },
}
GATHER_FOCUS_KEYWORDS = {
    "title": ("제목", "프로젝트명", "주제"),
    "goal": ("목표", "무엇을 만들", "최종적으로 만들", "해결하려는 문제"),
    "teamSize": ("몇 명", "팀원", "인원", "팀 규모"),
    "roles": ("역할", "역할 분담", "담당", "누가 맡"),
    "dueDate": ("마감", "마감일", "언제까지", "제출", "발표", "데드라인"),
    "deliverables": ("산출물", "결과물", "제출물", "무엇을 제출", "최종 산출"),
    "targetUser": ("누가 혜택", "대상 사용자", "누가 쓰", "누구를 위한", "누가 받"),
    "importance": ("왜 중요", "왜 필요한", "이유", "왜 문제", "중요한가"),
}
HELP_REQUEST_KEYWORDS = (
    "추천",
    "예시",
    "후보",
    "뭐가 좋",
    "어떻게",
    "왜",
    "이유",
    "설명",
    "알려",
    "모르겠",
    "도와",
)
TITLE_INSTRUCTION_KEYWORDS = (
    "넣어",
    "추천",
    "알려",
    "해줘",
    "보여",
    "데이터",
    "후보",
)
TITLE_EXPLICIT_PATTERN = re.compile(
    r"^\s*(?:프로젝트\s*주제|프로젝트명|주제|제목)\s*(?:은|는|:)?\s*",
    re.IGNORECASE,
)
TEAM_SIZE_PATTERN = re.compile(r"(?:팀\s*인원|팀원|인원)\D{0,8}(\d{1,2})\s*명?")
ROLE_PATTERN = re.compile(r"(?:역할|롤|role)\s*(?:은|는|:)?\s*(.+)$", re.IGNORECASE)
DUE_DATE_PATTERN = re.compile(
    r"(?:마감(?:일)?|데드라인|due)\s*(?:은|는|:)?\s*"
    r"([0-9]{4}[./-][0-9]{1,2}[./-][0-9]{1,2}|[0-9]{1,2}월\s*[0-9]{1,2}일)",
    re.IGNORECASE,
)
GOAL_PATTERN = re.compile(r"(?:목표|goal)\s*(?:은|는|:)?\s*(.+)$", re.IGNORECASE)
DELIVERABLES_PATTERN = re.compile(
    r"(?:산출물|결과물|제출물|deliverable[s]?)\s*(?:은|는|:)?\s*(.+)$",
    re.IGNORECASE,
)
CHOICE_INDEX_PATTERN = re.compile(r"^\s*([1-9])\s*번\s*$")
NUMBERED_OPTION_LINE_PATTERN = re.compile(r"^\s*(\d{1,2})[)\.\-:]\s*(.+?)\s*$")
NUMBERED_OPTION_INLINE_PATTERN = re.compile(
    r"(\d{1,2})[)\.\-:]\s*(.+?)(?=\s+\d{1,2}[)\.\-:]|$)"
)
DIRECT_FACT_ENDING_PATTERN = re.compile(
    r"\s*(?:입니다|이에요|예요|이야|야|요)\s*$"
)
AI_RESPONSE_MAX_CHARS = max(80, int(settings.AI_RESPONSE_MAX_CHARS))
FAST_RAG_PHASES = {"EXPLORE", "TOPIC_SET", "GATHER"}
RAG_CACHE_MAX_ITEMS = 128
RAG_CONTEXT_CACHE: dict[tuple[str, str, tuple[str, ...], tuple[str, ...], int], str] = {}


PLAIN_LANGUAGE_RULES = """
- 참고 레퍼런스의 전문용어를 그대로 복붙하지 말고, 초보 팀도 이해할 수 있는 쉬운 한국어로 풀어서 설명하세요.
- 꼭 필요한 전문용어를 써야 하면 한 번만 쓰고, 바로 뒤에 괄호나 짧은 설명으로 뜻을 덧붙이세요.
- 문장은 짧고 분명하게 쓰고, 현업자가 아닌 사람도 바로 이해할 수 있는 표현을 우선하세요.
""".strip()

RAG_FILTERS_BY_PHASE: dict[str, dict[str, list[str]]] = {
    "EXPLORE": {
        "topics": ["value_proposition", "team_playbook"],
        "doc_types": ["reference", "playbook"],
    },
    "TOPIC_SET": {
        "topics": ["design_sprint", "team_playbook", "scrum_guide"],
        "doc_types": ["reference", "playbook", "guide"],
    },
    "GATHER": {
        "topics": [
            "value_proposition",
            "design_sprint",
            "team_playbook",
            "scrum_guide",
        ],
        "doc_types": ["reference", "playbook", "guide"],
    },
    "READY_PLAN": {
        "topics": ["scrum_guide", "team_playbook", "design_sprint"],
        "doc_types": ["guide", "playbook", "reference"],
    },
    "READY_DEV": {
        "topics": ["api_design", "software_engineering_standard", "scrum_guide"],
        "doc_types": ["reference", "guide"],
    },
}


def _normalize_message(value: str) -> str:
    lowered = MATES_MENTION_PATTERN.sub(" ", str(value or "").strip().lower())
    compact = SHORT_MESSAGE_PATTERN.sub("", lowered)
    return compact.strip()


def _strip_mates_mention(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    stripped = MATES_MENTION_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", stripped).strip()


def _effective_user_message(state: AgentState) -> str:
    return _strip_mates_mention(state.get("user_message"))


def _is_trivial_message(user_message: str) -> bool:
    normalized = _normalize_message(user_message)
    return normalized in TRIVIAL_MESSAGES


def _normalize_button_token(value: object) -> str:
    lowered = str(value or "").strip().lower()
    return BUTTON_ONLY_PATTERN.sub("", lowered)


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


def _is_initial_button_selection(state: AgentState) -> bool:
    action = state.get("action_type")
    if action not in {"BTN_NO", "BTN_YES", "BTN_GO_DEF"}:
        return False

    message_candidate = _clean_text(state.get("user_message")) or _clean_text(
        state.get("selected_message")
    )
    if not message_candidate:
        return True

    return _matches_initial_button_message(action, message_candidate)


def _get_turn_policy(state: AgentState) -> TurnPolicy:
    return state.get("turn_policy", "ANSWER_THEN_ASK")


def _is_answer_only_turn(state: AgentState) -> bool:
    return _get_turn_policy(state) == "ANSWER_ONLY"


def _is_capture_title_turn(state: AgentState) -> bool:
    return _get_turn_policy(state) == "CAPTURE_TITLE"


def _extract_topic_candidate(user_message: str) -> str:
    text = _clean_text(user_message)
    if not text or "?" in text:
        return ""

    patterns = (
        r"^(?:프로젝트\s*주제|주제|아이디어)\s*(?:는|은|으론|로는|은요|는요)?\s*(.+)$",
        r"^(?:저희|우리는|우리 팀은|저는)\s+(.+)$",
    )
    candidate = text
    for pattern in patterns:
        match = re.match(pattern, candidate, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            break

    cleanup_patterns = (
        r"\s*(?:을|를)?\s*주제로\s*(?:할게요|하려고 해요|하고 싶어요)$",
        r"\s*(?:을|를)?\s*만들고\s*싶어요$",
        r"\s*(?:을|를)?\s*생각하고\s*있어요$",
        r"\s*(?:으로|로)\s*하려고\s*해요$",
    )
    for pattern in cleanup_patterns:
        candidate = re.sub(pattern, "", candidate, flags=re.IGNORECASE).strip()

    candidate = TRAILING_TOPIC_ENDINGS_PATTERN.sub("", candidate).strip(" .,!?:;\"'")
    if not candidate:
        return ""

    if len(candidate) > 60:
        return ""

    return candidate


def _normalize_topic_title(candidate: str) -> str:
    title = _clean_text(candidate)
    if not title:
        return ""

    title = re.sub(
        r"^\s*(?:프로젝트\s*주제|프로젝트명|주제|제목|아이디어)\s*(?:은|는|:)?\s*",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(
        r"^\s*(?:저희(?:는)?|우리는|우리\s*팀은|저는|이번\s*프로젝트(?:는)?|이\s*프로젝트(?:는)?)\s*",
        "",
        title,
        flags=re.IGNORECASE,
    )

    trailing_patterns = (
        r"\s*(?:주제|아이디어|프로젝트)\s*$",
        r"\s*(?:같은\s*(?:거|것)|같은\s*느낌(?:의)?\s*(?:거|것)?)\s*$",
        r"\s*(?:으로|로)?\s*(?:정했어요|하려고\s*해요|하고\s*싶어요|해보려고\s*해요|생각하고\s*있어요)\s*$",
        r"\s*(?:느낌이에요|느낌입니다|같아요|정도예요|정도입니다)\s*$",
    )
    for pattern in trailing_patterns:
        title = re.sub(pattern, "", title, flags=re.IGNORECASE).strip()

    for separator in (
        r"\s*[,\n]\s*",
        r"\s+(?:그리고|근데|다만|인데|이지만|이라서|해서|라서)\s+",
    ):
        parts = re.split(separator, title, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2 and len(parts[0].strip()) >= 6:
            title = parts[0].strip()
            break

    title = re.sub(r"\s+", " ", title).strip(" .,!?:;\"'")
    if not title:
        return ""
    if len(title) > 40:
        return ""
    if _looks_like_title_instruction(title):
        return ""
    return title


def _seed_topic_title(state: AgentState, current_data: dict[str, str]) -> dict[str, str]:
    if state.get("current_phase") != "TOPIC_SET":
        return {}
    if _is_meaningful_fact(current_data.get("title")):
        return {}

    candidate = _normalize_topic_title(
        _extract_topic_candidate(_effective_user_message(state))
    )
    if not candidate:
        return {}

    return {"title": candidate}


def _looks_like_question_line(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    candidate = cleaned.lstrip("-*0123456789. ").strip()
    return bool(QUESTION_LINE_ENDING_PATTERN.search(candidate))


def _trim_trailing_question_lines(message: str) -> str:
    lines = [line.rstrip() for line in str(message or "").splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and _looks_like_question_line(lines[-1]):
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
    return "\n".join(lines).strip()


def _truncate_ai_message(message: str, max_chars: int = AI_RESPONSE_MAX_CHARS) -> str:
    text = str(message or "").strip()
    if len(text) <= max_chars:
        return text
    truncated = text[: max_chars - 1].rstrip()
    if not truncated:
        return text[:max_chars]
    if truncated[-1] in {".", "!", "?", "…"}:
        return truncated
    return f"{truncated}…"


def _answer_only_fallback(state: AgentState, message: str) -> str:
    if str(message or "").strip():
        return str(message).strip()

    phase = state.get("current_phase", "GATHER")
    if phase == "EXPLORE":
        return "지금까지 나온 이야기만 보면 먼저 팀이 실제로 자주 겪는 불편을 한 가지로 좁혀보는 게 좋겠습니다."
    if phase == "TOPIC_SET":
        return "지금까지 나온 내용을 기준으로 이 주제가 풀고 싶은 핵심 문제를 한 문장으로 정리해보는 게 좋겠습니다."
    return "지금까지 나온 내용을 기준으로 핵심 방향부터 짧게 정리하고 다음 판단으로 넘어가는 게 좋겠습니다."


def _apply_turn_policy_to_message(state: AgentState, message: str) -> str:
    cleaned = str(message or "").strip()
    if not cleaned:
        return _truncate_ai_message(_answer_only_fallback(state, cleaned))

    if _is_answer_only_turn(state):
        trimmed = _trim_trailing_question_lines(cleaned)
        return _truncate_ai_message(_answer_only_fallback(state, trimmed))

    return _truncate_ai_message(cleaned)


def _should_skip_rag(state: AgentState) -> bool:
    user_message = _effective_user_message(state)
    selected_message = _strip_mates_mention(state.get("selected_message"))
    recent_messages = [msg for msg in state.get("recent_messages", []) if str(msg).strip()]
    return _is_trivial_message(user_message) and not selected_message.strip() and not recent_messages


def _is_topic_not_set(state: AgentState) -> bool:
    current_data = _prune_collected_data(state.get("collected_data") or {})
    return not _is_meaningful_fact(current_data.get("title"))


def _should_use_rag(state: AgentState, phase: str, query: str) -> bool:
    if _should_skip_rag(state):
        return False

    action = state.get("action_type")
    if action in {"BTN_NO", "BTN_YES", "BTN_GO_DEF"}:
        return False

    if phase == "TOPIC_SET" and _is_topic_not_set(state):
        return False

    query_text = str(query or "").strip()
    if not query_text:
        return False
    if phase not in FAST_RAG_PHASES:
        return True

    has_question = "?" in query_text or "？" in query_text
    asks_help = _is_help_request(query_text)
    if not has_question and not asks_help and len(query_text) < max(1, settings.RAG_QUERY_MIN_CHARS):
        return False

    if _is_answer_only_turn(state) and not has_question and not asks_help and len(query_text) < 24:
        return False

    return True


def _select_rag_top_k(state: AgentState, phase: str, query: str) -> int:
    base_k = max(1, int(settings.RAG_TOP_K))
    if phase not in FAST_RAG_PHASES:
        return base_k

    fast_k = min(base_k, max(1, int(settings.RAG_CHAT_TOP_K)))
    if "?" in query or "？" in query or _is_help_request(query):
        return min(base_k, fast_k + 1)
    return fast_k


def _trim_rag_context_for_phase(context: str, phase: str) -> str:
    text = str(context or "").strip()
    if not text:
        return RAG_EMPTY_CONTEXT
    if phase not in FAST_RAG_PHASES:
        return text

    max_chars = max(200, int(settings.RAG_CHAT_MAX_CONTEXT_CHARS))
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _invoke_llm(llm: ChatOpenAI, prompt: str, *, label: str, **kwargs):
    started_at = perf_counter()
    try:
        response = llm.invoke(prompt, **kwargs)
    except Exception:
        logger.exception("%s failed in %.2fs", label, perf_counter() - started_at)
        return None
    logger.info("%s completed in %.2fs", label, perf_counter() - started_at)
    return response


def _build_rag_query(state: AgentState) -> str:
    user_message = _effective_user_message(state)
    selected = _strip_mates_mention(state.get("selected_message"))
    recent = [
        _strip_mates_mention(msg)
        for msg in state.get("recent_messages", [])
        if _strip_mates_mention(msg)
    ]

    parts = [user_message]
    if selected:
        parts.append(selected)
    if recent:
        parts.append(" | ".join(recent[-5:]))
    return "\n".join(part for part in parts if part).strip()


def _fetch_rag_context(
    state: AgentState,
    phase: str,
    *,
    topics: list[str] | None = None,
    doc_types: list[str] | None = None,
) -> str:
    query = _build_rag_query(state)
    if not _should_use_rag(state, phase, query):
        logger.info(
            "rag skipped phase=%s action=%s query_chars=%d",
            phase,
            state.get("action_type"),
            len(query or ""),
        )
        return RAG_EMPTY_CONTEXT

    top_k = _select_rag_top_k(state, phase, query)
    topics_key = tuple((topics or []))
    doc_types_key = tuple((doc_types or []))
    cache_key = (phase, query, topics_key, doc_types_key, top_k)
    cached_context = RAG_CONTEXT_CACHE.get(cache_key)
    if cached_context:
        logger.info(
            "rag cache hit phase=%s query_chars=%d context_chars=%d",
            phase,
            len(query),
            len(cached_context),
        )
        return cached_context

    started_at = perf_counter()
    context = get_rag_context(
        query=query,
        current_phase=phase,
        k=top_k,
        topics=topics,
        doc_types=doc_types,
    )
    context = _trim_rag_context_for_phase(context, phase)
    RAG_CONTEXT_CACHE[cache_key] = context or RAG_EMPTY_CONTEXT
    if len(RAG_CONTEXT_CACHE) > RAG_CACHE_MAX_ITEMS:
        oldest_key = next(iter(RAG_CONTEXT_CACHE))
        RAG_CONTEXT_CACHE.pop(oldest_key, None)

    logger.info(
        "rag fetched in %.2fs phase=%s query_chars=%d context_chars=%d top_k=%d",
        perf_counter() - started_at,
        phase,
        len(query),
        len(context or ""),
        top_k,
    )
    return context or RAG_EMPTY_CONTEXT


def _get_rag_filters(filter_key: str) -> dict[str, list[str]]:
    return RAG_FILTERS_BY_PHASE.get(filter_key, {})


def _clean_text(value: object) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _is_meaningful_fact(value: object) -> bool:
    cleaned = _clean_text(value)
    if not cleaned:
        return False
    normalized = cleaned.lower()
    if "@mates" in normalized or "?" in cleaned:
        return False
    negative_fact_keywords = (
        "모르겠",
        "모름",
        "없음",
        "없어요",
        "미정",
        "tbd",
        "unknown",
        "not sure",
        "idk",
        "don't know",
        "dont know",
        "no idea",
    )
    return not any(keyword in normalized for keyword in negative_fact_keywords)


def _prune_collected_data(data: dict[str, str]) -> dict[str, str]:
    return {
        key: _clean_text(value)
        for key, value in (data or {}).items()
        if _is_meaningful_fact(value)
    }


def _detect_gather_focus(text: str) -> str | None:
    normalized = _clean_text(text)
    if not normalized:
        return None

    for focus, keywords in GATHER_FOCUS_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return focus
    return None


def _infer_conversation_focus(state: AgentState) -> str | None:
    candidates: list[str] = []
    selected_message = _clean_text(_strip_mates_mention(state.get("selected_message")))
    recent_messages = [
        _clean_text(_strip_mates_mention(msg))
        for msg in state.get("recent_messages", [])
        if _clean_text(_strip_mates_mention(msg))
    ]

    if selected_message:
        candidates.append(selected_message)
    candidates.extend(reversed(recent_messages[-4:]))

    for candidate in candidates:
        focus = _detect_gather_focus(candidate)
        if focus:
            return focus
    return None


def _build_missing_field_summary(current_data: dict[str, str]) -> str:
    missing_fields = [
        f'- {key}: {meta["label"]}'
        for key, meta in GATHER_FIELD_GUIDE.items()
        if not _clean_text(current_data.get(key))
    ]
    return "\n".join(missing_fields) if missing_fields else "- 없음"


def _build_gather_focus_instruction(focus_type: str | None) -> str:
    if focus_type in GATHER_FIELD_GUIDE:
        focus_info = GATHER_FIELD_GUIDE[focus_type]
        return (
            f'최근 대화의 초점은 "{focus_info["label"]}" 입니다. '
            "우선 이 항목을 중심으로 답하되, 사용자가 다른 필드의 확정 사실을 함께 말하면 "
            "그 사실도 updated_data에 반영하세요. "
            f'답이 부족하면 "{focus_info["question"]}"처럼 같은 흐름에서 한 가지 질문만 이어가세요.'
        )

    if focus_type in UNSUPPORTED_GATHER_TOPICS:
        return UNSUPPORTED_GATHER_TOPICS[focus_type]["instruction"]

    return (
        "고정된 질문 순서를 따르지 말고, 최근 대화 흐름에서 가장 자연스러운 주제를 이어가세요. "
        "지금 맥락과 맞는 비어 있는 정보 하나만 골라 한 가지 질문으로 이어가세요."
    )


def _is_help_request(user_message: str) -> bool:
    normalized = _clean_text(user_message)
    return any(keyword in normalized for keyword in HELP_REQUEST_KEYWORDS)


def _looks_like_title_instruction(candidate: str) -> bool:
    normalized = str(candidate or "").strip().lower()
    if not normalized:
        return True
    return any(keyword in normalized for keyword in TITLE_INSTRUCTION_KEYWORDS)


def _normalize_direct_fact_value(value: str) -> str:
    normalized = DIRECT_FACT_ENDING_PATTERN.sub("", str(value or "").strip())
    return normalized.strip(" .,!?:;\"'")


def _extract_direct_fact_updates(user_message: str) -> dict[str, str]:
    message = _clean_text(_strip_mates_mention(user_message))
    if not message:
        return {}

    updates: dict[str, str] = {}

    if TITLE_EXPLICIT_PATTERN.match(message):
        topic_candidate = _normalize_topic_title(_extract_topic_candidate(message))
        if topic_candidate and not _looks_like_title_instruction(topic_candidate):
            updates["title"] = topic_candidate

    team_size_match = TEAM_SIZE_PATTERN.search(message)
    if team_size_match:
        updates["teamSize"] = team_size_match.group(1).strip()

    role_match = ROLE_PATTERN.search(message)
    if role_match:
        updates["roles"] = _normalize_direct_fact_value(role_match.group(1))

    due_date_match = DUE_DATE_PATTERN.search(message)
    if due_date_match:
        updates["dueDate"] = (
            due_date_match.group(1).strip().replace("/", "-").replace(".", "-")
        )

    goal_match = GOAL_PATTERN.search(message)
    if goal_match:
        updates["goal"] = _normalize_direct_fact_value(goal_match.group(1))

    deliverables_match = DELIVERABLES_PATTERN.search(message)
    if deliverables_match:
        updates["deliverables"] = _normalize_direct_fact_value(deliverables_match.group(1))

    return _sanitize_gather_updates(updates)


def _extract_choice_based_title(state: AgentState) -> str:
    current_message = _effective_user_message(state)
    choice_match = CHOICE_INDEX_PATTERN.match(current_message)
    if not choice_match:
        return ""

    choice_index = int(choice_match.group(1))
    text_blocks: list[str] = []

    selected_raw = MATES_MENTION_PATTERN.sub(" ", str(state.get("selected_message") or ""))
    selected = _clean_text(selected_raw)
    if selected:
        text_blocks.append(selected)

    recent_messages = [
        _clean_text(MATES_MENTION_PATTERN.sub(" ", str(msg or "")))
        for msg in state.get("recent_messages", [])
        if _clean_text(MATES_MENTION_PATTERN.sub(" ", str(msg or "")))
    ]
    text_blocks.extend(reversed(recent_messages[-6:]))

    for block in text_blocks:
        matched_line_option = False
        for line in block.splitlines():
            line_match = NUMBERED_OPTION_LINE_PATTERN.match(line.strip())
            if not line_match:
                continue
            matched_line_option = True
            if int(line_match.group(1)) != choice_index:
                continue

            candidate = _normalize_topic_title(line_match.group(2))
            if candidate and not _looks_like_title_instruction(candidate):
                return candidate

        if matched_line_option:
            continue

        for inline_match in NUMBERED_OPTION_INLINE_PATTERN.finditer(block):
            if int(inline_match.group(1)) != choice_index:
                continue

            candidate = _normalize_topic_title(inline_match.group(2))
            if candidate and not _looks_like_title_instruction(candidate):
                return candidate

    return ""


def _sanitize_gather_updates(updated_data: dict[str, str]) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    for key, value in updated_data.items():
        if key == "title":
            value = _normalize_topic_title(value)
        if _is_meaningful_fact(value):
            sanitized[key] = _clean_text(value)
    return sanitized


def _filter_gather_updates(
    user_message: str,
    updated_data: dict[str, str],
    *,
    focus_type: str | None,
) -> dict[str, str]:
    if _is_help_request(user_message):
        return {}

    sanitized = _sanitize_gather_updates(updated_data)
    if focus_type in UNSUPPORTED_GATHER_TOPICS:
        return {}
    return sanitized


def _count_ready_fields(current_data: dict[str, str]) -> int:
    return sum(1 for key in GATHER_FIELD_GUIDE if _is_meaningful_fact(current_data.get(key)))


def _is_template_ready(current_data: dict[str, str]) -> bool:
    normalized = _prune_collected_data(current_data)
    if not _is_meaningful_fact(normalized.get("title")):
        return False
    if not (
        _is_meaningful_fact(normalized.get("goal"))
        or _is_meaningful_fact(normalized.get("deliverables"))
    ):
        return False
    return _count_ready_fields(normalized) >= 4


def _build_project_snapshot(data: dict) -> dict[str, str]:
    title = _clean_text(data.get("title")) or "프로젝트 제목 미정"
    goal = _clean_text(data.get("goal")) or "프로젝트 목표는 추가 논의가 필요합니다."
    team_size = _clean_text(data.get("teamSize")) or "팀 규모 미정"
    roles = _clean_text(data.get("roles")) or "역할 분담 추가 논의 필요"
    due_date = _clean_text(data.get("dueDate")) or "마감 일정 미정"
    deliverables = _clean_text(data.get("deliverables")) or "산출물 범위 추가 논의 필요"

    return {
        "title": title,
        "goal": goal,
        "team_size": team_size,
        "roles": roles,
        "due_date": due_date,
        "deliverables": deliverables,
    }


def _build_default_template_sections(state: AgentState, mode: str = "plan") -> dict:
    snapshot = _build_project_snapshot(state.get("collected_data") or {})
    title = snapshot["title"]
    goal = snapshot["goal"]
    team_size = snapshot["team_size"]
    roles = snapshot["roles"]
    due_date = snapshot["due_date"]
    deliverables = snapshot["deliverables"]

    if mode == "dev":
        return {
            "project_home": {
                "project_overview": (
                    f"{title} 프로젝트의 개발 실행 초안입니다. 목표는 {goal} "
                    f"현재 기준 팀 규모는 {team_size}, 역할은 {roles}, "
                    f"주요 산출물은 {deliverables}, 마감 기준은 {due_date}입니다."
                ),
            },
            "planning": {
                "project_intro": f"{title} 개발 실행 초안: {goal}",
                "problem_definition": [
                    {
                        "id": 1,
                        "situation": (
                            "개발 범위, 책임 분담, 산출물 연결 방식이 한 문서에 정리되지 않으면 "
                            "구현 우선순위와 핸드오프가 흔들릴 수 있습니다."
                        ),
                        "reason": (
                            f"팀 규모 {team_size}에서 역할({roles})과 마감({due_date})이 있는 만큼 "
                            "실행 기준을 미리 고정해야 개발 속도와 품질을 맞출 수 있습니다."
                        ),
                        "limitation": (
                            "현재 수집된 정보만으로는 세부 기술 스택, API 명세, 데이터 모델, "
                            "세부 태스크 단위까지는 확정할 수 없습니다."
                        ),
                    }
                ],
                "solution": {
                    "core_summary": (
                        "목표, 역할, 산출물, 마감 기준을 바탕으로 개발 범위와 실행 기준을 단일 문서로 정리합니다."
                    ),
                    "problem_solutions": [
                        {
                            "problem_id": 1,
                            "solution_desc": (
                                f"역할은 {roles}, 산출물은 {deliverables}, 마감은 {due_date} 기준으로 "
                                "개발 범위와 책임을 정리하고 구현 전에 공통 기준을 맞춥니다."
                            ),
                        }
                    ],
                    "features": [
                        f"개발 목표 정리: {goal}",
                        f"역할 및 책임 기준: {roles} / {team_size}",
                        f"구현 산출물 및 일정 기준: {deliverables} / {due_date}",
                    ],
                },
                "target_persona": {
                    "name": "추가 논의 필요",
                    "age": "추가 논의 필요",
                    "job_role": "개발 실행 담당 팀",
                    "main_activities": deliverables,
                    "pain_points": [
                        "구현 범위와 책임 분담이 문서로 정리되지 않으면 실행 속도가 떨어집니다.",
                        "산출물 기준이 분산되면 개발과 검토 과정에서 반복 조율이 발생합니다.",
                    ],
                    "needs": [
                        f"{due_date}까지 바로 실행 가능한 개발 기준 문서가 필요합니다.",
                        "역할, 산출물, 우선순위를 한 번에 볼 수 있는 정리된 실행 초안이 필요합니다.",
                    ],
                },
            },
            "ground_rules": (
                "1. 개발 범위와 우선순위는 수집된 목표, 역할, 산출물, 마감 기준을 우선 근거로 삼습니다.\n"
                "2. 역할별 책임과 산출물 연결 관계를 문서에 명시한 뒤 작업을 시작합니다.\n"
                "3. 세부 기술 선택과 구현 방식은 팀 합의 전까지 확정 표현으로 쓰지 않습니다.\n"
                "4. 마감에 영향을 주는 변경 사항은 문서에 즉시 반영하고 공유합니다."
            ),
        }

    return {
        "project_home": {
            "project_overview": (
                f"{title} 프로젝트입니다. 목표는 {goal} "
                f"현재 기준 팀 규모는 {team_size}, 주요 산출물은 {deliverables}, "
                f"마감 기준은 {due_date}입니다."
            ),
        },
        "planning": {
            "project_intro": f"{title}: {goal}",
            "problem_definition": [
                {
                    "id": 1,
                    "situation": (
                        f"{title} 프로젝트는 목표가 정해져 있지만 실행 기준을 한 문서로 "
                        "정리할 필요가 있습니다."
                    ),
                    "reason": (
                        f"팀 규모 {team_size} 기준으로 역할, 산출물, 일정이 분산되면 "
                        "실행 우선순위가 흔들릴 수 있습니다."
                    ),
                    "limitation": (
                        "현재 수집된 최소 정보만으로는 세부 사용자 시나리오와 세부 요구사항을 "
                        "확정하기 어렵습니다."
                    ),
                }
            ],
            "solution": {
                "core_summary": (
                    "목표, 역할, 산출물, 일정 기준을 하나의 템플릿에 정리해 팀의 실행 기준을 맞춥니다."
                ),
                "problem_solutions": [
                    {
                        "problem_id": 1,
                        "solution_desc": (
                            f"{goal}에 맞춰 역할은 {roles}, 산출물은 {deliverables}, "
                            f"마감은 {due_date} 기준으로 정리합니다."
                        ),
                    }
                ],
                "features": [
                    f"프로젝트 목표 정리: {goal}",
                    f"팀 운영 기준 정리: {roles} / {team_size}",
                    f"산출물 및 일정 기준 정리: {deliverables} / {due_date}",
                ],
            },
            "target_persona": {
                "name": "추가 논의 필요",
                "age": "추가 논의 필요",
                "job_role": roles,
                "main_activities": deliverables,
                "pain_points": [
                    "목표는 정해졌지만 세부 실행 기준과 우선순위가 문서로 정리되지 않았습니다.",
                    f"팀 규모 {team_size}에서 역할 경계와 일정 기준을 맞춰야 합니다.",
                ],
                "needs": [
                    f"{due_date}까지 공유 가능한 실행 템플릿이 필요합니다.",
                    "산출물과 역할을 한 번에 확인할 수 있는 기획 문서가 필요합니다.",
                ],
            },
        },
        "ground_rules": (
            "1. 프로젝트 목표, 역할, 산출물, 일정은 수집된 데이터를 기준으로 정리합니다.\n"
            "2. 최근 팀 대화는 문서의 톤과 우선순위를 보완하는 참고 문맥으로 활용합니다.\n"
            "3. 역할, 산출물, 일정은 팀이 합의한 표현으로 명시합니다.\n"
            "4. 목표와 직접 연결되는 실행 기준부터 먼저 정리합니다."
        ),
    }


def _get_template_mode_config(action_type: str) -> dict[str, str]:
    if action_type == "BTN_DEV":
        return {
            "mode": "dev",
            "mode_label": "개발용 템플릿",
            "focus": (
                "문제 정의보다 개발 범위, 구현 기준, 역할 분담, 산출물 연결, "
                "실행 우선순위가 더 잘 드러나도록 작성하세요."
            ),
            "summary_fallback": "개발용 템플릿 초안을 생성했습니다.",
        }

    return {
        "mode": "plan",
        "mode_label": "기획용 템플릿",
        "focus": (
            "사용자 문제, 기획 의도, 산출물 방향, 팀 정렬 포인트가 더 잘 드러나도록 작성하세요."
        ),
        "summary_fallback": "기획용 템플릿 초안을 생성했습니다.",
    }


def _merge_template_sections(base: dict, generated: dict) -> dict:
    merged = dict(base)

    project_home = generated.get("project_home")
    if isinstance(project_home, dict):
        overview = str(project_home.get("project_overview", "")).strip()
        if overview:
            merged["project_home"] = {"project_overview": overview}

    planning = generated.get("planning")
    if isinstance(planning, dict):
        merged_planning = dict(base["planning"])

        project_intro = str(planning.get("project_intro", "")).strip()
        if project_intro:
            merged_planning["project_intro"] = project_intro

        problem_definition = planning.get("problem_definition")
        if isinstance(problem_definition, list) and problem_definition:
            cleaned_problem_definition = []
            for index, item in enumerate(problem_definition, start=1):
                if not isinstance(item, dict):
                    continue
                cleaned_problem_definition.append(
                    {
                        "id": item.get("id") or index,
                        "situation": str(item.get("situation", "")).strip(),
                        "reason": str(item.get("reason", "")).strip(),
                        "limitation": str(item.get("limitation", "")).strip(),
                    }
                )
            if cleaned_problem_definition:
                merged_planning["problem_definition"] = cleaned_problem_definition

        solution = planning.get("solution")
        if isinstance(solution, dict):
            merged_solution = dict(base["planning"]["solution"])

            core_summary = str(solution.get("core_summary", "")).strip()
            if core_summary:
                merged_solution["core_summary"] = core_summary

            problem_solutions = solution.get("problem_solutions")
            if isinstance(problem_solutions, list) and problem_solutions:
                cleaned_problem_solutions = []
                for item in problem_solutions:
                    if not isinstance(item, dict):
                        continue
                    solution_desc = str(item.get("solution_desc", "")).strip()
                    if not solution_desc:
                        continue
                    cleaned_problem_solutions.append(
                        {
                            "problem_id": item.get("problem_id") or 1,
                            "solution_desc": solution_desc,
                        }
                    )
                if cleaned_problem_solutions:
                    merged_solution["problem_solutions"] = cleaned_problem_solutions

            features = solution.get("features")
            if isinstance(features, list):
                cleaned_features = [
                    str(feature).strip() for feature in features if str(feature).strip()
                ]
                if cleaned_features:
                    merged_solution["features"] = cleaned_features

            merged_planning["solution"] = merged_solution

        target_persona = planning.get("target_persona")
        if isinstance(target_persona, dict):
            merged_persona = dict(base["planning"]["target_persona"])
            for key in ["name", "age", "job_role", "main_activities"]:
                value = str(target_persona.get(key, "")).strip()
                if value:
                    merged_persona[key] = value

            for key in ["pain_points", "needs"]:
                values = target_persona.get(key)
                if isinstance(values, list):
                    cleaned_values = [
                        str(value).strip() for value in values if str(value).strip()
                    ]
                    if cleaned_values:
                        merged_persona[key] = cleaned_values

            merged_planning["target_persona"] = merged_persona
        merged["planning"] = merged_planning

    ground_rules = generated.get("ground_rules")
    if isinstance(ground_rules, str) and ground_rules.strip():
        merged["ground_rules"] = ground_rules.strip()

    return merged


def _build_notion_template_payload(state: AgentState, sections: dict) -> dict:
    return {
        "projectId": int(state["project_id"]),
        "templates": [
            {
                "key": "PROJECT_HOME",
                "parentKey": None,
                "title": "Project Home",
                "content": sections["project_home"],
            },
            {
                "key": "PLANNING",
                "parentKey": "PROJECT_HOME",
                "title": "기획",
                "content": sections["planning"],
            },
            {
                "key": "GROUND_RULES",
                "parentKey": "PROJECT_HOME",
                "title": "그라운드룰",
                "content": sections["ground_rules"],
            },
        ],
    }


def _build_recent_context(state: AgentState) -> str:
    selected_message = _strip_mates_mention(state.get("selected_message"))
    recent_messages = [
        _strip_mates_mention(msg)
        for msg in state.get("recent_messages", [])
        if _strip_mates_mention(msg)
    ]

    context_blocks: list[str] = []
    if selected_message:
        context_blocks.append(f"[현재 집중 포인트]\n- {selected_message}")
    if recent_messages:
        latest_messages = recent_messages[-5:]
        flow_summary = "\n".join(f"- {message}" for message in latest_messages)
        context_blocks.append("[최근 흐름 요약]\n" + flow_summary)

    return "\n\n".join(context_blocks) if context_blocks else "(전달된 최근 대화 없음)"


def _build_template_input_summary(state: AgentState) -> str:
    snapshot = _build_project_snapshot(state.get("collected_data") or {})
    return (
        f"- 제목: {snapshot['title']}\n"
        f"- 목표: {snapshot['goal']}\n"
        f"- 팀 규모: {snapshot['team_size']}\n"
        f"- 역할: {snapshot['roles']}\n"
        f"- 마감일: {snapshot['due_date']}\n"
        f"- 산출물: {snapshot['deliverables']}"
    )


def _build_template_content_example() -> dict:
    return {
        "project_home": {
            "project_overview": "내용",
        },
        "planning": {
            "project_intro": "프로젝트 한 줄 소개",
            "problem_definition": [
                {
                    "id": 1,
                    "situation": "불편한 상황 (상황·경험 중심)",
                    "reason": "왜 문제인가?",
                    "limitation": "기존 해결 방식의 한계",
                }
            ],
            "solution": {
                "core_summary": "핵심 솔루션 한 줄 요약",
                "problem_solutions": [
                    {
                        "problem_id": 1,
                        "solution_desc": "문제 1에 대한 우리 서비스의 해결 방식",
                    }
                ],
                "features": [
                    "우리 솔루션의 특징 1",
                    "우리 솔루션의 특징 2",
                    "우리 솔루션의 특징 3",
                ],
            },
            "target_persona": {
                "name": "이름(가명)",
                "age": "나이",
                "job_role": "직업 / 역할",
                "main_activities": "주요 활동",
                "pain_points": [
                    "불편함 1",
                    "불편함 2",
                ],
                "needs": [
                    "니즈 1",
                    "니즈 2",
                ],
            },
        },
        "ground_rules": " ",
    }


def _build_topic_exists_fallback_message() -> str:
    return FAST_TOPIC_EXISTS_REPLY


# ----------------------------------------------------
# 1. 아이디어가 없을 때 (NO 선택) : 탐색 노드
def explore_problem_node(state: AgentState):
    user_message = _effective_user_message(state)
    turn_policy = _get_turn_policy(state)

    if _is_initial_button_selection(state):
        return {
            "ai_message": FAST_EXPLORE_REPLY,
            "next_phase": "EXPLORE",
        }

    if _is_trivial_message(user_message) and not state.get("recent_messages"):
        return {
            "ai_message": FAST_EXPLORE_REPLY,
            "next_phase": "EXPLORE",
        }

    rag_context = _fetch_rag_context(
        state,
        phase="EXPLORE",
        **_get_rag_filters("EXPLORE"),
    )
    recent_context = _build_recent_context(state)
    prompt = f"""
    당신은 친절하고 센스 있는 브레인스토밍 파트너입니다. 
    사용자가 아직 프로젝트 주제가 없거나, 불편함을 탐색하는 중입니다.

    [참고 레퍼런스]
    {rag_context}

    [최근 대화 문맥]
    {recent_context}

    [중요 지시사항]
    1. 사용자가 방금 말한 내용이나 최근 팀 대화에 담긴 질문/고민에 먼저 반응하고 답하세요.
    2. 질문은 **정말 다음 판단에 필요할 때만** 마지막에 1개까지 하세요. 답변만으로 흐름이 충분히 이어지면 질문 없이 끝내세요.
    3. 사용자가 대답을 했다면, 먼저 그 대답에 깊이 공감하거나 요약한 뒤에 필요한 경우에만 꼬리 질문을 1개 던지세요.
    4. 첫 시작이고 아직 문맥이 거의 없다면 이렇게 가볍게 물어보세요: "최근 일주일 동안 '아, 이거 진짜 귀찮다' 했던 적이 있나요?"
    5. 대화가 자연스럽게 이어지도록 친구처럼 편안한 말투를 사용하세요.
    6. 직접 사용자 입력이 비어 있더라도 최근 팀 대화에 unresolved point가 있으면 그 문맥을 이어서 답하세요.
    7. 아래 원칙을 반드시 지키세요.
    8. 최종 답변은 공백 포함 300자 이내로 작성하세요.
    {PLAIN_LANGUAGE_RULES}

    [현재 턴 정책]
    {turn_policy}
    - ANSWER_ONLY면 설명이나 정리로 답변을 끝내고, 질문으로 마무리하지 마세요.
    - ANSWER_THEN_ASK면 답변이 충분한 경우 질문 없이 끝내도 됩니다.
    
    [사용자 입력]
    {user_message}
    """

    response = _invoke_llm(conversation_llm, prompt, label="explore_problem_node llm")
    ai_message = (
        str(response.content)
        if response is not None and str(getattr(response, "content", "")).strip()
        else FAST_EXPLORE_REPLY
    )

    return {
        "ai_message": _apply_turn_policy_to_message(state, ai_message),
        "next_phase": "EXPLORE",  # 계속 탐색 단계 유지
    }


# ----------------------------------------------------
# 1-2. 아이디어가 있을 때 (YES 선택) : 팀 대화 안내 노드
# ----------------------------------------------------
def topic_exists_node(state: AgentState):
    if _is_initial_button_selection(state):
        return {
            "ai_message": FAST_TOPIC_EXISTS_REPLY,
            "collected_data": _prune_collected_data(state.get("collected_data") or {}),
            "is_sufficient": False,
            "next_phase": "TOPIC_SET",
        }

    user_message = _effective_user_message(state)
    current_data = _prune_collected_data(state.get("collected_data") or {})
    extracted_title = ""
    if _is_capture_title_turn(state) or not _is_meaningful_fact(current_data.get("title")):
        extracted_title = _normalize_topic_title(_extract_topic_candidate(user_message))
    if extracted_title:
        merged_data = merge_collected_data(current_data, {"title": extracted_title})
        return {
            "ai_message": (
                f"좋아요. 주제는 '{extracted_title}'로 정리해둘게요. "
                "이 주제로 어떤 문제를 풀고 싶은지, 또는 어떤 결과물을 만들고 싶은지 한두 줄로 말해 주세요."
            ),
            "collected_data": merged_data,
            "is_sufficient": _is_template_ready(merged_data),
            "next_phase": "GATHER",
        }
    if not _is_meaningful_fact(current_data.get("title")) and user_message:
        return {
            "ai_message": (
                "주제 후보를 이해하려면 한 줄로 더 구체적으로 적어주셔야 합니다. "
                "예를 들면 '대학생 팀플 일정 관리 앱'처럼 적어 주세요."
            ),
            "collected_data": current_data,
            "is_sufficient": False,
            "next_phase": "TOPIC_SET",
        }

    recent_context = _build_recent_context(state)
    has_recent_context = recent_context != "(전달된 최근 대화 없음)"

    if not user_message and not has_recent_context:
        ai_message = _build_topic_exists_fallback_message()
    else:
        prompt = f"""
        당신은 프로젝트 주제가 이미 정해진 팀을 돕는 AI PM입니다.
        사용자의 최신 입력과 최근 팀 대화를 읽고, 지금 상황에 맞는 짧은 안내를 한국어로 작성하세요.
        아래 규칙을 지키세요.

        규칙:
        - 2~3문장으로만 답하세요.
        - 최종 답변은 공백 포함 300자 이내로 작성하세요.
        - 먼저 사용자의 현재 논의 주제나 관심사를 짧게 짚어 주세요.
        - 그 다음 팀원끼리 이어서 논의해도 된다는 점과, 막히면 @mates로 도움받을 수 있다는 점을 자연스럽게 안내하세요.
        - "무엇을 도와드릴까요?", "다음 중 선택하세요" 같은 고객센터식 문장은 금지합니다.
        - 입력 정보가 적으면 과장하지 말고 일반적인 표현으로 답하세요.
        - 아래 원칙을 반드시 지키세요.
        {PLAIN_LANGUAGE_RULES}

        [최근 대화 문맥]
        {recent_context}

        [사용자 입력]
        {user_message}
        """
        response = _invoke_llm(conversation_llm, prompt, label="topic_exists_node llm")
        ai_message = (
            str(getattr(response, "content", "")).strip()
            if response is not None
            else _build_topic_exists_fallback_message()
        )
        if not ai_message:
            ai_message = _build_topic_exists_fallback_message()

    return {
        "ai_message": _apply_turn_policy_to_message(state, ai_message),
        "next_phase": "TOPIC_SET",
    }


# ----------------------------------------------------
# 2. 아이디어가 있을 때 (YES 선택) : 정보 수집 노드 (자연스러운 HMW)
# ----------------------------------------------------
def gather_information_node(state: AgentState):
    user_message = _effective_user_message(state)
    turn_policy = _get_turn_policy(state)
    current_data = _prune_collected_data(state.get("collected_data") or {})
    prefilled_data = merge_collected_data(current_data, _seed_topic_title(state, current_data))
    was_ready = _is_template_ready(current_data)
    focus_type = _infer_conversation_focus(state)
    if state.get("current_phase") == "TOPIC_SET" and _is_meaningful_fact(prefilled_data.get("title")):
        focus_type = focus_type or "goal"
    focus_instruction = _build_gather_focus_instruction(focus_type)
    missing_field_summary = _build_missing_field_summary(prefilled_data)
    rag_context = _fetch_rag_context(
        state,
        phase=state.get("current_phase", "GATHER"),
        **_get_rag_filters("GATHER"),
    )
    recent_context = _build_recent_context(state)

    eval_prompt = f"""
    You are an AI PM who helps a project team make decisions, not a form-filling chatbot.
    The user is a teammate who may ask for advice, options, examples, or summaries while the project is still unclear.
    Respond in Korean.

    First classify the user intent into one of these values:
    - answer_fact: the user provides a concrete fact or decision
    - ask_advice: the user asks for explanation, reasoning, criteria, or next-step guidance
    - ask_idea: the user asks for ideas, options, or recommendations
    - ask_summary: the user asks to summarize or organize the current session
    - uncertain: the user does not know yet and wants help narrowing things down
    - frustrated: the user is irritated or rejecting the current flow and needs a reset
    - general: anything else

    Core rules:
    - If the user asks a question, answer it first.
    - If the direct user message is empty because the user only mentioned @mates, infer the request from the recent conversation and respond to the most relevant unresolved point.
    - If current_phase is TOPIC_SET and the user seems to have provided a project subject, capture it as title in updated_data.title.
    - If title is already identified in TOPIC_SET, the next turn should help clarify what problem the topic solves or what outcome the team wants, instead of asking for the title again.
    - If the user does not know, do not repeat the same question. Give options, examples, decision criteria, or a recommendation.
    - If the user asks for ideas, provide 2 to 4 realistic options and recommend one when possible.
    - If the user asks for a summary, summarize confirmed decisions, unresolved points, and the next decision to make.
    - Do not jump to the next checklist question before addressing the current request.
    - Ask at most one short follow-up question, and only if it meaningfully helps the next decision.
    - If your answer already feels complete enough for this turn, end without a follow-up question.
    - Do not promise future actions like "I will show it soon". Finish this turn with concrete content now.
    - Avoid customer-support tone, interview-style repetitive questioning, and system-style phrases.
    - If the user sounds frustrated, acknowledge it briefly and then move to practical help.
    - collected_data is internal structured state. Only store confirmed facts from the conversation.
    - Never store guesses, temporary ideas, user questions, complaints, or "I don't know" style replies in updated_data.
    - Discussion about target users, benefit, or importance may appear in ai_message, but must not be force-mapped into unrelated collected_data keys.
    - {focus_instruction}
    - Follow these writing rules as well:
    {PLAIN_LANGUAGE_RULES}

    [Current turn policy]
    {turn_policy}
    - ANSWER_ONLY: answer the current request and finish without a follow-up question.
    - ANSWER_THEN_ASK: answer first, then optionally add one short follow-up question only if needed.
    - CAPTURE_TITLE: if a project subject is present, store it as updated_data.title before anything else.

    [Reference context]
    {rag_context}

    [Recent conversation]
    {recent_context}

    [Current collected data]
    {json.dumps(prefilled_data, ensure_ascii=False)}

    [Still missing]
    {missing_field_summary}

    [Current focus]
    {focus_type or "none"}

    [User message]
    {user_message}

    [Required JSON output]
    {{
        "intent": "answer_fact | ask_advice | ask_idea | ask_summary | uncertain | frustrated | general",
        "ai_message": "A practical PM-style reply in Korean",
        "updated_data": {build_collected_data_json_example()},
        "is_sufficient": false
    }}

    [Output criteria]
    - ai_message should be 2 to 5 sentences, or up to 4 short bullets.
    - ai_message must be within 300 characters including spaces.
    - If the user asked for advice, ideas, or a summary, ai_message must contain real content. Do not only ask another question.
    - Prefer finishing with an answer. Add a follow-up question only when the conversation genuinely needs one more decision input.
    - updated_data must contain only confirmed facts from this turn.
    - If the user asked for recommendations, explanations, or said they do not know, updated_data may be empty.
    - Be conservative with is_sufficient. The server will validate readiness again.
    """

    response = _invoke_llm(
        structured_llm,
        eval_prompt,
        label="gather_information_node llm",
        response_format={"type": "json_object"},
    )
    if response is None:
        return {
            "ai_message": _apply_turn_policy_to_message(state, _answer_only_fallback(state, "")),
            "collected_data": prefilled_data,
            "is_sufficient": was_ready,
            "next_phase": "READY" if was_ready else "GATHER",
        }

    try:
        raw_result = json.loads(response.content)
        result = GatherLLMResponse.model_validate(raw_result)
    except (JSONDecodeError, ValidationError) as exc:
        logger.exception(
            "failed to parse gather_information_node JSON: %s raw_output=%s",
            exc,
            getattr(response, "content", ""),
        )
        return {
            "ai_message": _apply_turn_policy_to_message(state, _answer_only_fallback(state, "")),
            "collected_data": prefilled_data,
            "is_sufficient": was_ready,
            "next_phase": "READY" if was_ready else "GATHER",
        }

    filtered_updates = _filter_gather_updates(
        user_message,
        result.normalized_updated_data(),
        focus_type=focus_type,
    )
    direct_updates = _extract_direct_fact_updates(user_message)
    if "title" not in direct_updates and not _is_meaningful_fact(prefilled_data.get("title")):
        choice_title = _extract_choice_based_title(state)
        if choice_title:
            direct_updates["title"] = choice_title
    merged_data = merge_collected_data(prefilled_data, filtered_updates)
    merged_data = merge_collected_data(merged_data, direct_updates)
    if filtered_updates or direct_updates:
        logger.info(
            "collected_data updates llm=%s direct=%s merged=%s",
            filtered_updates,
            direct_updates,
            merged_data,
        )
    is_sufficient = _is_template_ready(merged_data)
    ai_msg = str(result.ai_message or "").strip()

    if is_sufficient and not was_ready:
        ai_msg += (
            "\n\n???? ?? ???? ?? ??? ??? ?? ? ????. "
            "??? ?? ??? ???? ???."
        )
    ai_msg = _apply_turn_policy_to_message(state, ai_msg)

    next_phase = "READY" if is_sufficient else "GATHER"

    return {
        "ai_message": ai_msg,
        "collected_data": merged_data,
        "is_sufficient": is_sufficient,
        "next_phase": next_phase,
    }


