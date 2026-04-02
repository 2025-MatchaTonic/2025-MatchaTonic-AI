# app/ai/graph/nodes.py
import json
import logging
import re
from json import JSONDecodeError
from time import perf_counter

from pydantic import ValidationError

from app.ai.graph.collected_data import (
    build_collected_data_json_example,
    merge_collected_data,
)
from app.ai.graph.llm_clients import (
    conversation_llm,
    invoke_llm as _invoke_llm,
    structured_llm,
)
from app.ai.graph.state import AgentState, TurnPolicy
from app.ai.graph.template_support import build_recent_context as _build_recent_context
from app.ai.graph.text_support import (
    MATES_MENTION_PATTERN,
    PLAIN_LANGUAGE_RULES,
    clean_text as _clean_text,
    strip_mates_mention as _strip_mates_mention,
    truncate_message as _truncate_ai_message,
)
from app.ai.schemas.llm_outputs import GatherLLMResponse
from app.core.config import settings
from app.rag.retriever import get_rag_context

logger = logging.getLogger(__name__)

RAG_EMPTY_CONTEXT = "(관련 레퍼런스를 찾지 못했습니다.)"
FAST_EXPLORE_REPLY = "좋아요. 최근 일주일 동안 '이거 좀 불편하다' 싶었던 순간이 있었나요?"
SHORT_MESSAGE_PATTERN = re.compile(r"[^a-z0-9가-힣]+")
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
GREETING_TOKENS = {
    "hi",
    "hello",
    "hey",
    "\uc548\ub155",
    "\uc548\ub155\ud558\uc138\uc694",
    "\ubc18\uac00\uc6cc",
    "\ubc18\uac11\uc2b5\ub2c8\ub2e4",
}
TEAM_SIZE_GENERIC_PATTERN = re.compile(r"(?<!\d)(\d{1,2})\s*명(?!\d)")
ROLE_SPLIT_PATTERN = re.compile(r"\s*(?:,|/|및|그리고|와|과)\s*")
ROLE_PREFIX_PATTERN = re.compile(
    r"^\s*(?:역할|역할은|담당|담당은|구성|구성은|멤버|멤버는)\s*[:은는이가]?\s*"
)
ROLE_TRAILING_SPLIT_PATTERN = re.compile(
    r"\s*(?:이렇게|정도로|정할게|정할 거|정할|나눌 거|나눌게|나눌|나눠|운영|하려고|할게|세부|포지션)"
)
SUMMARY_REQUEST_KEYWORDS = (
    "요약",
    "정리해줘",
    "정리해 줘",
    "지금 모인 정보",
    "현재까지",
    "세션 요약",
)
TEAM_SIZE_HINT_KEYWORDS = ("팀", "인원", "멤버", "우리", "총", "전체")
ROLE_TOKEN_HINTS = (
    "개발자",
    "개발",
    "기획자",
    "기획",
    "pm",
    "po",
    "디자이너",
    "디자인",
    "백엔드",
    "프론트엔드",
    "프론트",
    "서버",
    "ios",
    "android",
    "안드로이드",
    "데이터",
    "ai",
)
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
FILL_REMAINING_EXACT_KEYWORDS = (
    "나머지 다 확정해줘",
    "나머지 확정해줘",
    "나머지 다 정해줘",
    "나머지 다 지정해줘",
    "나머지 다 채워줘",
    "남은 것들 다 확정해줘",
    "전부 확정해줘",
    "다 확정해줘",
    "다 정해줘",
    "다 지정해줘",
    "다 채워줘",
    "전부 정해줘",
)
FILL_REMAINING_TRIGGER_KEYWORDS = (
    "확정",
    "정해",
    "지정",
    "채워",
    "반영",
    "업데이트",
    "저장",
    "기록",
    "맞춰",
    "완성",
)
FILL_REMAINING_SCOPE_KEYWORDS = (
    "나머지",
    "남은",
    "전부",
    "전체",
    "모두",
    "세션 요약",
    "세션요약",
    "핵심 결정사항",
    "저 부분",
    "위 내용",
    "방금 내용",
    "이걸로",
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
CHOICE_INDEX_PATTERN = re.compile(r"^\s*([1-9])\s*(?:번)?\s*$")
NUMBERED_OPTION_LINE_PATTERN = re.compile(r"^\s*(\d{1,2})[)\.\-:]\s*(.+?)\s*$")
NUMBERED_OPTION_INLINE_PATTERN = re.compile(
    r"(\d{1,2})[)\.\-:]\s*(.+?)(?=\s+\d{1,2}[)\.\-:]|$)"
)
DIRECT_FACT_ENDING_PATTERN = re.compile(
    r"\s*(?:입니다|이에요|예요|이야|야|요)\s*$"
)
FAST_RAG_PHASES = {"EXPLORE", "TOPIC_SET", "GATHER"}
RAG_CACHE_MAX_ITEMS = 128
RAG_CONTEXT_CACHE: dict[tuple[str, str, tuple[str, ...], tuple[str, ...], int], str] = {}

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


def _effective_user_message(state: AgentState) -> str:
    return _strip_mates_mention(state.get("user_message"))


def _is_trivial_message(user_message: str) -> bool:
    normalized = _normalize_message(user_message)
    return normalized in TRIVIAL_MESSAGES


def _is_greeting_message(user_message: str) -> bool:
    normalized = _normalize_button_token(_strip_mates_mention(user_message))
    return normalized in GREETING_TOKENS


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
        r"\s+(?:목표|goal|역할|role|team\s*size|팀\s*인원|인원|마감|deadline|due(?:\s*date)?|산출물|결과물|deliverable(?:s)?)\s*(?:은|는|이|가|:)\s+",
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


def _extract_title_updates_for_topic_set(
    state: AgentState, current_data: dict[str, str] | None = None
) -> dict[str, str]:
    current_data = _prune_collected_data(current_data or state.get("collected_data") or {})
    if _is_meaningful_fact(current_data.get("title")):
        return {}

    user_message = _effective_user_message(state)
    if not user_message:
        return {}

    direct_updates = _extract_direct_fact_updates(user_message)
    if "title" in direct_updates:
        return {"title": direct_updates["title"]}

    choice_title = _extract_choice_based_title(state)
    if choice_title:
        return {"title": choice_title}

    if _is_capture_title_turn(state) or state.get("current_phase") in {"EXPLORE", "TOPIC_SET"}:
        candidate = _normalize_topic_title(_extract_topic_candidate(user_message))
        if candidate:
            return {"title": candidate}

    return {}


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


def _answer_only_fallback(state: AgentState, message: str) -> str:
    if str(message or "").strip():
        return str(message).strip()

    user_message = _effective_user_message(state)
    current_data = _prune_collected_data(state.get("collected_data") or {})
    title = _clean_text(current_data.get("title"))
    requested_focus = _detect_requested_gather_focus(user_message) or _infer_conversation_focus(
        state
    )

    if _is_greeting_message(user_message):
        if title:
            return (
                f"\uc548\ub155\ud558\uc138\uc694. '{title}' \uae30\uc900\uc73c\ub85c "
                "\ubaa9\ud45c, \uc5ed\ud560, \uc77c\uc815 \uc911 \ud544\uc694\ud55c \uac83\ubd80\ud130 "
                "\ubc14\ub85c \uc815\ub9ac\ud574\ub4dc\ub9b4\uac8c\uc694."
            )
        return (
            "\uc548\ub155\ud558\uc138\uc694. \uc8fc\uc81c\ub098 \ud544\uc694\ud55c \ud56d\ubaa9\uc744 "
            "\ud55c \uc904\ub85c \ub9d0\ud574\uc8fc\uc2dc\uba74 \uadf8 \ub2e8\uacc4\ubd80\ud130 "
            "\ubc14\ub85c \uc774\uc5b4\uac00\uaca0\uc2b5\ub2c8\ub2e4."
        )

    if title and requested_focus == "goal":
        return f"'{title}' \uae30\uc900\uc73c\ub85c \ud504\ub85c\uc81d\ud2b8 \ubaa9\ud45c\ubd80\ud130 \uc815\ub9ac\ud574\ubcfc\uac8c\uc694."
    if title and requested_focus == "roles":
        return f"'{title}' \uae30\uc900\uc73c\ub85c \ud300 \uc5ed\ud560 \ubd84\ub2f4\ubd80\ud130 \uc815\ub9ac\ud574\ubcfc\uac8c\uc694."
    if title and requested_focus == "teamSize":
        return f"'{title}' \uae30\uc900\uc73c\ub85c \ud300 \uc778\uc6d0 \uad6c\uc131\ubd80\ud130 \uc815\ub9ac\ud574\ubcfc\uac8c\uc694."
    if title and requested_focus == "dueDate":
        return f"'{title}' \uae30\uc900\uc73c\ub85c \ub9c8\uac10 \uc77c\uc815\ubd80\ud130 \uc815\ub9ac\ud574\ubcfc\uac8c\uc694."
    if title and requested_focus == "deliverables":
        return f"'{title}' \uae30\uc900\uc73c\ub85c \uc0b0\ucd9c\ubb3c\ubd80\ud130 \uc815\ub9ac\ud574\ubcfc\uac8c\uc694."
    if title:
        return f"'{title}' \uae30\uc900\uc73c\ub85c \ub2e4\uc74c \ud56d\ubaa9\uc744 \uc774\uc5b4\uc11c \uc815\ub9ac\ud574\ubcfc\uac8c\uc694."

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


def _is_valid_collected_value(key: str, value: object) -> bool:
    cleaned = _clean_text(value)
    if not _is_meaningful_fact(cleaned):
        return False

    if key in {"title", "goal"} and re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
        return False

    return True


def _prune_collected_data(data: dict[str, str]) -> dict[str, str]:
    return {
        key: _clean_text(value)
        for key, value in (data or {}).items()
        if _is_valid_collected_value(key, value)
    }


def _detect_gather_focus(text: str) -> str | None:
    normalized = _clean_text(text)
    if not normalized:
        return None

    for focus, keywords in GATHER_FOCUS_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return focus
    return None


def _detect_requested_gather_focus(text: str) -> str | None:
    normalized = _clean_text(text)
    if not normalized:
        return None

    matched_focus: str | None = None
    matched_index = -1
    for focus, keywords in GATHER_FOCUS_KEYWORDS.items():
        keyword_index = max((normalized.rfind(keyword) for keyword in keywords if keyword in normalized), default=-1)
        if keyword_index > matched_index:
            matched_focus = focus
            matched_index = keyword_index

    return matched_focus if matched_index >= 0 else None


def _infer_latest_user_intent(state: AgentState) -> str:
    user_message = _effective_user_message(state)
    current_data = _prune_collected_data(state.get("collected_data") or {})

    if _is_greeting_message(user_message):
        return "greeting"
    if _is_fill_remaining_request(user_message, current_data):
        return "fill_remaining"

    requested_focus = _detect_requested_gather_focus(user_message)
    if requested_focus:
        return f"direct_request:{requested_focus}"

    if _is_help_request(user_message):
        return "help_request"
    if _extract_title_updates_for_topic_set(state, current_data):
        return "set_topic"
    return "general"


def _infer_conversation_focus(state: AgentState) -> str | None:
    candidates: list[str] = []
    user_message = _clean_text(_effective_user_message(state))
    selected_message = _clean_text(_strip_mates_mention(state.get("selected_message")))
    recent_messages = [
        _clean_text(_strip_mates_mention(msg))
        for msg in state.get("recent_messages", [])
        if _clean_text(_strip_mates_mention(msg))
    ]

    if user_message:
        explicit_focus = _detect_requested_gather_focus(user_message)
        if explicit_focus:
            return explicit_focus
        candidates.append(user_message)
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
        if not _is_valid_collected_value(key, current_data.get(key))
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


def _is_summary_request(user_message: str) -> bool:
    normalized = _clean_text(_strip_mates_mention(user_message))
    if not normalized:
        return False
    return any(keyword in normalized for keyword in SUMMARY_REQUEST_KEYWORDS)


def _extract_team_size_from_message(message: str) -> str:
    explicit_match = TEAM_SIZE_PATTERN.search(message)
    if explicit_match:
        return explicit_match.group(1).strip()

    generic_matches = list(TEAM_SIZE_GENERIC_PATTERN.finditer(message))
    if not generic_matches:
        return ""

    if len(generic_matches) == 1:
        if any(keyword in message for keyword in TEAM_SIZE_HINT_KEYWORDS) or len(message) <= 16:
            return generic_matches[0].group(1).strip()
        return ""

    for match in generic_matches:
        window_start = max(0, match.start() - 8)
        window_end = min(len(message), match.end() + 8)
        window = message[window_start:window_end]
        if any(keyword in window for keyword in TEAM_SIZE_HINT_KEYWORDS):
            return match.group(1).strip()

    return ""


def _looks_like_role_token(token: str) -> bool:
    lowered = token.lower()
    if not token or len(token) > 20:
        return False
    return any(keyword in lowered for keyword in ROLE_TOKEN_HINTS)


def _normalize_role_token(token: str) -> str:
    cleaned = _clean_text(token)
    if not cleaned:
        return ""

    cleaned = ROLE_PREFIX_PATTERN.sub("", cleaned)
    cleaned = ROLE_TRAILING_SPLIT_PATTERN.split(cleaned, maxsplit=1)[0].strip()
    cleaned = cleaned.strip(" .,!?:;\"'()[]")
    if not cleaned:
        return ""

    lowered = cleaned.lower()
    if lowered == "pm":
        return "PM"
    if lowered == "po":
        return "PO"
    if lowered == "ai":
        return "AI"
    if lowered == "ios":
        return "iOS"
    return cleaned


def _extract_roles_from_message(message: str) -> str:
    role_match = ROLE_PATTERN.search(message)
    candidate = role_match.group(1) if role_match else message
    candidate = ROLE_TRAILING_SPLIT_PATTERN.split(candidate, maxsplit=1)[0]
    candidate = candidate.split("\n", 1)[0].strip()

    roles: list[str] = []
    for part in ROLE_SPLIT_PATTERN.split(candidate):
        normalized = _normalize_role_token(part)
        if not _looks_like_role_token(normalized):
            continue
        if normalized not in roles:
            roles.append(normalized)

    return ", ".join(roles)


def _build_collected_data_summary(current_data: dict[str, str]) -> str:
    normalized = _prune_collected_data(current_data)
    labels = {
        "title": "제목",
        "goal": "목표",
        "teamSize": "팀 인원",
        "roles": "역할",
        "dueDate": "마감일",
        "deliverables": "산출물",
    }
    ordered_keys = ["title", "goal", "teamSize", "roles", "dueDate", "deliverables"]

    confirmed_parts = []
    for key in ordered_keys:
        if not _is_meaningful_fact(normalized.get(key)):
            continue
        value = normalized[key]
        if key == "teamSize":
            value = f"{value}명"
        confirmed_parts.append(f"{labels[key]} {value}")
    missing_parts = [labels[key] for key in ordered_keys if not _is_meaningful_fact(normalized.get(key))]

    if confirmed_parts and missing_parts:
        return (
            "현재까지 확정된 정보는 "
            + ", ".join(confirmed_parts)
            + "입니다. 아직 미정인 항목은 "
            + ", ".join(missing_parts)
            + "입니다."
        )
    if confirmed_parts:
        return "현재까지 확정된 정보는 " + ", ".join(confirmed_parts) + "입니다."
    return "아직 확정된 정보는 없습니다. 제목, 목표, 팀 인원, 역할 중 하나부터 정리하면 됩니다."


def _build_next_missing_field_prompt(current_data: dict[str, str]) -> str:
    normalized = _prune_collected_data(current_data)
    for key in ("title", "goal", "dueDate", "deliverables", "roles", "teamSize"):
        if _is_meaningful_fact(normalized.get(key)):
            continue
        if key == "title":
            return "다음으로 프로젝트 제목을 한 줄로 정해볼까요?"
        return GATHER_FIELD_GUIDE[key]["question"]
    return ""


def _looks_like_template_ready_claim(message: str) -> bool:
    normalized = _clean_text(message)
    if not normalized:
        return False
    template_keywords = ("템플릿", "template", "생성", "만들")
    ready_keywords = ("충분", "모였", "만들 수", "준비", "ready")
    return any(keyword in normalized for keyword in template_keywords) and any(
        keyword in normalized for keyword in ready_keywords
    )


def _trim_option_description(candidate: str) -> str:
    text = _clean_text(candidate)
    if not text:
        return ""

    parts = re.split(r"\s+[—-]\s+", text, maxsplit=1)
    if len(parts) == 2 and len(parts[0].strip()) >= 2:
        return parts[0].strip()
    return text


def _looks_like_choice_token(value: str) -> bool:
    return bool(CHOICE_INDEX_PATTERN.match(str(value or "").strip()))


def _looks_like_multi_option_block(value: str) -> bool:
    text = _clean_text(value)
    if not text:
        return False

    line_match_count = sum(
        1
        for line in text.splitlines()
        if NUMBERED_OPTION_LINE_PATTERN.match(line.strip())
    )
    if line_match_count >= 2:
        return True

    inline_match_count = sum(1 for _ in NUMBERED_OPTION_INLINE_PATTERN.finditer(text))
    return inline_match_count >= 2


def _is_fill_remaining_request(user_message: str, current_data: dict[str, str]) -> bool:
    normalized = _clean_text(_strip_mates_mention(user_message)).lower()
    if not normalized:
        return False

    missing_fields = [
        key for key in GATHER_FIELD_GUIDE if not _is_meaningful_fact(current_data.get(key))
    ]
    if not missing_fields:
        return False

    if any(keyword in normalized for keyword in FILL_REMAINING_EXACT_KEYWORDS):
        return True

    has_trigger = any(keyword in normalized for keyword in FILL_REMAINING_TRIGGER_KEYWORDS)
    has_scope = any(keyword in normalized for keyword in FILL_REMAINING_SCOPE_KEYWORDS)
    if has_trigger and has_scope:
        return True

    if has_trigger and any(keyword in normalized for keyword in ("다", "전부", "모두", "전체")):
        return True

    return False


def _extract_direct_fact_updates(user_message: str) -> dict[str, str]:
    message = _clean_text(_strip_mates_mention(user_message))
    if not message:
        return {}

    updates: dict[str, str] = {}

    if TITLE_EXPLICIT_PATTERN.match(message):
        topic_candidate = _normalize_topic_title(_extract_topic_candidate(message))
        if topic_candidate and not _looks_like_title_instruction(topic_candidate):
            updates["title"] = topic_candidate

    team_size = _extract_team_size_from_message(message)
    if team_size:
        updates["teamSize"] = team_size

    roles = _extract_roles_from_message(message)
    if roles:
        updates["roles"] = roles

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

            candidate = _normalize_topic_title(_trim_option_description(line_match.group(2)))
            if candidate and not _looks_like_title_instruction(candidate):
                return candidate

        if matched_line_option:
            continue

        for inline_match in NUMBERED_OPTION_INLINE_PATTERN.finditer(block):
            if int(inline_match.group(1)) != choice_index:
                continue

            candidate = _normalize_topic_title(_trim_option_description(inline_match.group(2)))
            if candidate and not _looks_like_title_instruction(candidate):
                return candidate

    return ""


def _sanitize_gather_updates(updated_data: dict[str, str]) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    for key, value in updated_data.items():
        if key != "teamSize" and _looks_like_choice_token(value):
            continue
        if key in {"title", "goal"} and _looks_like_multi_option_block(value):
            continue
        if key == "title":
            value = _normalize_topic_title(value)
        if _is_valid_collected_value(key, value):
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
    return sum(1 for key in GATHER_FIELD_GUIDE if _is_valid_collected_value(key, current_data.get(key)))


def _is_template_ready(current_data: dict[str, str]) -> bool:
    normalized = _prune_collected_data(current_data)
    return all(_is_valid_collected_value(key, normalized.get(key)) for key in GATHER_FIELD_GUIDE)


def _build_topic_exists_fallback_message() -> str:
    return FAST_TOPIC_EXISTS_REPLY


# ----------------------------------------------------
# 1. 아이디어가 없을 때 (NO 선택) : 탐색 노드
def explore_problem_node(state: AgentState):
    user_message = _effective_user_message(state)
    turn_policy = _get_turn_policy(state)
    latest_intent = _infer_latest_user_intent(state)

    if _is_initial_button_selection(state):
        return {
            "ai_message": FAST_EXPLORE_REPLY,
            "next_phase": "EXPLORE",
        }

    if latest_intent == "greeting":
        return {
            "ai_message": _apply_turn_policy_to_message(state, _answer_only_fallback(state, "")),
            "next_phase": "EXPLORE",
        }

    if _is_trivial_message(user_message) and not state.get("recent_messages"):
        return {
            "ai_message": _apply_turn_policy_to_message(state, _answer_only_fallback(state, "")),
            "next_phase": "EXPLORE",
        }

    rag_context = _fetch_rag_context(
        state,
        phase="EXPLORE",
        **_get_rag_filters("EXPLORE"),
    )
    recent_context = _build_recent_context(state)
    prompt = f"""
    You are an AI PM helping a team that does not have a topic yet.
    Respond in Korean.

    Rules:
    - Answer the current request first.
    - Keep the reply within 220 characters.
    - Ask at most one short follow-up question only if needed.
    - If the user wants topic ideas, give exactly 3 options from different domains or user groups and recommend 1.
    - Do not give 3 variants from the same category.
    - Sound practical, not like customer support.
    {PLAIN_LANGUAGE_RULES}

    [Turn policy]
    {turn_policy}

    [Latest user intent]
    {latest_intent}

    [Reference context]
    {rag_context}

    [Recent conversation]
    {recent_context}

    [User message]
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
    latest_intent = _infer_latest_user_intent(state)
    title_updates = _extract_title_updates_for_topic_set(state, current_data)
    extracted_title = title_updates.get("title", "")
    if extracted_title:
        merged_data = merge_collected_data(current_data, title_updates)
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
        You are an AI PM for a team that already has a topic.
        Respond in Korean.

        Rules:
        - Give a short practical reply for the current situation.
        - Keep it to 2 or 3 sentences and within 220 characters.
        - Reflect the current topic or recent context first.
        - Prioritize the latest user request over phase guidance.
        - If helpful, suggest only one next useful step.
        - Avoid customer-support phrasing.
        {PLAIN_LANGUAGE_RULES}

        [Latest user intent]
        {latest_intent}

        [Recent conversation]
        {recent_context}

        [User message]
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
    current_phase = str(state.get("current_phase") or "")
    current_data = _prune_collected_data(state.get("collected_data") or {})
    prefilled_data = dict(current_data)
    was_ready = _is_template_ready(current_data)
    latest_intent = _infer_latest_user_intent(state)
    focus_type = _infer_conversation_focus(state)
    direct_updates = {
        key: value
        for key, value in _extract_direct_fact_updates(user_message).items()
        if key != "title"
    }
    merged_preview = merge_collected_data(prefilled_data, direct_updates)
    if state.get("current_phase") == "TOPIC_SET" and _is_meaningful_fact(prefilled_data.get("title")):
        focus_type = focus_type or "goal"
    if _is_summary_request(user_message):
        preview_is_sufficient = _is_template_ready(merged_preview)
        return {
            "ai_message": _apply_turn_policy_to_message(
                state, _build_collected_data_summary(merged_preview)
            ),
            "collected_data": merged_preview,
            "is_sufficient": preview_is_sufficient,
            "next_phase": "READY" if preview_is_sufficient else "GATHER",
        }
    focus_instruction = _build_gather_focus_instruction(focus_type)
    missing_field_summary = _build_missing_field_summary(prefilled_data)
    fill_remaining_request = _is_fill_remaining_request(user_message, prefilled_data)
    rag_context = _fetch_rag_context(
        state,
        phase=state.get("current_phase", "GATHER"),
        **_get_rag_filters("GATHER"),
    )
    recent_context = _build_recent_context(state)

    eval_prompt = f"""
    You are an AI PM. Respond in Korean and output JSON only.

    intent must be one of:
    answer_fact | ask_advice | ask_idea | ask_summary | uncertain | frustrated | general

    Rules:
    - Answer the current request first.
    - Prioritize the latest user request over phase guidance.
    - ai_message must be practical and within 220 characters.
    - Ask at most one short follow-up question only if needed.
    - updated_data stores confirmed facts only.
    - Never store guesses, questions, complaints, or temporary ideas.
    - Do not say the project is ready for template generation unless all collected data fields are filled with valid values.
    - If fill_remaining_request is true, you may fill empty fields with practical defaults grounded in the topic and recent context. Do not overwrite confirmed values unless the user clearly changes them.
    - {focus_instruction}
    {PLAIN_LANGUAGE_RULES}

    [Turn policy]
    {turn_policy}

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

    [Latest user intent]
    {latest_intent}

    [Fill remaining request]
    {fill_remaining_request}

    [User message]
    {user_message}

    Output:
    {{
        "intent": "answer_fact | ask_advice | ask_idea | ask_summary | uncertain | frustrated | general",
        "ai_message": "short Korean reply",
        "updated_data": {build_collected_data_json_example()},
        "is_sufficient": false
    }}
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
    filtered_updates.pop("title", None)
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

    if not is_sufficient and _looks_like_template_ready_claim(ai_msg):
        follow_up = _build_next_missing_field_prompt(merged_data)
        ai_msg = _build_collected_data_summary(merged_data)
        if follow_up:
            ai_msg = f"{ai_msg} {follow_up}"

    should_prompt_template = is_sufficient and (not was_ready or current_phase != "READY")
    if should_prompt_template and "템플릿" not in ai_msg:
        if _is_answer_only_turn(state):
            prompt_message = (
                "이제 템플릿을 만들 수 있을 만큼 정보가 모였어요. "
                "원하시면 기획용 또는 개발용 템플릿을 생성할 수 있어요."
            )
        else:
            prompt_message = "이제 템플릿을 만들 수 있을 만큼 정보가 모였어요. 템플릿 만드시겠습니까?"
        ai_msg = f"{ai_msg}\n\n{prompt_message}".strip() if ai_msg else prompt_message
    ai_msg = _apply_turn_policy_to_message(state, ai_msg)

    next_phase = "READY" if is_sufficient else "GATHER"

    return {
        "ai_message": ai_msg,
        "collected_data": merged_data,
        "is_sufficient": is_sufficient,
        "next_phase": next_phase,
    }


