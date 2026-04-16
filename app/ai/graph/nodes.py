# app/ai/graph/nodes.py
import json
import logging
import re
from datetime import date
from json import JSONDecodeError
from time import perf_counter

from app.ai.graph.collected_data import (
    FIELD_POLICY,
    CandidateDecision,
    apply_collected_data_updates,
    build_decision_context,
    build_collected_data_json_example,
    build_role_team_size_conflict_message,
    choose_next_question_field,
    classify_role_team_size_conflict,
    derive_phase_from_collected_data,
    evaluate_candidate_update,
    format_collected_value,
    has_role_team_size_conflict,
    infer_contextual_prompted_slot,
    infer_prompted_slot,
    is_template_ready as _shared_is_template_ready,
    is_valid_collected_value as _shared_is_valid_collected_value,
    is_placeholder_value,
    is_request_like_value,
    is_undecided_value,
    looks_like_non_committal_value,
    merge_collected_data,
    missing_collected_fields as _shared_missing_collected_fields,
    normalize_roles,
    sanitize_llm_updated_data,
    sanitize_candidate_updates as _shared_sanitize_candidate_updates,
    sanitize_collected_data as _shared_sanitize_collected_data,
    subject_needs_problem_definition,
)
from app.ai.graph.llm_clients import (
    conversation_llm,
    invoke_llm as _invoke_llm,
    structured_llm,
)
from app.ai.graph.conversation_heuristics import (
    CHOICE_INDEX_PATTERN,
    CHOICE_PREFIX_PATTERN,
    DELIVERABLES_PATTERN,
    DIRECT_FACT_ENDING_PATTERN,
    DUE_DATE_PATTERN,
    FAST_RAG_PHASES,
    FILL_REMAINING_EXACT_KEYWORDS,
    FILL_REMAINING_SCOPE_KEYWORDS,
    FILL_REMAINING_TRIGGER_KEYWORDS,
    GATHER_FIELD_GUIDE,
    GATHER_FOCUS_KEYWORDS,
    GOAL_PATTERN,
    GREETING_TOKENS,
    KOREAN_DUE_DATE_CANDIDATE_PATTERNS,
    NUMBERED_OPTION_INLINE_PATTERN,
    NUMBERED_OPTION_LINE_PATTERN,
    PROBLEM_AREA_CONTEXT_PATTERNS,
    PROBLEM_AREA_PATTERN,
    QUESTION_LINE_ENDING_PATTERN,
    RAG_FILTERS_BY_PHASE,
    ROLE_PATTERN,
    ROLE_PREFIX_PATTERN,
    ROLE_SPLIT_PATTERN,
    ROLE_TOKEN_HINTS,
    ROLE_TRAILING_SPLIT_PATTERN,
    SHORT_MESSAGE_PATTERN,
    SUBJECT_PATTERN,
    TARGET_FACILITY_NOUN_PATTERN,
    TARGET_FACILITY_PROMPT_PATTERN,
    TEAM_SIZE_GENERIC_PATTERN,
    TEAM_SIZE_HINT_KEYWORDS,
    TEAM_SIZE_PATTERN,
    TITLE_EXPLICIT_PATTERN,
    TITLE_INSTRUCTION_KEYWORDS,
    TRAILING_TOPIC_ENDINGS_PATTERN,
    TRIVIAL_MESSAGES,
    UNSUPPORTED_GATHER_TOPICS,
)
from app.ai.graph.conversation_signals import (
    is_guidance_signal as _signal_is_guidance_signal,
    is_help_request as _signal_is_help_request,
    is_meta_conversation_message as _signal_is_meta_conversation_message,
    is_next_step_request as _signal_is_next_step_request,
    is_summary_request as _signal_is_summary_request,
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
from app.ai.graph.topic_presence import (
    _is_topic_presence_negative_message,
    _matches_initial_button_message,
    _matches_topic_presence_button_message,
    _normalize_button_token,
)
from app.core.config import settings
from app.rag.retriever import get_rag_context

logger = logging.getLogger(__name__)

RAG_EMPTY_CONTEXT = "(관련 레퍼런스를 찾지 못했습니다.)"
FAST_EXPLORE_REPLY = "좋아요. 최근 일주일 동안 '이거 좀 불편하다' 싶었던 순간이 있었나요?"
FAST_TOPIC_EXISTS_REPLY = (
    "좋아요. 이미 생각해둔 주제가 있다면 지금 한두 줄로 보내주세요. "
    "입력해주신 내용을 바탕으로 프로젝트 주제를 짧게 정리해 반영하겠습니다."
)
RAG_CACHE_MAX_ITEMS = 128
RAG_CONTEXT_CACHE: dict[tuple[str, str, tuple[str, ...], tuple[str, ...], int], str] = {}


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


def _is_initial_button_selection(state: AgentState) -> bool:
    action = state.get("action_type")
    message_candidate = _clean_text(state.get("user_message")) or _clean_text(
        state.get("selected_message")
    )

    if action == "CHAT":
        return _matches_topic_presence_button_message(message_candidate)

    if action not in {"BTN_NO", "BTN_YES", "BTN_GO_DEF"}:
        return False

    if not message_candidate:
        return True

    return _matches_initial_button_message(action, message_candidate)


def _build_initial_button_reset_response(state: AgentState) -> dict[str, object]:
    message_candidate = _clean_text(state.get("user_message")) or _clean_text(
        state.get("selected_message")
    )
    is_negative = state.get("action_type") == "BTN_NO" or _is_topic_presence_negative_message(
        message_candidate
    )
    next_phase = "EXPLORE" if is_negative else "TOPIC_SET"
    ai_message = FAST_EXPLORE_REPLY if is_negative else FAST_TOPIC_EXISTS_REPLY
    logger.info(
        "initial_button_reset action=%s current_phase=%s message=%r stale_collected_data=%s next_phase=%s",
        state.get("action_type"),
        state.get("current_phase"),
        message_candidate,
        state.get("collected_data") or {},
        next_phase,
    )
    return {
        "ai_message": _apply_turn_policy_to_message(state, ai_message),
        "collected_data": {},
        "is_sufficient": False,
        "next_phase": next_phase,
    }


def _get_turn_policy(state: AgentState) -> TurnPolicy:
    return state.get("turn_policy", "ANSWER_THEN_ASK")


def _is_answer_only_turn(state: AgentState) -> bool:
    return _get_turn_policy(state) == "ANSWER_ONLY"


def _is_capture_title_turn(state: AgentState) -> bool:
    return _get_turn_policy(state) == "CAPTURE_TITLE"


def _extract_topic_candidate(user_message: str) -> str:
    text = _clean_text(_strip_mates_mention(user_message))
    if not text or "?" in text:
        return ""
    if _is_meta_conversation_message(text):
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
    else:
        loose_match = re.search(
            r"(?:프로젝트\s*주제|주제|아이디어)\s*(?:는|은|으론|로는|은요|는요)?\s*(.+)$",
            candidate,
            flags=re.IGNORECASE,
        )
        if loose_match:
            candidate = loose_match.group(1).strip()

    cleanup_patterns = (
        r"\s*(?:을|를)?\s*주제로\s*(?:할게요|하려고 해요|하고 싶어요)$",
        r"\s*(?:을|를)?\s*만들고\s*싶어요$",
        r"\s*(?:을|를)?\s*만들고\s*싶어$",
        r"\s*(?:을|를|으로|로)?\s*생각하고\s*있어요$",
        r"\s*(?:을|를|으로|로)?\s*생각하고\s*있어$",
        r"\s*(?:으로|로)\s*하려고\s*해요$",
        r"\s*(?:으로|로)\s*하려고\s*해$",
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
    if looks_like_non_committal_value(title):
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
        r"\s*(?:으로|로)?\s*(?:정했어요|하려고\s*해요|하려고\s*해|하고\s*싶어요|하고\s*싶어|해보려고\s*해요|생각하고\s*있어요|생각하고\s*있어)\s*$",
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
    if looks_like_non_committal_value(title):
        return ""
    if _looks_like_title_instruction(title):
        return ""
    return title


def _normalize_topic_title_with_llm(candidate: str, *, field_name: str) -> str:
    normalized = _normalize_topic_title(candidate)
    if not normalized:
        return normalized
    if _looks_like_choice_token(normalized) or is_placeholder_value(normalized):
        return ""
    if _extract_due_date_candidate_from_message(normalized):
        return ""
    if _is_non_storable_freeform_message(normalized) or _is_guidance_signal(normalized):
        logger.info(
            "skip llm normalization for %s candidate because it is non-storable: %r",
            field_name,
            normalized,
        )
        return ""
    if not settings.OPENAI_API_KEY:
        return normalized

    prompt = f"""
    You normalize one short Korean project {field_name} candidate extracted from a user chat message.

    Rules:
    - Fix only obvious typos, spacing, and minor wording noise.
    - Preserve the original meaning. Do not invent a different idea.
    - Keep it short and natural for a project {field_name}.
    - If the input is already natural, return it unchanged.
    - If it is not a storable project {field_name}, return an empty string.

    Input: {normalized}

    Output JSON:
    {{
      "normalized": "..."
    }}
    """

    response = _invoke_llm(
        structured_llm,
        prompt,
        label=f"normalize_{field_name}_candidate llm",
        cache_key=("normalize_topic_title", field_name, normalized),
        response_format={"type": "json_object"},
    )
    if response is None:
        return normalized

    try:
        raw_result = json.loads(response.content)
    except JSONDecodeError:
        logger.warning(
            "failed to parse normalized %s candidate JSON: raw_output=%s",
            field_name,
            getattr(response, "content", ""),
        )
        return normalized

    refined = _normalize_topic_title(str(raw_result.get("normalized", "")).strip())
    if not refined:
        return normalized

    if refined != normalized:
        logger.info(
            "refined %s candidate with llm before=%r after=%r",
            field_name,
            normalized,
            refined,
        )
    return refined


def _postprocess_explicit_title_value(candidate: str) -> str:
    title = _normalize_topic_title(candidate)
    if not title:
        return ""

    explicit_tail_patterns = (
        r"\s*로\s*할래$",
        r"\s*로\s*할게$",
        r"\s*로\s*하자$",
        r"\s*로\s*할까$",
        r"\s*로\s*정할래$",
        r"\s*로\s*정하자$",
        r"\s*로\s*정할게$",
        r"\s*이름으로\s*할래$",
        r"\s*주제로\s*할래$",
        r"\s*이야$",
        r"\s*야$",
    )
    for pattern in explicit_tail_patterns:
        stripped = re.sub(pattern, "", title, flags=re.IGNORECASE).strip(" .,!?:;\"'")
        if stripped and len(stripped) >= 2:
            title = stripped
            break
    return title


def _extract_title_updates_for_topic_set(
    state: AgentState,
    current_data: dict[str, str] | None = None,
    *,
    direct_updates: dict[str, object] | None = None,
) -> dict[str, str]:
    current_data = _prune_collected_data(current_data or state.get("collected_data") or {})
    user_message = _effective_user_message(state)
    if not user_message:
        return {}
    if _matches_topic_presence_button_message(user_message):
        return {}

    direct_updates = dict(direct_updates or _extract_direct_fact_updates(user_message))
    fresh_topic_candidate = _extract_fresh_topic_submission_candidate(
        state,
        current_data,
        direct_updates=direct_updates,
    )
    if fresh_topic_candidate:
        return {"subject": fresh_topic_candidate}

    has_subject = _is_meaningful_fact(current_data.get("subject"))
    has_title = _is_meaningful_fact(current_data.get("title"))
    if has_subject and has_title:
        return {}
    if "subject" in direct_updates and not has_subject:
        return {"subject": direct_updates["subject"]}
    if "title" in direct_updates:
        return {"title": direct_updates["title"]}

    if _is_storage_control_message(user_message, current_data):
        return {}

    choice_title = _extract_choice_based_title(state)
    if has_subject and _extract_problem_area_candidate(state, current_data, direct_updates=direct_updates):
        choice_title = ""
    if choice_title and not has_subject:
        return {"subject": choice_title}
    if choice_title:
        return {"title": choice_title}

    if _is_capture_title_turn(state) or state.get("current_phase") in {"EXPLORE", "TOPIC_SET"}:
        candidate = _normalize_topic_title_with_llm(
            _extract_topic_candidate(user_message),
            field_name="subject" if not has_subject else "title",
        )
        if candidate:
            if not has_subject:
                return {"subject": candidate}
            if not has_title:
                return {"title": candidate}

    return {}


def _extract_fresh_topic_submission_candidate(
    state: AgentState,
    current_data: dict[str, object] | None = None,
    *,
    direct_updates: dict[str, object] | None = None,
) -> str:
    current_data = _prune_collected_data(current_data or state.get("collected_data") or {})
    current_anchor = _get_topic_anchor(current_data, allow_title_fallback=False)
    if not current_anchor:
        return ""

    if str(state.get("current_phase") or "") not in {"TOPIC_SET", "PROBLEM_DEFINE"}:
        return ""

    user_message = _effective_user_message(state)
    normalized_message = _clean_text(user_message)
    if not normalized_message:
        return ""

    direct_updates = dict(direct_updates or {})
    explicit_topic_candidate = str(
        direct_updates.get("subject") or direct_updates.get("title") or ""
    ).strip()
    if explicit_topic_candidate and explicit_topic_candidate != current_anchor:
        return explicit_topic_candidate

    if _is_storage_control_message(normalized_message, current_data):
        return ""
    if _extract_choice_based_title(state):
        return ""
    if _is_awaiting_target_facility(state):
        return ""

    heuristic_candidate = _normalize_topic_title(_extract_topic_candidate(normalized_message))
    candidate = explicit_topic_candidate or heuristic_candidate
    if not candidate:
        return ""
    if candidate == current_anchor:
        return ""

    compact_candidate = re.sub(r"\s+", "", candidate)
    if len(compact_candidate) < 8 and len(candidate.split()) < 2:
        return ""

    return _normalize_topic_title_with_llm(candidate, field_name="subject")


def _looks_like_question_line(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    candidate = cleaned.lstrip("-*0123456789. ").strip()
    return bool(QUESTION_LINE_ENDING_PATTERN.search(candidate))


def _looks_like_open_question(text: str) -> bool:
    cleaned = _clean_text(text)
    if not cleaned:
        return False
    return "?" in cleaned or "？" in cleaned


def _has_structured_fact_updates(direct_updates: dict[str, object] | None) -> bool:
    if not direct_updates:
        return False
    return any(
        key in direct_updates
        for key in ("goal", "targetUser", "teamSize", "roles", "dueDate", "deliverables")
    )


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
    topic_anchor = _get_topic_anchor(current_data)
    requested_focus = _detect_requested_gather_focus(user_message) or _infer_conversation_focus(
        state
    )

    if _is_greeting_message(user_message):
        if topic_anchor:
            return (
                f"\uc548\ub155\ud558\uc138\uc694. '{topic_anchor}' \uae30\uc900\uc73c\ub85c "
                "\ubaa9\ud45c, \uc5ed\ud560, \uc77c\uc815 \uc911 \ud544\uc694\ud55c \uac83\ubd80\ud130 "
                "\ubc14\ub85c \uc815\ub9ac\ud574\ub4dc\ub9b4\uac8c\uc694."
            )
        return (
            "\uc548\ub155\ud558\uc138\uc694. \uc8fc\uc81c\ub098 \ud544\uc694\ud55c \ud56d\ubaa9\uc744 "
            "\ud55c \uc904\ub85c \ub9d0\ud574\uc8fc\uc2dc\uba74 \uadf8 \ub2e8\uacc4\ubd80\ud130 "
            "\ubc14\ub85c \uc774\uc5b4\uac00\uaca0\uc2b5\ub2c8\ub2e4."
        )

    if topic_anchor and requested_focus == "goal":
        return f"'{topic_anchor}' \uae30\uc900\uc73c\ub85c \ud504\ub85c\uc81d\ud2b8 \ubaa9\ud45c\ubd80\ud130 \uc815\ub9ac\ud574\ubcfc\uac8c\uc694."
    if topic_anchor and requested_focus == "roles":
        return f"'{topic_anchor}' \uae30\uc900\uc73c\ub85c \ud300 \uc5ed\ud560 \ubd84\ub2f4\ubd80\ud130 \uc815\ub9ac\ud574\ubcfc\uac8c\uc694."
    if topic_anchor and requested_focus == "teamSize":
        return f"'{topic_anchor}' \uae30\uc900\uc73c\ub85c \ud300 \uc778\uc6d0 \uad6c\uc131\ubd80\ud130 \uc815\ub9ac\ud574\ubcfc\uac8c\uc694."
    if topic_anchor and requested_focus == "dueDate":
        return f"'{topic_anchor}' \uae30\uc900\uc73c\ub85c \ub9c8\uac10 \uc77c\uc815\ubd80\ud130 \uc815\ub9ac\ud574\ubcfc\uac8c\uc694."
    if topic_anchor and requested_focus == "deliverables":
        return f"'{topic_anchor}' \uae30\uc900\uc73c\ub85c \uc0b0\ucd9c\ubb3c\ubd80\ud130 \uc815\ub9ac\ud574\ubcfc\uac8c\uc694."
    if topic_anchor:
        return f"'{topic_anchor}' \uae30\uc900\uc73c\ub85c \ub2e4\uc74c \ud56d\ubaa9\uc744 \uc774\uc5b4\uc11c \uc815\ub9ac\ud574\ubcfc\uac8c\uc694."

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
    return not _is_meaningful_fact(current_data.get("subject"))


def _get_topic_anchor(
    current_data: dict[str, object],
    *,
    allow_title_fallback: bool = True,
) -> str:
    subject = _clean_text(current_data.get("subject"))
    if subject:
        return subject
    if not allow_title_fallback:
        return ""
    return _clean_text(current_data.get("title"))


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
    try:
        context = get_rag_context(
            query=query,
            current_phase=phase,
            k=top_k,
            topics=topics,
            doc_types=doc_types,
        )
    except Exception:
        logger.exception("rag fetch failed phase=%s query=%r", phase, query)
        return RAG_EMPTY_CONTEXT
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
    return _shared_is_valid_collected_value(key, value)


def _prune_collected_data(data: dict[str, str]) -> dict[str, str]:
    return _shared_sanitize_collected_data(data)


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


def _interpret_turn_type(
    state: AgentState,
    current_data: dict[str, str] | None = None,
    *,
    direct_updates: dict[str, object] | None = None,
) -> str:
    current_data = _prune_collected_data(current_data or state.get("collected_data") or {})
    user_message = _effective_user_message(state)
    direct_updates = dict(direct_updates or _extract_direct_fact_updates(user_message))
    current_subject = str(current_data.get("subject") or "").strip()
    broad_subject = subject_needs_problem_definition(current_subject)
    target_facility_candidate = _extract_target_facility_candidate(state, current_data)
    problem_area_candidate = _extract_problem_area_candidate(
        state,
        current_data,
        direct_updates=direct_updates,
    )
    due_date_candidate = str(direct_updates.get("dueDate") or "").strip()
    request_like = _is_request_like_value(user_message)
    undecided = _is_undecided_value(user_message)
    meta_request = _is_meta_conversation_message(user_message)
    prompted_slot = _get_active_prompted_slot(state, current_data)

    logger.info(
        "turn_type_detection message=%r request_like=%s undecided=%s meta=%s direct_candidates=%s target_facility=%r problem_area=%r due_date=%r broad_subject=%s current_data=%s",
        user_message,
        request_like,
        undecided,
        meta_request,
        direct_updates,
        target_facility_candidate,
        problem_area_candidate,
        due_date_candidate,
        broad_subject,
        current_data,
    )

    if meta_request:
        return "meta_request"
    if _is_summary_request(user_message):
        return "request_summary"
    if _is_fill_remaining_request(user_message, current_data):
        return "request_fill_missing"
    if _has_structured_fact_updates(direct_updates):
        return "provide_fact"
    if due_date_candidate:
        return "provide_due_date_candidate"
    if target_facility_candidate:
        return "provide_target_facility"
    if problem_area_candidate and (current_phase := str(state.get("current_phase") or "")) == "PROBLEM_DEFINE":
        return "provide_problem_area"
    if problem_area_candidate and broad_subject:
        return "provide_problem_area"
    if problem_area_candidate:
        return "provide_problem_area"
    if _should_offer_goal_guidance(current_data, user_message, direct_updates=direct_updates):
        return "request_goal_guidance"
    if _should_offer_topic_guidance(current_data, user_message, direct_updates=direct_updates):
        return "request_guided_exploration"
    if _extract_title_updates_for_topic_set(state, current_data, direct_updates=direct_updates):
        return "provide_topic"
    if direct_updates:
        return "provide_fact"
    if prompted_slot and (
        _is_help_request(user_message)
        or _is_guidance_signal(user_message)
        or undecided
        or "잘 모르겠" in user_message
    ) and not (
        _looks_like_advice_request(user_message)
        or _looks_like_planning_request(user_message)
        or any(token in user_message for token in ("정해줘", "추천", "알려줘", "정리해줘"))
    ):
        return "request_help_needed"
    if _is_next_step_request(user_message) or _is_help_request(user_message) or "템플릿" in _clean_text(user_message):
        return "request_next_step"
    return "general"


def _infer_latest_user_intent(state: AgentState) -> str:
    user_message = _effective_user_message(state)
    current_data = _prune_collected_data(state.get("collected_data") or {})

    if _is_greeting_message(user_message):
        return "greeting"
    return _interpret_turn_type(state, current_data)


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


def _get_active_prompted_slot(
    state: AgentState,
    current_data: dict[str, object],
    *,
    followup_fields: list[str] | None = None,
    rejected_updates: dict[str, object] | None = None,
) -> str:
    explicit_slot = str(
        state.get("current_slot") or state.get("next_question_field") or ""
    ).strip()
    if explicit_slot in {
        "subject",
        "title",
        "goal",
        "targetUser",
        "teamSize",
        "roles",
        "dueDate",
        "deliverables",
    }:
        return explicit_slot
    return infer_prompted_slot(
        recent_messages=state.get("recent_messages", []),
        selected_message=state.get("selected_message"),
        current_data=current_data,
        current_phase=str(state.get("current_phase") or "GATHER"),
        followup_fields=followup_fields,
        rejected_updates=rejected_updates,
    )


def _get_contextual_prompted_slot(
    state: AgentState,
    current_data: dict[str, object],
) -> str:
    explicit_slot = str(
        state.get("current_slot") or state.get("next_question_field") or ""
    ).strip()
    if explicit_slot in {
        "subject",
        "title",
        "goal",
        "targetUser",
        "teamSize",
        "roles",
        "dueDate",
        "deliverables",
    }:
        return explicit_slot
    return infer_contextual_prompted_slot(
        recent_messages=state.get("recent_messages", []),
        selected_message=state.get("selected_message"),
    )


def _looks_like_factual_slot_answer(user_message: str) -> bool:
    normalized = _clean_text(user_message)
    if not normalized:
        return False
    if (
        _is_meta_conversation_message(normalized)
        or _is_request_like_value(normalized)
        or _is_undecided_value(normalized)
        or _looks_like_question_line(normalized)
        or _looks_like_open_question(normalized)
    ):
        return False
    return True


def _infer_prompted_slot_updates(
    *,
    prompted_slot: str,
    user_message: str,
    current_data: dict[str, object],
) -> dict[str, object]:
    if prompted_slot not in {
        "goal",
        "targetUser",
        "teamSize",
        "roles",
        "dueDate",
        "deliverables",
    }:
        return {}

    normalized_message = _clean_text(_strip_mates_mention(user_message))
    if not _looks_like_factual_slot_answer(normalized_message):
        return {}

    updates: dict[str, object] = {}
    if prompted_slot == "goal":
        candidate = _extract_goal_candidate_from_message(normalized_message) or _normalize_goal_candidate(
            normalized_message
        )
        if candidate:
            updates["goal"] = candidate
    elif prompted_slot == "targetUser":
        candidate = _normalize_direct_fact_value(normalized_message)
        if candidate:
            updates["targetUser"] = candidate
    elif prompted_slot == "teamSize":
        candidate = _extract_team_size_from_message(normalized_message)
        if candidate:
            updates["teamSize"] = candidate
    elif prompted_slot == "roles":
        candidate = _extract_roles_from_message(normalized_message)
        if candidate:
            updates["roles"] = candidate
    elif prompted_slot == "dueDate":
        candidate = _extract_due_date_candidate_from_message(normalized_message)
        if candidate:
            updates["dueDate"] = candidate
    elif prompted_slot == "deliverables":
        candidate = _normalize_direct_fact_value(normalized_message)
        if candidate:
            updates["deliverables"] = candidate

    if updates:
        logger.info(
            "prompted_slot_inference slot=%s message=%r inferred_updates=%s current_data=%s",
            prompted_slot,
            normalized_message,
            updates,
            current_data,
        )
    return updates


def _build_slot_help_message(slot: str, current_data: dict[str, object]) -> str:
    if slot == "goal":
        return _build_goal_guidance_message(current_data)

    examples = {
        "subject": "예: 공공시설 예약, 팀 프로젝트 일정 관리, 빈 강의실 찾기",
        "title": "예: 혼잡도 나침반, 예약메이트, 팀코디",
        "teamSize": "예: 4명, 5명",
        "roles": "예: PM, 프론트엔드, 백엔드, 디자이너",
        "dueDate": "예: 6월 말, 2026년 6월 20일",
        "deliverables": "예: 웹 앱, 발표 자료, 시연 영상",
    }
    question = GATHER_FIELD_GUIDE.get(slot, {}).get("question", "")
    if not question:
        return _build_next_missing_field_prompt(current_data)
    example = examples.get(slot)
    if example:
        return f"{question} {example}"
    return question


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
    return _signal_is_help_request(user_message)


def _is_request_like_value(user_message: str) -> bool:
    normalized = _clean_text(user_message)
    return bool(normalized) and is_request_like_value(normalized)


def _is_undecided_value(user_message: str) -> bool:
    normalized = _clean_text(user_message)
    return bool(normalized) and is_undecided_value(normalized)


def _is_guidance_signal(user_message: str) -> bool:
    return _signal_is_guidance_signal(user_message)


def _is_meta_conversation_message(user_message: str) -> bool:
    return _signal_is_meta_conversation_message(user_message)


def _is_non_storable_freeform_message(user_message: str) -> bool:
    normalized = _clean_text(user_message)
    if not normalized:
        return False
    return (
        _is_request_like_value(normalized)
        or _is_undecided_value(normalized)
        or looks_like_non_committal_value(normalized)
        or _is_guidance_signal(normalized)
        or _is_meta_conversation_message(normalized)
    )


def _is_storage_control_message(
    user_message: str,
    current_data: dict[str, object] | None = None,
) -> bool:
    normalized = _clean_text(user_message)
    if not normalized:
        return False
    if _matches_topic_presence_button_message(normalized):
        return True
    if _is_summary_request(normalized):
        return True
    if _is_fill_remaining_request(normalized, current_data or {}):
        return True
    return _is_non_storable_freeform_message(normalized)


def _should_offer_topic_guidance(
    current_data: dict[str, object],
    user_message: str,
    *,
    direct_updates: dict[str, object] | None = None,
) -> bool:
    direct_updates = direct_updates or {}
    topic_anchor = _get_topic_anchor(current_data, allow_title_fallback=False) or str(
        direct_updates.get("subject") or direct_updates.get("title") or ""
    ).strip()
    if not topic_anchor:
        return False
    if direct_updates and not (
        direct_updates.get("subject") or direct_updates.get("title")
    ):
        return False
    if _is_valid_collected_value("goal", current_data.get("goal")):
        return False
    if not (
        _is_guidance_signal(user_message)
        or _is_undecided_value(user_message)
        or _is_request_like_value(user_message)
        or any(keyword in _clean_text(user_message) for keyword in ("모르", "추천", "같이 정", "같이 좁혀"))
    ):
        return False
    return True


def _should_offer_goal_guidance(
    current_data: dict[str, object],
    user_message: str,
    *,
    direct_updates: dict[str, object] | None = None,
) -> bool:
    direct_updates = direct_updates or {}
    topic_anchor = _get_topic_anchor(current_data, allow_title_fallback=False) or str(
        direct_updates.get("subject") or direct_updates.get("title") or ""
    ).strip()
    if not topic_anchor:
        return False
    if _is_valid_collected_value("goal", current_data.get("goal")):
        return False
    if direct_updates.get("goal"):
        return False

    normalized = _clean_text(user_message)
    asks_goal = _detect_requested_gather_focus(normalized) == "goal" or any(
        keyword in normalized for keyword in ("목표", "최종적으로", "무엇을 만들", "결과물")
    )
    wants_help = (
        _is_guidance_signal(user_message)
        or _is_undecided_value(user_message)
        or _is_request_like_value(user_message)
        or _is_help_request(user_message)
    )
    return asks_goal and wants_help


def _build_problem_definition_prompt(subject: str) -> str:
    return (
        f"좋아요. 주제 방향은 '{subject}'로 정리해둘게요. "
        f"{_build_contextual_slot_question({'subject': subject}, 'problemArea')}"
    )


GUIDED_PROMPT_SLOT_META: dict[str, dict[str, str]] = {
    "problemArea": {
        "intro": "좋아요. '{topic_anchor}' 기준으로 어떤 문제를 먼저 풀지 같이 좁혀볼게요.",
        "cta": "번호로 답하거나, 더 맞는 방향이 있다면 한 줄로 바로 말해 주세요.",
        "fallback": "좋아요. '{topic_anchor}' 기준으로 사용자가 가장 먼저 겪는 불편을 한 줄로 말해 주세요.",
    },
    "goal": {
        "intro": "좋아요. '{topic_anchor}' 기준이면 목표는 이렇게 잡아볼 수 있어요.",
        "cta": "가장 가까운 번호 하나를 고르거나, 원하는 방향으로 한 줄 수정해서 말해 주세요.",
        "fallback": "좋아요. '{topic_anchor}' 기준으로 이 프로젝트가 만들고 싶은 변화를 한 줄로 말해 주세요.",
    },
    "roles": {
        "intro": "좋아요. '{topic_anchor}' 기준으로 필요한 역할은 이렇게 생각해볼 수 있어요.",
        "cta": "가까운 번호를 고르거나, 필요한 역할 후보를 직접 적어 주세요.",
        "fallback": "좋아요. '{topic_anchor}' 기준으로 필요한 역할 후보만 먼저 적어 주세요.",
    },
    "deliverables": {
        "intro": "좋아요. '{topic_anchor}' 기준으로 산출물은 이렇게 정리해볼 수 있어요.",
        "cta": "가까운 번호를 고르거나, 만들 결과물을 직접 적어 주세요.",
        "fallback": "좋아요. '{topic_anchor}' 기준으로 최종 산출물을 한 줄로 적어 주세요.",
    },
}


def _build_contextual_slot_question(
    current_data: dict[str, object],
    slot: str,
    *,
    current_phase: str = "GATHER",
) -> str:
    context = build_decision_context(
        current_data,
        current_phase=current_phase,
        prompted_slot=slot,
    )
    topic_anchor = _clean_text(context.get("topic_anchor"))
    problem_context = dict(context.get("problem_definition_context") or {})
    problem_area = _clean_text(problem_context.get("problemArea"))
    target_user = _clean_text(problem_context.get("targetUser"))
    target_facility = _clean_text(problem_context.get("targetFacility"))

    if slot == "subject":
        return "최근에 불편했던 점이나 만들고 싶은 아이디어를 한 줄로 말해 주세요."
    if slot == "problemArea":
        if topic_anchor and target_facility:
            return f"좋아요. '{topic_anchor}' 중 '{target_facility}' 기준으로 어떤 문제를 해결하고 싶은지, 사용자가 가장 먼저 겪는 문제를 한 줄로 말해 주세요."
        if topic_anchor:
            return f"좋아요. '{topic_anchor}'에서 어떤 문제를 해결하고 싶은지, 사용자가 가장 먼저 겪는 문제를 한 줄로 말해 주세요."
        return "어떤 문제를 해결하고 싶은지, 사용자가 가장 먼저 겪는 문제를 한 줄로 말해 주세요."
    if slot == "goal":
        if topic_anchor and problem_area:
            return f"좋아요. '{topic_anchor}'에서 '{problem_area}' 문제를 줄이기 위해 만들고 싶은 변화를 한 줄로 말해 주세요."
        if topic_anchor:
            return f"좋아요. '{topic_anchor}' 기준으로 이 프로젝트가 만들고 싶은 변화를 한 줄로 말해 주세요."
        return "이 프로젝트가 만들고 싶은 변화를 한 줄로 말해 주세요."
    if slot == "roles":
        if topic_anchor and problem_area:
            return f"좋아요. '{topic_anchor}'에서 '{problem_area}' 문제를 풀려면 어떤 역할이 필요할지 후보만 먼저 적어 주세요."
        if topic_anchor:
            return f"좋아요. '{topic_anchor}' 기준으로 필요한 역할 후보를 먼저 적어 주세요."
        return "필요한 역할 후보를 먼저 적어 주세요."
    if slot == "deliverables":
        if topic_anchor and problem_area:
            return f"좋아요. '{topic_anchor}'에서 '{problem_area}' 문제를 풀기 위해 최종적으로 만들 산출물을 한 줄로 적어 주세요."
        if topic_anchor:
            return f"좋아요. '{topic_anchor}' 기준으로 최종 산출물을 한 줄로 적어 주세요."
        return "최종 산출물을 한 줄로 적어 주세요."
    if slot == "dueDate":
        if topic_anchor:
            return f"좋아요. '{topic_anchor}' 기준으로 마감일이나 발표일을 한 줄로 적어 주세요."
        return "마감일이나 발표일을 한 줄로 적어 주세요."
    if slot == "teamSize":
        if topic_anchor:
            return f"좋아요. '{topic_anchor}'를 진행하는 팀 인원을 몇 명으로 생각하는지 적어 주세요. 예: 4명, 5명"
        return "팀 인원을 몇 명으로 생각하는지 적어 주세요. 예: 4명, 5명"
    if slot == "title":
        if topic_anchor:
            return f"좋아요. 지금까지 정한 '{topic_anchor}' 방향을 바탕으로 프로젝트 제목을 한 줄로 적어 주세요."
        return "프로젝트 제목을 한 줄로 적어 주세요."
    if slot == "targetUser":
        if topic_anchor:
            return f"좋아요. '{topic_anchor}' 문제를 가장 먼저 겪는 대상을 한 줄로 적어 주세요."
        return "이 문제를 가장 먼저 겪는 대상을 한 줄로 적어 주세요."
    return GATHER_FIELD_GUIDE.get(slot, {}).get("question", "")


def _coerce_guided_prompt_result(
    raw_result: object,
    *,
    target_slot: str,
    topic_anchor: str,
) -> dict[str, object]:
    fallback_question = _build_next_missing_field_prompt(
        {"subject": topic_anchor} if topic_anchor else {},
        current_phase="GATHER",
    )
    if not isinstance(raw_result, dict):
        return {
            "slot": target_slot,
            "question": "",
            "options": [],
            "fallback_question": fallback_question,
            "generation_reason": "invalid_llm_payload",
        }

    options: list[str] = []
    if isinstance(raw_result.get("options"), list):
        for item in raw_result["options"]:
            candidate = _clean_text(item)
            if (
                not candidate
                or candidate in options
                or _looks_like_question_line(candidate)
                or _looks_like_multi_option_block(candidate)
            ):
                continue
            options.append(candidate)
            if len(options) >= 4:
                break

    return {
        "slot": target_slot,
        "question": _clean_text(raw_result.get("question")),
        "options": options,
        "fallback_question": _clean_text(raw_result.get("fallback_question")) or fallback_question,
        "generation_reason": _clean_text(raw_result.get("generation_reason")) or "llm_generated",
    }


def _build_guided_prompt_fallback_payload(
    *,
    target_slot: str,
    topic_anchor: str,
) -> dict[str, object]:
    fallback_template = GUIDED_PROMPT_SLOT_META.get(target_slot, {}).get("fallback", "")
    fallback_question = (
        fallback_template.format(topic_anchor=topic_anchor)
        if fallback_template and topic_anchor
        else GATHER_FIELD_GUIDE.get(target_slot, {}).get("question", "")
    )
    return {
        "slot": target_slot,
        "question": "",
        "options": [],
        "fallback_question": fallback_question,
        "generation_reason": "llm_failed_fallback",
    }


def _build_guided_prompt_prompt(
    *,
    context: dict[str, object],
    target_slot: str,
) -> str:
    topic_anchor = _clean_text(context.get("topic_anchor")) or "현재 주제"
    return f"""
    You are assisting a team-project decision support workflow.
    Respond in Korean and output JSON only.

    Goal:
    - Generate a short follow-up prompt for the next decision slot.
    - Base the suggestions on the current confirmed facts.
    - Do not reuse canned domain templates or fixed category lists.
    - Suggest only options that fit the current topic and problem context.
    - If context is weak, return options as an empty array and write a direct fallback_question.

    Output JSON schema:
    {{
      "slot": "{target_slot}",
      "question": "short guidance question",
      "options": ["specific option 1", "specific option 2", "specific option 3"],
      "fallback_question": "short direct question",
      "generation_reason": "why these options fit"
    }}

    Rules:
    - Keep options to 2-4 items.
    - Each option must be concise and specific to the topic.
    - Avoid broad labels unless the current topic clearly implies them.
    - Never output facts as already decided.
    - Never rewrite subject/title in this step.

    [Topic anchor]
    {topic_anchor}

    [Decision context]
    {json.dumps(context, ensure_ascii=False)}
    """


def _generate_guided_prompt_payload(
    *,
    current_data: dict[str, object],
    target_slot: str,
    current_phase: str = "GATHER",
    recent_messages: list[str] | None = None,
    user_message: str = "",
) -> dict[str, object]:
    context = build_decision_context(
        current_data,
        current_phase=current_phase,
        prompted_slot=target_slot,
        recent_messages=recent_messages,
        user_message=user_message,
    )
    logger.info(
        "guided_prompt_context target_slot=%s phase=%s confirmed=%s open_slots=%s problem_context=%s recent_user_message=%r",
        target_slot,
        context.get("effective_phase"),
        context.get("confirmed_facts"),
        context.get("open_slots"),
        context.get("problem_definition_context"),
        context.get("recent_user_message"),
    )

    topic_anchor = _clean_text(context.get("topic_anchor"))
    if not topic_anchor and target_slot in {"problemArea", "goal"}:
        return _build_guided_prompt_fallback_payload(
            target_slot=target_slot,
            topic_anchor=topic_anchor,
        )

    try:
        response = _invoke_llm(
            conversation_llm,
            _build_guided_prompt_prompt(context=context, target_slot=target_slot),
            label="guided_prompt_generator",
        )
        payload = _coerce_guided_prompt_result(
            json.loads(getattr(response, "content", "")),
            target_slot=target_slot,
            topic_anchor=topic_anchor,
        )
    except Exception as exc:
        logger.info(
            "guided_prompt_fallback target_slot=%s reason=llm_failed error=%s",
            target_slot,
            exc,
        )
        payload = _build_guided_prompt_fallback_payload(
            target_slot=target_slot,
            topic_anchor=topic_anchor,
        )

    logger.info(
        "guided_prompt_generated target_slot=%s options=%s generation_reason=%s fallback_question=%r",
        target_slot,
        payload.get("options"),
        payload.get("generation_reason"),
        payload.get("fallback_question"),
    )
    return payload


def _format_guided_prompt_message(
    payload: dict[str, object],
    *,
    current_data: dict[str, object],
    target_slot: str,
) -> str:
    topic_anchor = _get_topic_anchor(current_data, allow_title_fallback=False) or "이 주제"
    slot_meta = GUIDED_PROMPT_SLOT_META.get(target_slot, {})
    intro = slot_meta.get("intro", "").format(topic_anchor=topic_anchor).strip()
    question = _clean_text(payload.get("question"))
    options = [
        _clean_text(option)
        for option in (payload.get("options") or [])
        if _clean_text(option)
    ]
    if not options:
        return _clean_text(payload.get("fallback_question")) or _build_next_missing_field_prompt(
            current_data
        )

    lines: list[str] = []
    if intro:
        lines.append(intro)
    if question and question not in intro:
        lines.append(question)
    lines.append("\n".join(f"{index}. {option}" for index, option in enumerate(options, start=1)))
    cta = slot_meta.get("cta", "").strip()
    if cta:
        lines.append(cta)
    return "\n".join(line for line in lines if line).strip()


def _build_guided_prompt_message(
    *,
    current_data: dict[str, object],
    target_slot: str,
    current_phase: str = "GATHER",
    recent_messages: list[str] | None = None,
    user_message: str = "",
) -> str:
    payload = _generate_guided_prompt_payload(
        current_data=current_data,
        target_slot=target_slot,
        current_phase=current_phase,
        recent_messages=recent_messages,
        user_message=user_message,
    )
    return _format_guided_prompt_message(
        payload,
        current_data=current_data,
        target_slot=target_slot,
    )


def _build_topic_refinement_message(current_data: dict[str, object]) -> str:
    return _build_guided_prompt_message(
        current_data=current_data,
        target_slot="problemArea",
    )


def _build_goal_guidance_message(current_data: dict[str, object]) -> str:
    return _build_guided_prompt_message(
        current_data=current_data,
        target_slot="goal",
    )


def _build_slot_help_message(
    slot: str,
    current_data: dict[str, object],
    *,
    current_phase: str = "GATHER",
    recent_messages: list[str] | None = None,
    user_message: str = "",
) -> str:
    if slot in {"goal", "roles", "deliverables"}:
        return _build_guided_prompt_message(
            current_data=current_data,
            target_slot=slot,
            current_phase=current_phase,
            recent_messages=recent_messages,
            user_message=user_message,
        )
    question = _build_contextual_slot_question(
        current_data,
        slot,
        current_phase=current_phase,
    )
    if question:
        return question
    return _build_next_missing_field_prompt(current_data, current_phase=current_phase)


def _looks_like_title_instruction(candidate: str) -> bool:
    normalized = str(candidate or "").strip().lower()
    if not normalized:
        return True
    meta_request_keywords = (
        "\uc694\uc57d",
        "\uc815\ub9ac\ud574\uc918",
        "\uc815\ub9ac\ud574 \uc918",
        "\ucc44\uc6cc\uc918",
        "\ucc44\uc6cc \uc918",
        "\ubbf8\uc815\uc778 \ud56d\ubaa9",
        "\uc81c\ub300\ub85c \uc815\uc758\ub418\uc9c0 \uc54a\uc740 \ubd80\ubd84",
        "\ub2e4 \ucc44\uc6cc\uc918",
        "\ub0a8\uc740 \ud56d\ubaa9",
        "\uc9c0\uae08 \ubaa8\uc778 \uc815\ubcf4",
        "\uc138\uc158 \uc694\uc57d",
        "fill",
        "summary",
    )
    return any(keyword in normalized for keyword in TITLE_INSTRUCTION_KEYWORDS) or any(
        keyword in normalized for keyword in meta_request_keywords
    )


def _normalize_direct_fact_value(value: str) -> str:
    normalized = DIRECT_FACT_ENDING_PATTERN.sub("", str(value or "").strip())
    return normalized.strip(" .,!?:;\"'")


def _normalize_due_date_candidate(value: str) -> str:
    candidate = _clean_text(value).replace("/", "-").replace(".", "-")
    if not candidate:
        return ""
    candidate = re.sub(r"\s*쯤$", "", candidate).strip()
    short_year_match = re.fullmatch(r"(\d{2})년\s*(초|중|말)", candidate)
    if short_year_match:
        year = int(short_year_match.group(1))
        candidate = f"{2000 + year}년 {short_year_match.group(2)}"
    return candidate


def _extract_due_date_candidate_from_message(message: str) -> str:
    normalized = _clean_text(message)
    if not normalized:
        return ""

    explicit_match = DUE_DATE_PATTERN.search(normalized)
    if explicit_match:
        return _normalize_due_date_candidate(explicit_match.group(1))

    for pattern in KOREAN_DUE_DATE_CANDIDATE_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return _normalize_due_date_candidate(match.group(1))
    return ""


def _recent_message_blocks(state: AgentState) -> list[str]:
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
    return text_blocks


def _get_recent_problem_area_from_context(state: AgentState) -> str:
    stored_problem_area = _clean_text(
        state.get("problem_area")
        or (state.get("collected_data") or {}).get("problemArea")
    )
    if stored_problem_area:
        return stored_problem_area
    recent_messages = [
        _clean_text(MATES_MENTION_PATTERN.sub(" ", str(msg or "")))
        for msg in state.get("recent_messages", [])
        if _clean_text(MATES_MENTION_PATTERN.sub(" ", str(msg or "")))
    ]
    if not recent_messages:
        return ""
    latest_block = recent_messages[-1]
    for pattern in PROBLEM_AREA_CONTEXT_PATTERNS:
        match = pattern.search(latest_block)
        if not match:
            continue
        return _normalize_direct_fact_value(match.group(2))
    return ""


def _is_awaiting_target_facility(state: AgentState) -> bool:
    blocks = _recent_message_blocks(state)
    if not blocks:
        return False
    latest_block = blocks[0]
    return bool(TARGET_FACILITY_PROMPT_PATTERN.search(latest_block))


def _extract_target_facility_candidate(
    state: AgentState,
    current_data: dict[str, object] | None = None,
) -> str:
    current_data = current_data or {}
    if not _get_topic_anchor(current_data, allow_title_fallback=False):
        return ""
    if not _is_awaiting_target_facility(state):
        return ""

    user_message = _effective_user_message(state)
    normalized = _clean_text(_strip_mates_mention(user_message))
    if not normalized:
        return ""
    if _is_request_like_value(normalized) or _is_undecided_value(normalized):
        return ""
    if _is_meta_conversation_message(normalized) or _extract_due_date_candidate_from_message(normalized):
        return ""
    if _extract_choice_index(normalized) is not None:
        return ""

    candidate = normalized
    facility_patterns = (
        re.compile(r"^(.+?)\s*(?:을|를)\s*대상(?:으로)?(?:\s*(?:한다고|할게|하고\s*싶어|으로\s*할게))?$"),
        re.compile(r"^(.+?)\s*대상(?:으로)?(?:\s*(?:한다고|할게|하고\s*싶어))?$"),
    )
    for pattern in facility_patterns:
        match = pattern.match(candidate)
        if match:
            candidate = match.group(1).strip()
            break

    candidate = _normalize_direct_fact_value(candidate).strip(" '\"")
    if not candidate:
        return ""
    compact = re.sub(r"\s+", "", candidate)
    if len(compact) > 12:
        return ""
    if any(token in candidate for token in ("문제", "관리", "개선", "효율", "혼잡", "예약", "접근성", "안내")):
        return ""
    if not TARGET_FACILITY_NOUN_PATTERN.search(candidate):
        return ""
    if candidate == _get_topic_anchor(current_data, allow_title_fallback=False):
        return ""
    if _is_non_storable_freeform_message(candidate):
        return ""
    if _looks_like_title_instruction(candidate):
        return ""
    return candidate


def _extract_problem_area_candidate(
    state: AgentState,
    current_data: dict[str, object] | None = None,
    *,
    direct_updates: dict[str, object] | None = None,
) -> str:
    current_data = current_data or {}
    direct_updates = direct_updates or {}
    topic_anchor = _get_topic_anchor(current_data, allow_title_fallback=False) or str(
        direct_updates.get("subject") or direct_updates.get("title") or ""
    ).strip()
    if not topic_anchor:
        return ""

    user_message = _effective_user_message(state)
    normalized = _clean_text(_strip_mates_mention(user_message))
    if not normalized:
        return ""
    if _is_meta_conversation_message(normalized):
        return ""
    if _has_structured_fact_updates(direct_updates):
        return ""
    if _matches_topic_presence_button_message(normalized):
        return ""
    if _extract_target_facility_candidate(state, current_data):
        return ""
    requested_focus = _detect_requested_gather_focus(normalized)
    if requested_focus in {"goal", "teamSize", "roles", "dueDate", "deliverables"}:
        return ""

    choice_candidate = _extract_choice_based_title(state)
    if choice_candidate:
        recent_context = "\n".join(str(msg or "") for msg in state.get("recent_messages", [])[-4:])
        selected_message = str(state.get("selected_message") or "")
        if "같이 좁혀볼게요" in recent_context or "같이 좁혀볼게요" in selected_message:
            logger.info(
                "guided_choice_interpretation slot=problemArea choice=%s mapped_candidate=%r source=generated_prompt",
                _extract_choice_index(user_message),
                choice_candidate,
            )
            return choice_candidate

    trimmed = _trim_subject_candidate_clause(normalized)
    if trimmed == normalized and not _is_guidance_signal(normalized):
        match = PROBLEM_AREA_PATTERN.search(normalized)
        if match:
            trimmed = match.group(1).strip()

    trimmed = _normalize_direct_fact_value(trimmed)
    if not trimmed:
        return ""
    compact = re.sub(r"\s+", "", trimmed)
    if len(compact) <= 10 and TARGET_FACILITY_NOUN_PATTERN.search(trimmed):
        return ""
    if trimmed == topic_anchor:
        return ""
    if _is_storage_control_message(trimmed):
        return ""
    if _looks_like_title_instruction(trimmed):
        return ""
    if _extract_due_date_candidate_from_message(trimmed):
        return ""
    if _looks_like_open_question(trimmed):
        return ""
    return trimmed


def _build_problem_area_follow_up(topic_anchor: str, problem_area: str) -> str:
    if any(keyword in _clean_text(topic_anchor) for keyword in ("공공시설", "시설", "도서관", "공원", "주민센터", "버스터미널")):
        return (
            f"좋아요. {topic_anchor}의 {problem_area} 문제로 좁혀볼게요. "
            "어떤 시설을 대상으로 하나요? 예: 도서관, 공원, 주민센터, 버스터미널"
        )
    return (
        f"좋아요. {topic_anchor}에서 '{problem_area}' 방향으로 좁혀볼게요. "
        "이 문제를 가장 먼저 겪는 대상이나 상황을 한 줄로 말해 주세요."
    )


def _build_target_facility_follow_up(topic_anchor: str, problem_area: str, target_facility: str) -> str:
    if problem_area:
        return (
            f"좋아요. {topic_anchor}의 {problem_area} 문제를 {target_facility} 기준으로 볼게요. "
            "이용자가 가장 불편한 순간은 언제인가요? "
            "예: 빈자리 확인, 예약 절차, 운영시간 찾기"
        )
    return (
        f"좋아요. {topic_anchor} 중 {target_facility}를 대상으로 볼게요. "
        "이 시설에서 가장 먼저 해결하고 싶은 불편을 한 줄로 적어 주세요."
    )


def _build_problem_area_commit_response(
    *,
    contextual_data: dict[str, object],
    topic_anchor: str,
    problem_area: str,
    current_phase: str,
) -> tuple[str, str, bool]:
    next_phase = derive_phase_from_collected_data(
        contextual_data,
        current_phase=current_phase,
    )
    is_sufficient = _is_template_ready(contextual_data)
    follow_up = _build_next_missing_field_prompt(
        contextual_data,
        current_phase=next_phase,
    )
    base_message = _build_problem_area_follow_up(topic_anchor, problem_area)
    if follow_up and follow_up not in base_message:
        return f"{base_message} {follow_up}".strip(), next_phase, is_sufficient
    return base_message, next_phase, is_sufficient


def _trim_subject_candidate_clause(value: str) -> str:
    candidate = _clean_text(value)
    if not candidate:
        return ""

    clause_patterns = (
        re.compile(r"^(.+?)\s*(?:이고|인데|지만|이며)\s+(.+)$"),
        re.compile(r"^(.+?)[,.!?\n]\s*(.+)$"),
    )
    for pattern in clause_patterns:
        match = pattern.match(candidate)
        if not match:
            continue

        head = match.group(1).strip(" '\"")
        tail = match.group(2).strip()
        if head and (_is_guidance_signal(tail) or _is_non_storable_freeform_message(tail)):
            return head

    return candidate


def _is_summary_request(user_message: str) -> bool:
    return _signal_is_summary_request(user_message)


def _extract_corrected_team_size_from_message(message: str) -> str:
    correction_patterns = (
        r"(\d{1,2})\s*명\s*(?:이|은|는)?\s*아니라\s*(\d{1,2})\s*명",
        r"아니[^0-9]{0,12}(?:팀원|팀 인원|인원)?[^0-9]{0,12}(\d{1,2})\s*명",
    )
    for pattern in correction_patterns:
        match = re.search(pattern, message)
        if not match:
            continue
        if match.lastindex and match.lastindex >= 2:
            return match.group(match.lastindex).strip()
        return match.group(1).strip()
    fallback_counts = re.findall(r"(\d{1,2})\s*명", message)
    if "아니" in message and fallback_counts:
        return fallback_counts[-1].strip()
    return ""


def _extract_team_size_from_message(message: str) -> str:
    corrected_team_size = _extract_corrected_team_size_from_message(message)
    if corrected_team_size:
        return corrected_team_size

    explicit_match = TEAM_SIZE_PATTERN.search(message)
    if explicit_match:
        return explicit_match.group(1).strip()

    explicit_fallback = re.search(r"(?:팀원|팀 인원|인원)\s*(?:은|는|이)?\s*(\d{1,2})\s*명", message)
    if explicit_fallback:
        return explicit_fallback.group(1).strip()

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


def _is_next_step_request(user_message: str) -> bool:
    return _signal_is_next_step_request(user_message)


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
    cleaned = re.sub(r"(?:으로|로)$", "", cleaned).strip()
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
    if not role_match:
        role_match = re.search(r"(?:역할|롤)\s*(?:은|는|:)?\s*(.+)$", message, flags=re.IGNORECASE)
    candidate = role_match.group(1) if role_match else message
    candidate = ROLE_TRAILING_SPLIT_PATTERN.split(candidate, maxsplit=1)[0]
    candidate = re.split(r"[.?!]\s*|다음엔|다음에|그다음", candidate, maxsplit=1)[0]
    candidate = re.split(r"\s+정도(?:로)?\b", candidate, maxsplit=1)[0]
    candidate = candidate.split("\n", 1)[0].strip()

    roles: list[str] = []
    for part in ROLE_SPLIT_PATTERN.split(candidate):
        normalized = _normalize_role_token(part)
        if not _looks_like_role_token(normalized):
            continue
        if normalized not in roles:
            roles.append(normalized)

    if not roles and role_match:
        fallback_tokens = [
            token.strip(" .,!?:;\"'()[]")
            for token in re.split(r"\s*(?:,|/|그리고)\s*", role_match.group(1))
            if token.strip()
        ]
        roles = [token for token in fallback_tokens if _looks_like_role_token(token)]

    normalized_roles = [
        role for role in (normalize_roles(candidate) or []) if _looks_like_role_token(role)
    ]
    if normalized_roles and len(normalized_roles) > len(roles):
        roles = normalized_roles

    return ", ".join(roles)


def _build_collected_data_summary(current_data: dict[str, object]) -> str:
    normalized = _prune_collected_data(current_data)
    labels = {
        "subject": "주제",
        "title": "제목",
        "goal": "목표",
        "teamSize": "팀 인원",
        "roles": "역할",
        "dueDate": "마감일",
        "deliverables": "산출물",
    }
    ordered_keys = ["subject", "title", "goal", "teamSize", "roles", "dueDate", "deliverables"]

    confirmed_parts = []
    for key in ordered_keys:
        if not _is_valid_collected_value(key, normalized.get(key)):
            continue
        value = format_collected_value(key, normalized[key])
        if key == "teamSize":
            value = f"{value}명"
        confirmed_parts.append(f"{labels[key]} {value}")
    missing_keys = set(_shared_missing_collected_fields(normalized))
    missing_parts = [labels[key] for key in ordered_keys if key in missing_keys]

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
    return "아직 확정된 정보는 없습니다. 주제, 목표, 팀 인원, 역할 중 하나부터 정리하면 됩니다."


def _build_next_missing_field_prompt(current_data: dict[str, object]) -> str:
    normalized = _prune_collected_data(current_data)
    subject = str(normalized.get("subject") or "").strip()
    if not subject:
        return "먼저 어떤 분야나 문제를 다루는 프로젝트인지 한 줄로 정해볼까요?"
    if subject_needs_problem_definition(subject):
        return (
            f"좋아요. '{subject}' 쪽으로 갈게요. "
            "이 안에서 어떤 문제를 해결하고 싶은지 한 줄로 말해 주세요."
        )

    for key in ("goal", "dueDate", "deliverables", "roles", "teamSize", "title"):
        if _is_valid_collected_value(key, normalized.get(key)):
            continue
        if key == "title":
            return "다음으로 프로젝트 제목을 한 줄로 정해볼까요?"
        return GATHER_FIELD_GUIDE[key]["question"]
    return ""


def _build_fact_confirmation_message(
    merged_data: dict[str, object], accepted_updates: dict[str, object]
) -> str:
    confirmations: list[str] = []
    if "subject" in accepted_updates and merged_data.get("subject"):
        confirmations.append(f"주제는 '{merged_data['subject']}'")
    if "title" in accepted_updates and merged_data.get("title"):
        confirmations.append(f"제목은 '{merged_data['title']}'")
    if "goal" in accepted_updates and merged_data.get("goal"):
        confirmations.append(f"목표는 '{merged_data['goal']}'")
    if "teamSize" in accepted_updates and merged_data.get("teamSize"):
        confirmations.append(
            f"팀 인원은 {format_collected_value('teamSize', merged_data['teamSize'])}명"
        )
    if "roles" in accepted_updates and merged_data.get("roles"):
        confirmations.append(f"역할은 {format_collected_value('roles', merged_data['roles'])}")
    if "dueDate" in accepted_updates and merged_data.get("dueDate"):
        confirmations.append(f"마감일은 {merged_data['dueDate']}")
    if "deliverables" in accepted_updates and merged_data.get("deliverables"):
        confirmations.append(f"산출물은 {merged_data['deliverables']}")

    if not confirmations:
        return _build_collected_data_summary(merged_data)

    message = "좋아요. " + ", ".join(confirmations) + " 정리할게요."
    follow_up = _build_next_missing_field_prompt(merged_data)
    if follow_up:
        message += f" {follow_up}"
    return message


def _finalize_committed_response(
    *,
    turn_type: str,
    merged_data: dict[str, object],
    accepted_updates: dict[str, object],
    ai_message: str,
    is_sufficient: bool,
) -> str:
    if turn_type == "request_summary":
        return _build_collected_data_summary(merged_data)
    if turn_type == "request_fill_missing":
        if is_sufficient:
            return _build_collected_data_summary(merged_data)
        follow_up = _build_next_missing_field_prompt(merged_data)
        if follow_up:
            return follow_up
        return _build_collected_data_summary(merged_data)
    if turn_type == "request_goal_guidance":
        return _build_goal_guidance_message(merged_data)
    if turn_type in {"request_refine_topic", "request_guided_exploration"}:
        subject = str(merged_data.get("subject") or "").strip()
        if subject and not subject_needs_problem_definition(subject):
            follow_up = _build_next_missing_field_prompt(merged_data)
            if follow_up:
                return follow_up
        return _build_topic_refinement_message(merged_data)
    if turn_type == "request_next_step":
        if is_sufficient:
            return _build_collected_data_summary(merged_data)
        follow_up = _build_next_missing_field_prompt(merged_data)
        if follow_up:
            return follow_up
        return _build_collected_data_summary(merged_data)
    if turn_type in {"provide_fact", "provide_topic"} and accepted_updates:
        return _build_fact_confirmation_message(merged_data, accepted_updates)
    if not accepted_updates and _looks_like_commit_confirmation(ai_message):
        follow_up = _build_next_missing_field_prompt(merged_data)
        if follow_up:
            return follow_up
        return _build_collected_data_summary(merged_data)
    if not is_sufficient and _looks_like_template_ready_claim(ai_message):
        follow_up = _build_next_missing_field_prompt(merged_data)
        summary = _build_collected_data_summary(merged_data)
        return f"{summary} {follow_up}".strip() if follow_up else summary
    return ai_message


def _build_next_missing_field_prompt(
    current_data: dict[str, object],
    *,
    current_phase: str = "GATHER",
    followup_fields: list[str] | None = None,
    rejected_updates: dict[str, object] | None = None,
) -> str:
    normalized = _prune_collected_data(current_data)
    subject = str(normalized.get("subject") or "").strip()
    if subject and subject_needs_problem_definition(subject) and not _clean_text(
        normalized.get("problemArea")
    ):
        return _build_contextual_slot_question(
            normalized,
            "problemArea",
            current_phase=current_phase,
        )
    next_field = choose_next_question_field(
        normalized,
        current_phase=current_phase,
        followup_fields=followup_fields,
        rejected_updates=rejected_updates,
    )
    if next_field:
        question = _build_contextual_slot_question(
            normalized,
            next_field,
            current_phase=current_phase,
        )
        if question:
            return question
    return ""


def _build_fact_confirmation_message(
    merged_data: dict[str, object],
    accepted_updates: dict[str, object],
    *,
    current_phase: str = "GATHER",
    followup_fields: list[str] | None = None,
    rejected_updates: dict[str, object] | None = None,
) -> str:
    confirmations: list[str] = []
    if "subject" in accepted_updates and merged_data.get("subject"):
        confirmations.append(f"주제는 '{merged_data['subject']}'")
    if "title" in accepted_updates and merged_data.get("title"):
        confirmations.append(f"제목은 '{merged_data['title']}'")
    if "goal" in accepted_updates and merged_data.get("goal"):
        confirmations.append(f"목표는 '{merged_data['goal']}'")
    if "teamSize" in accepted_updates and merged_data.get("teamSize"):
        confirmations.append(
            f"팀 인원은 {format_collected_value('teamSize', merged_data['teamSize'])}명"
        )
    if "roles" in accepted_updates and merged_data.get("roles"):
        confirmations.append(f"역할은 {format_collected_value('roles', merged_data['roles'])}")
    if "dueDate" in accepted_updates and merged_data.get("dueDate"):
        confirmations.append(f"마감일은 {merged_data['dueDate']}")
    if "deliverables" in accepted_updates and merged_data.get("deliverables"):
        confirmations.append(f"산출물은 {merged_data['deliverables']}")

    if not confirmations:
        return _build_collected_data_summary(merged_data)

    message = "좋아요. " + ", ".join(confirmations) + "로 반영할게요."
    follow_up = _build_next_missing_field_prompt(
        merged_data,
        current_phase=current_phase,
        followup_fields=followup_fields,
        rejected_updates=rejected_updates,
    )
    if follow_up:
        message += f" {follow_up}"
    return message


def _build_rejection_response(
    merged_data: dict[str, object],
    *,
    rejected_updates: dict[str, object],
    rejected_reasons: dict[str, str],
    current_phase: str,
    followup_fields: list[str] | None = None,
    ai_message: str = "",
) -> str:
    labels = {
        "subject": "주제",
        "title": "제목",
        "goal": "목표",
        "teamSize": "팀 인원",
        "roles": "역할",
        "dueDate": "마감일",
        "deliverables": "산출물",
    }
    key = next(iter(rejected_reasons), "")
    reason = rejected_reasons.get(key, "")
    label = labels.get(key, key or "이 내용")

    if reason in {
        "turn_type_blocks_storage",
        "subject_request_like_not_storable",
        "goal_request_like_not_storable",
        "title_request_like_not_storable",
    }:
        message = f"방금 말씀은 요청으로 이해돼서 {label}로 바로 저장하진 않았어요."
    elif reason in {"title_requires_explicit_confirmation", "title_requires_recent_context"}:
        message = "제목은 명시적으로 정하거나 직전 제안을 확정한 경우에만 반영할게요."
    elif reason in {
        "overwrite_requires_explicit_correction",
        "protected_field_requires_explicit_or_strong_restatement",
        "guarded_field_kept_existing_value",
    }:
        message = f"기존 {label}은 수정 의도가 분명할 때만 바꿀게요."
    else:
        message = f"{label}은 이번 턴에 바로 저장하지 않았어요."

    if ai_message and not _looks_like_commit_confirmation(ai_message):
        message = f"{message} {ai_message}".strip()
    follow_up = _build_next_missing_field_prompt(
        merged_data,
        current_phase=current_phase,
        followup_fields=followup_fields,
        rejected_updates=rejected_updates,
    )
    if follow_up:
        message = f"{message} {follow_up}".strip()
    return message


def _finalize_committed_response(
    *,
    turn_type: str,
    merged_data: dict[str, object],
    accepted_updates: dict[str, object],
    ai_message: str,
    is_sufficient: bool,
    current_phase: str = "GATHER",
    rejected_updates: dict[str, object] | None = None,
    rejected_reasons: dict[str, str] | None = None,
    followup_fields: list[str] | None = None,
) -> str:
    if turn_type == "request_summary":
        return _build_collected_data_summary(merged_data)
    if turn_type == "request_fill_missing":
        if is_sufficient:
            return _build_collected_data_summary(merged_data)
        follow_up = _build_next_missing_field_prompt(
            merged_data,
            current_phase=current_phase,
            followup_fields=followup_fields,
            rejected_updates=rejected_updates,
        )
        if follow_up:
            return follow_up
        return _build_collected_data_summary(merged_data)
    if turn_type == "request_goal_guidance":
        return _build_goal_guidance_message(merged_data)
    if turn_type == "request_help_needed":
        slot = ""
        if followup_fields:
            slot = followup_fields[0]
        if not slot:
            slot = choose_next_question_field(
                merged_data,
                current_phase=current_phase,
                followup_fields=followup_fields,
                rejected_updates=rejected_updates,
            )
        return _build_slot_help_message(
            slot,
            merged_data,
            current_phase=current_phase,
        )
    if turn_type in {"request_refine_topic", "request_guided_exploration"}:
        subject = str(merged_data.get("subject") or "").strip()
        if subject and not subject_needs_problem_definition(subject):
            follow_up = _build_next_missing_field_prompt(
                merged_data,
                current_phase=current_phase,
                followup_fields=followup_fields,
                rejected_updates=rejected_updates,
            )
            if follow_up:
                return follow_up
        return _build_topic_refinement_message(merged_data)
    if turn_type == "request_next_step":
        if is_sufficient:
            return _build_collected_data_summary(merged_data)
        follow_up = _build_next_missing_field_prompt(
            merged_data,
            current_phase=current_phase,
            followup_fields=followup_fields,
            rejected_updates=rejected_updates,
        )
        if follow_up:
            return follow_up
        return _build_collected_data_summary(merged_data)
    if turn_type in {"provide_fact", "provide_topic"} and accepted_updates:
        return _build_fact_confirmation_message(
            merged_data,
            accepted_updates,
            current_phase=current_phase,
            followup_fields=followup_fields,
            rejected_updates=rejected_updates,
        )
    if turn_type in {"provide_fact", "provide_topic"} and rejected_reasons:
        return _build_rejection_response(
            merged_data,
            rejected_updates=rejected_updates or {},
            rejected_reasons=rejected_reasons,
            current_phase=current_phase,
            followup_fields=followup_fields,
            ai_message=ai_message,
        )
    if not accepted_updates and _looks_like_commit_confirmation(ai_message):
        follow_up = _build_next_missing_field_prompt(
            merged_data,
            current_phase=current_phase,
            followup_fields=followup_fields,
            rejected_updates=rejected_updates,
        )
        if follow_up:
            return follow_up
        return _build_collected_data_summary(merged_data)
    if not is_sufficient and _looks_like_template_ready_claim(ai_message):
        follow_up = _build_next_missing_field_prompt(
            merged_data,
            current_phase=current_phase,
            followup_fields=followup_fields,
            rejected_updates=rejected_updates,
        )
        summary = _build_collected_data_summary(merged_data)
        return f"{summary} {follow_up}".strip() if follow_up else summary
    return ai_message


SOURCE_BASE_WEIGHTS = {
    "direct_structured": 3,
    "direct_heuristic": 2,
    "llm": 2,
}


def _serialize_decision(decision: CandidateDecision) -> dict[str, object]:
    return {
        "approved": decision.approved,
        "normalized_value": decision.normalized_value,
        "reason": decision.reason,
        "overwrite_mode": decision.overwrite_mode,
        "source": decision.source,
        "requires_followup_question": decision.requires_followup_question,
        "conflict_severity": decision.conflict_severity,
    }


def _source_weight_for_field(key: str, source: str) -> int:
    weight = SOURCE_BASE_WEIGHTS.get(source, 1)
    policy = FIELD_POLICY.get(key, {})
    source_bias = str(policy.get("source_bias") or "mixed")

    if source_bias == "structured":
        if source == "direct_structured":
            weight += 2
        elif source == "direct_heuristic":
            weight += 1
        elif source == "llm":
            weight -= 1
    elif source_bias == "context":
        if source == "llm":
            weight += 2
        elif source == "direct_heuristic":
            weight += 1
    elif source_bias == "mixed":
        if source in {"direct_structured", "llm"}:
            weight += 1

    return weight


def _candidate_score(
    *,
    key: str,
    source: str,
    value: object,
    user_message: str,
    current_data: dict[str, object],
) -> tuple[int, int, int]:
    overwrite_mode = evaluate_candidate_update(
        key=key,
        current_value=current_data.get(key),
        incoming_value=value,
        source=source,
        user_message=user_message,
        current_phase="",
        current_data=current_data,
        candidate_updates={key: value},
    ).overwrite_mode
    overwrite_rank = {"NONE": 0, "STRONG_RESTATEMENT": 1, "EXPLICIT": 2}.get(overwrite_mode, 0)
    shape_bonus = 1 if key in {"teamSize", "roles", "dueDate"} else 0
    return (_source_weight_for_field(key, source), overwrite_rank, shape_bonus)


def _merge_candidate_sources(
    *,
    current_data: dict[str, object],
    user_message: str,
    direct_updates: dict[str, object],
    llm_updates: dict[str, object],
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    merged_sources: dict[str, dict[str, object]] = {}

    for key, value in direct_updates.items():
        merged_sources[key] = {"value": value, "source": "direct_structured"}

    for key, value in llm_updates.items():
        candidate = {"value": value, "source": "llm"}
        existing = merged_sources.get(key)
        if existing is None:
            merged_sources[key] = candidate
            continue
        if _candidate_score(
            key=key,
            source=candidate["source"],
            value=candidate["value"],
            user_message=user_message,
            current_data=current_data,
        ) > _candidate_score(
            key=key,
            source=str(existing["source"]),
            value=existing["value"],
            user_message=user_message,
            current_data=current_data,
        ):
            merged_sources[key] = candidate

    merged_values = {
        key: payload["value"]
        for key, payload in merged_sources.items()
    }
    return merged_sources, merged_values


def _evaluate_candidate_updates(
    *,
    current_data: dict[str, object],
    candidate_sources: dict[str, dict[str, object]],
    candidate_updates: dict[str, object],
    user_message: str,
    current_phase: str,
    turn_type: str,
    recent_messages: list[str] | None = None,
    selected_message: str | None = None,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, str],
    dict[str, CandidateDecision],
    list[str],
]:
    _, audit = apply_collected_data_updates(
        current=current_data,
        candidate=candidate_updates,
        turn_type=turn_type,
        current_status=current_phase,
        recent_messages=recent_messages,
        selected_message=selected_message,
        user_message=user_message,
        candidate_sources=candidate_sources,
    )
    return (
        dict(audit.get("approved", {})),
        dict(audit.get("rejected", {})),
        dict(audit.get("rejected_reasons", {})),
        dict(audit.get("decisions", {})),
        list(audit.get("needs_confirmation", [])),
    )


def _log_collected_data_trace(
    state: AgentState,
    *,
    raw_collected_data: dict[str, object],
    normalized_collected_data: dict[str, object],
    direct_updates_raw: dict[str, object],
    llm_updates_raw: dict[str, object],
    candidate_updates_merged: dict[str, object],
    approved_updates: dict[str, object],
    rejected_updates: dict[str, object],
    rejected_reasons: dict[str, str],
    decisions: dict[str, CandidateDecision] | None = None,
) -> None:
    logger.info(
        "collected_data_trace project_id=%s phase=%s action=%s before=%s after=%s",
        state.get("project_id"),
        state.get("current_phase"),
        state.get("action_type"),
        normalized_collected_data,
        merge_collected_data(normalized_collected_data, approved_updates),
    )
    all_fields = []
    for source_payload in (direct_updates_raw, llm_updates_raw, candidate_updates_merged):
        for key in source_payload.keys():
            if key not in all_fields:
                all_fields.append(key)
    for key in all_fields:
        decision = decisions.get(key) if decisions is not None else None
        source = "unknown"
        if key in candidate_updates_merged:
            source = "merged"
        if key in llm_updates_raw:
            source = "llm"
        if key in direct_updates_raw:
            source = "direct"
        logger.info(
            "candidate_trace field=%s candidate=%r decision=%s reason=%s source=%s",
            key,
            candidate_updates_merged.get(key, direct_updates_raw.get(key, llm_updates_raw.get(key))),
            "approved" if key in approved_updates else "rejected" if key in rejected_reasons else "ignored",
            rejected_reasons.get(key) or (decision.reason if decision is not None else ""),
            source,
        )


def _find_role_team_size_conflict(
    current_data: dict[str, object],
    candidate_updates: dict[str, object],
) -> str:
    effective_team_size = candidate_updates.get("teamSize", current_data.get("teamSize"))
    if classify_role_team_size_conflict(candidate_updates.get("roles"), effective_team_size).value == "NONE":
        return ""
    return build_role_team_size_conflict_message(
        candidate_updates.get("roles"),
        effective_team_size,
    )


def _decision_conflict_message(
    current_data: dict[str, object],
    candidate_updates: dict[str, object],
    decisions: dict[str, CandidateDecision],
) -> str:
    if not any(
        decision.conflict_severity != "NONE" for decision in decisions.values()
    ):
        return ""
    effective_roles = candidate_updates.get("roles", current_data.get("roles"))
    effective_team_size = candidate_updates.get("teamSize", current_data.get("teamSize"))
    return build_role_team_size_conflict_message(effective_roles, effective_team_size)


def _looks_like_template_ready_claim(message: str) -> bool:
    normalized = _clean_text(message)
    if not normalized:
        return False
    template_keywords = ("템플릿", "template", "생성", "만들")
    ready_keywords = ("충분", "모였", "만들 수", "준비", "ready")
    return any(keyword in normalized for keyword in template_keywords) and any(
        keyword in normalized for keyword in ready_keywords
    )


def _looks_like_commit_confirmation(message: str) -> bool:
    normalized = _clean_text(message)
    if not normalized:
        return False
    commit_keywords = ("기록", "반영", "저장", "업데이트")
    commit_phrases = ("정리할게", "정리해둘게", "확정할게", "남겨둘게")
    return any(keyword in normalized for keyword in commit_keywords) or any(
        phrase in normalized for phrase in commit_phrases
    )


def _coerce_gather_llm_result(raw_result: object) -> dict[str, object]:
    if not isinstance(raw_result, dict):
        return {
            "intent": "general",
            "ai_message": "",
            "raw_updated_data": {},
            "updated_data": {},
            "is_sufficient": False,
        }

    raw_intent = str(raw_result.get("intent") or "general").strip()
    if raw_intent not in {
        "answer_fact",
        "ask_advice",
        "ask_idea",
        "ask_summary",
        "uncertain",
        "frustrated",
        "general",
    }:
        raw_intent = "general"

    raw_updates = raw_result.get("updated_data")
    if not isinstance(raw_updates, dict):
        raw_updates = {}
    sanitized_updates = sanitize_llm_updated_data(raw_updates)
    return {
        "intent": raw_intent,
        "ai_message": str(raw_result.get("ai_message") or "").strip(),
        "raw_updated_data": raw_updates,
        "updated_data": sanitized_updates,
        "is_sufficient": bool(raw_result.get("is_sufficient", False)),
    }


def _trim_option_description(candidate: str) -> str:
    text = _clean_text(candidate)
    if not text:
        return ""

    parts = re.split(r"\s+[—-]\s+", text, maxsplit=1)
    if len(parts) == 2 and len(parts[0].strip()) >= 2:
        return parts[0].strip()
    return text


def _looks_like_choice_token(value: str) -> bool:
    return _extract_choice_index(value) is not None


def _extract_choice_index(value: object) -> int | None:
    cleaned = _clean_text(_strip_mates_mention(value))
    if not cleaned:
        return None

    exact_match = CHOICE_INDEX_PATTERN.match(cleaned)
    if exact_match:
        return int(exact_match.group(1))

    prefix_match = CHOICE_PREFIX_PATTERN.match(cleaned)
    if prefix_match:
        return int(prefix_match.group(1))

    return None


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
        key for key in GATHER_FIELD_GUIDE if not _is_valid_collected_value(key, current_data.get(key))
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


def _extract_explicit_title_value(message: str) -> str:
    if not TITLE_EXPLICIT_PATTERN.match(message):
        return ""

    explicit_candidate = re.sub(TITLE_EXPLICIT_PATTERN, "", message, count=1).strip()
    if not explicit_candidate:
        return ""
    return _postprocess_explicit_title_value(explicit_candidate)


def _extract_direct_fact_updates(user_message: str) -> dict[str, object]:
    message = _clean_text(_strip_mates_mention(user_message))
    if not message:
        return {}

    updates: dict[str, object] = {}
    lowered_message = message.lower()
    has_explicit_topic_prefix = bool(
        "주제" in message
        or "subject" in lowered_message
        or "topic" in lowered_message
        or TITLE_EXPLICIT_PATTERN.match(message)
    )
    skip_topic_value_extraction = _matches_topic_presence_button_message(message) or (
        _is_storage_control_message(message) and not has_explicit_topic_prefix
    )

    if not skip_topic_value_extraction:
        subject_match = SUBJECT_PATTERN.search(message)
        if subject_match:
            subject = _trim_subject_candidate_clause(subject_match.group(1))
            subject = _normalize_direct_fact_value(subject)
            subject = _normalize_topic_title_with_llm(subject, field_name="subject")
            if subject:
                updates["subject"] = subject
        else:
            explicit_title = _extract_explicit_title_value(message)
            if explicit_title:
                updates["title"] = _normalize_topic_title_with_llm(
                    explicit_title,
                    field_name="title",
                )
            elif TITLE_EXPLICIT_PATTERN.match(message):
                topic_candidate = _normalize_topic_title_with_llm(
                    _extract_topic_candidate(message),
                    field_name="title",
                )
                if topic_candidate and not _looks_like_title_instruction(topic_candidate):
                    updates["title"] = topic_candidate

    team_size = _extract_team_size_from_message(message)
    if team_size:
        updates["teamSize"] = team_size

    roles = _extract_roles_from_message(message)
    if roles:
        updates["roles"] = roles

    due_date_candidate = _extract_due_date_candidate_from_message(message)
    if due_date_candidate:
        updates["dueDate"] = due_date_candidate

    goal_match = GOAL_PATTERN.search(message)
    if goal_match:
        goal_candidate = _normalize_direct_fact_value(goal_match.group(1))
        if goal_candidate and not _is_non_storable_freeform_message(goal_candidate):
            updates["goal"] = goal_candidate

    deliverables_match = DELIVERABLES_PATTERN.search(message)
    if deliverables_match:
        deliverables_candidate = _normalize_direct_fact_value(deliverables_match.group(1))
        if deliverables_candidate and not _is_non_storable_freeform_message(deliverables_candidate):
            updates["deliverables"] = deliverables_candidate

    logger.info(
        "direct_fact_candidates message=%r request_like=%s undecided=%s meta=%s due_date_candidate=%r raw_updates=%s",
        message,
        _is_request_like_value(message),
        _is_undecided_value(message),
        _is_meta_conversation_message(message),
        due_date_candidate,
        updates,
    )

    return _sanitize_gather_updates(updates)


def _extract_choice_based_title(state: AgentState) -> str:
    current_message = _effective_user_message(state)
    choice_index = _extract_choice_index(current_message)
    if choice_index is None:
        return ""
    text_blocks: list[str] = _recent_message_blocks(state)

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


def _extract_choice_based_goal(state: AgentState) -> str:
    current_message = _effective_user_message(state)
    choice_index = _extract_choice_index(current_message)
    if choice_index is None:
        return ""

    for block in _recent_message_blocks(state):
        if "목표는 이렇게 잡을 수 있어요" not in block:
            continue

        matched_line_option = False
        for line in block.splitlines():
            line_match = NUMBERED_OPTION_LINE_PATTERN.match(line.strip())
            if not line_match:
                continue
            matched_line_option = True
            if int(line_match.group(1)) != choice_index:
                continue

            candidate = _normalize_goal_candidate(_trim_option_description(line_match.group(2)))
            if candidate and not _is_non_storable_freeform_message(candidate):
                logger.info(
                    "guided_choice_interpretation slot=goal choice=%s mapped_candidate=%r source=generated_prompt",
                    choice_index,
                    candidate,
                )
                return candidate

        if matched_line_option:
            continue

        for inline_match in NUMBERED_OPTION_INLINE_PATTERN.finditer(block):
            if int(inline_match.group(1)) != choice_index:
                continue

            candidate = _normalize_goal_candidate(_trim_option_description(inline_match.group(2)))
            if candidate and not _is_non_storable_freeform_message(candidate):
                logger.info(
                    "guided_choice_interpretation slot=goal choice=%s mapped_candidate=%r source=generated_prompt",
                    choice_index,
                    candidate,
                )
                return candidate

    return ""


def _sanitize_gather_updates(updated_data: dict[str, object]) -> dict[str, object]:
    sanitized: dict[str, object] = {}
    for key, value in updated_data.items():
        if isinstance(value, str) and key != "teamSize" and _looks_like_choice_token(value):
            continue
        if isinstance(value, str) and key in {"subject", "title", "goal"} and _looks_like_multi_option_block(value):
            continue
        if key in {"subject", "title"} and isinstance(value, str):
            value = _normalize_topic_title(value)
        if isinstance(value, str):
            sanitized[key] = _clean_text(value)
        else:
            sanitized[key] = value
    return _shared_sanitize_candidate_updates(sanitized)


def _extract_confirmed_title_from_context(state: AgentState) -> str:
    user_message = _clean_text(_strip_mates_mention(_effective_user_message(state)))
    if not any(token in user_message for token in ("그 제목", "그 이름", "그걸로")):
        return ""

    blocks = [str(state.get("selected_message") or "")]
    blocks.extend(str(message or "") for message in state.get("recent_messages", [])[-6:])
    for block in reversed(blocks):
        if not block:
            continue
        for pattern in (
            r"'([^']+)'",
            r'"([^"]+)"',
            r"제목\s*제안[:：]?\s*([^\n]+)",
        ):
            match = re.search(pattern, block)
            if not match:
                continue
            candidate = _normalize_topic_title(match.group(1))
            if candidate and not _looks_like_title_instruction(candidate):
                return candidate
    return ""


def _is_current_goal_query(user_message: str) -> bool:
    normalized = _clean_text(_strip_mates_mention(user_message))
    if not normalized:
        return False

    patterns = (
        r"(?:기존|현재|지금|확정된)\s*목표(?:가|는|은)?\s*(?:뭔데|뭐야|뭐였(?:지|어)?|무엇(?:인가요)?|알려줘|말해줘)",
        r"목표(?:가|는|은)?\s*(?:뭐였(?:지|어)?|다시\s*(?:알려줘|말해줘)|기억이\s*안\s*나)",
        r"목표(?:가|는|은)?\s*뭔데",
    )
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns)


def _build_current_goal_response(
    current_data: dict[str, object],
    *,
    current_phase: str,
) -> str:
    goal = str(current_data.get("goal") or "").strip()
    if _is_valid_collected_value("goal", goal):
        return f"현재 목표는 '{goal}'입니다. 수정하려면 '목표는 ...'처럼 바로 말해 주세요."

    follow_up = _build_next_missing_field_prompt(current_data, current_phase=current_phase)
    if follow_up:
        return f"아직 확정된 목표는 없어요. {follow_up}"
    return (
        "아직 확정된 목표는 없어요. "
        + _build_contextual_slot_question(current_data, "goal", current_phase=current_phase)
    )


def _filter_gather_updates(
    user_message: str,
    updated_data: dict[str, object],
    *,
    focus_type: str | None,
) -> dict[str, object]:
    sanitized = _sanitize_gather_updates(updated_data)
    if _is_guidance_signal(user_message):
        topical_updates = {
            key: value
            for key, value in sanitized.items()
            if key in {"subject", "title"}
        }
        return topical_updates
    if focus_type in UNSUPPORTED_GATHER_TOPICS:
        return {}
    return sanitized


def _count_ready_fields(current_data: dict[str, object]) -> int:
    return len(GATHER_FIELD_GUIDE) - len(_shared_missing_collected_fields(current_data))


def _is_template_ready(current_data: dict[str, str]) -> bool:
    return _shared_is_template_ready(current_data)


def _build_topic_exists_fallback_message() -> str:
    return FAST_TOPIC_EXISTS_REPLY


def _build_topic_debug_payload(
    *,
    raw_user_message: object,
    sanitized_user_message: str,
    raw_current_data: dict[str, object],
    current_data: dict[str, object],
    direct_updates: dict[str, object],
    candidate_updates: dict[str, object],
    merged_data: dict[str, object],
) -> dict[str, object]:
    current_candidates = {
        key: str(raw_current_data.get(key) or "").strip()
        for key in ("subject", "title")
        if str(raw_current_data.get(key) or "").strip()
    }
    blocked_candidates = [
        {
            "source": f"collected_data.{key}",
            "value": str(raw_current_data.get(key) or "").strip(),
        }
        for key in ("subject", "title")
        if str(raw_current_data.get(key) or "").strip() and not str(current_data.get(key) or "").strip()
    ]
    latest_message_candidate = _extract_topic_candidate(sanitized_user_message)

    return {
        "raw_user_message": str(raw_user_message or ""),
        "sanitized_user_message": sanitized_user_message,
        "topic_candidates_by_source": {
            "latest_user_message": latest_message_candidate,
            "direct_updates": {
                key: direct_updates.get(key)
                for key in ("subject", "title")
                if direct_updates.get(key)
            },
            "candidate_updates": {
                key: candidate_updates.get(key)
                for key in ("subject", "title")
                if candidate_updates.get(key)
            },
            "current_collected_data": current_candidates,
        },
        "blocked_candidates": blocked_candidates,
        "final_merged_topic_title": {
            key: merged_data.get(key)
            for key in ("subject", "title")
            if merged_data.get(key)
        },
    }


def _split_topic_candidate_sources(
    state: AgentState,
    current_data: dict[str, object],
    *,
    direct_updates_raw: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    raw_topic_updates = {
        key: value
        for key, value in direct_updates_raw.items()
        if key in {"subject", "title"}
    }
    topic_candidates_raw = _extract_title_updates_for_topic_set(
        state,
        current_data,
        direct_updates=direct_updates_raw,
    )
    direct_topic_updates_raw: dict[str, object] = {}
    llm_topic_updates_raw: dict[str, object] = {}
    for key, value in topic_candidates_raw.items():
        if raw_topic_updates.get(key) == value:
            direct_topic_updates_raw[key] = value
        else:
            llm_topic_updates_raw[key] = value
    return (
        raw_topic_updates,
        topic_candidates_raw,
        direct_topic_updates_raw,
        llm_topic_updates_raw,
    )


# ----------------------------------------------------
# 1. 아이디어가 없을 때 (NO 선택) : 탐색 노드
def explore_problem_node(state: AgentState):
    user_message = _effective_user_message(state)
    turn_policy = _get_turn_policy(state)
    latest_intent = _infer_latest_user_intent(state)
    raw_current_data = dict(state.get("collected_data") or {})
    current_data = _prune_collected_data(raw_current_data)
    next_phase = derive_phase_from_collected_data(
        raw_current_data,
        current_phase=str(state.get("current_phase") or "EXPLORE"),
    )
    is_sufficient = _is_template_ready(current_data)

    if _is_initial_button_selection(state):
        return _build_initial_button_reset_response(state)

    if latest_intent == "greeting":
        return {
            "ai_message": _apply_turn_policy_to_message(state, _answer_only_fallback(state, "")),
            "collected_data": raw_current_data,
            "is_sufficient": is_sufficient,
            "next_phase": next_phase,
        }

    if _is_trivial_message(user_message) and not state.get("recent_messages"):
        return {
            "ai_message": _apply_turn_policy_to_message(state, _answer_only_fallback(state, "")),
            "collected_data": raw_current_data,
            "is_sufficient": is_sufficient,
            "next_phase": next_phase,
        }

    if latest_intent in {"request_summary", "request_fill_missing", "request_next_step"}:
        return {
            "ai_message": _apply_turn_policy_to_message(
                state,
                _finalize_committed_response(
                    turn_type=latest_intent,
                    merged_data=current_data,
                    accepted_updates={},
                    ai_message="",
                    is_sufficient=is_sufficient,
                ),
            ),
            "collected_data": raw_current_data,
            "is_sufficient": is_sufficient,
            "next_phase": next_phase,
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
        "collected_data": raw_current_data,
        "is_sufficient": is_sufficient,
        "next_phase": "EXPLORE",  # 계속 탐색 단계 유지
    }


# ----------------------------------------------------
# 1-2. 아이디어가 있을 때 (YES 선택) : 팀 대화 안내 노드
# ----------------------------------------------------
def topic_exists_node(state: AgentState):
    if _is_initial_button_selection(state):
        return _build_initial_button_reset_response(state)

    user_message = _effective_user_message(state)
    raw_current_data = dict(state.get("collected_data") or {})
    current_data = _prune_collected_data(raw_current_data)
    direct_updates_raw = _extract_direct_fact_updates(user_message)
    if "title" not in direct_updates_raw:
        confirmed_title = _extract_confirmed_title_from_context(state)
        if confirmed_title:
            direct_updates_raw["title"] = confirmed_title
    turn_type = _interpret_turn_type(state, current_data, direct_updates=direct_updates_raw)
    prompted_slot = _get_active_prompted_slot(state, current_data)
    problem_area_candidate = _extract_problem_area_candidate(
        state,
        current_data,
        direct_updates=direct_updates_raw,
    )
    (
        raw_topic_updates,
        topic_candidates_raw,
        topic_direct_updates_raw,
        llm_updates_raw,
    ) = _split_topic_candidate_sources(
        state,
        current_data,
        direct_updates_raw=direct_updates_raw,
    )
    candidate_sources, candidate_updates_merged = _merge_candidate_sources(
        current_data=current_data,
        user_message=user_message,
        direct_updates=_shared_sanitize_candidate_updates(
            topic_direct_updates_raw,
            current_data=current_data,
        ),
        llm_updates=_shared_sanitize_candidate_updates(
            llm_updates_raw,
            current_data=current_data,
        ),
    )
    turn_type = _interpret_turn_type(state, current_data, direct_updates=direct_updates_raw)
    approved_updates, rejected_updates, rejected_reasons, decisions, followup_fields = (
        _evaluate_candidate_updates(
            current_data=current_data,
            candidate_sources=candidate_sources,
            candidate_updates=candidate_updates_merged,
            user_message=user_message,
            current_phase="TOPIC_SET",
            turn_type=turn_type,
            recent_messages=state.get("recent_messages", []),
            selected_message=state.get("selected_message"),
        )
    )
    merged_data = merge_collected_data(current_data, approved_updates)
    if (
        approved_updates.get("subject")
        and current_data.get("title")
        and (
            not current_data.get("subject")
            or current_data.get("subject") != approved_updates.get("subject")
        )
        and not approved_updates.get("title")
    ):
        merged_data.pop("title", None)
    accepted_updates = {
        key: value for key, value in merged_data.items() if current_data.get(key) != value
    }
    next_phase = derive_phase_from_collected_data(merged_data, current_phase="TOPIC_SET")
    is_sufficient = _is_template_ready(merged_data)
    topic_debug_payload = _build_topic_debug_payload(
        raw_user_message=state.get("user_message"),
        sanitized_user_message=user_message,
        raw_current_data=raw_current_data,
        current_data=current_data,
        direct_updates=raw_topic_updates,
        candidate_updates=candidate_updates_merged,
        merged_data=merged_data,
    )
    logger.info(
        "topic_exists turn_type=%s user_message=%r before=%s candidates=%s approved=%s rejected=%s after=%s next_phase=%s is_sufficient=%s",
        turn_type,
        user_message,
        current_data,
        candidate_updates_merged,
        approved_updates,
        rejected_updates,
        merged_data,
        next_phase,
        is_sufficient,
    )
    logger.info("topic_resolution %s", topic_debug_payload)
    _log_collected_data_trace(
        state,
        raw_collected_data=raw_current_data,
        normalized_collected_data=current_data,
        direct_updates_raw=raw_topic_updates,
        llm_updates_raw=llm_updates_raw,
        candidate_updates_merged=candidate_updates_merged,
        approved_updates=approved_updates,
        rejected_updates=rejected_updates,
        rejected_reasons=rejected_reasons,
        decisions=decisions,
    )
    extracted_subject = accepted_updates.get("subject", "")
    if extracted_subject:
        broad_subject = subject_needs_problem_definition(extracted_subject)
        wants_guidance = turn_type in {"request_refine_topic", "request_guided_exploration"} or any(
            checker(user_message)
            for checker in (_is_guidance_signal, _is_undecided_value, _is_request_like_value)
        )
        if broad_subject and wants_guidance:
            return {
                "ai_message": (
                    f"좋아요. 주제 방향은 '{extracted_subject}'로 정리해둘게요. "
                    f"{_build_topic_refinement_message(merged_data)}"
                ),
                "collected_data": merged_data,
                "is_sufficient": is_sufficient,
                "next_phase": next_phase,
                "approved_updates": approved_updates,
                "rejected_updates": rejected_updates,
                "rejected_reasons": rejected_reasons,
                "followup_fields": followup_fields,
            }
        if broad_subject:
            return {
                "ai_message": _build_problem_definition_prompt(extracted_subject),
                "collected_data": merged_data,
                "is_sufficient": is_sufficient,
                "next_phase": next_phase,
                "approved_updates": approved_updates,
                "rejected_updates": rejected_updates,
                "rejected_reasons": rejected_reasons,
                "followup_fields": followup_fields,
            }
        return {
            "ai_message": (
                f"좋아요. 주제는 '{extracted_subject}'로 정리할게요. "
                "이제 목표나 결과물을 한두 줄로 말해 주세요."
            ),
            "collected_data": merged_data,
            "is_sufficient": is_sufficient,
            "next_phase": next_phase,
            "approved_updates": approved_updates,
            "rejected_updates": rejected_updates,
            "rejected_reasons": rejected_reasons,
            "followup_fields": followup_fields,
        }
    extracted_title = accepted_updates.get("title", "")
    if extracted_title:
        if turn_type in {"request_refine_topic", "request_guided_exploration"} or any(
            checker(user_message)
            for checker in (_is_guidance_signal, _is_undecided_value, _is_request_like_value)
        ):
            return {
                "ai_message": (
                    f"좋아요. 프로젝트 제목은 '{extracted_title}'로 정리해둘게요. "
                    f"{_build_topic_refinement_message(merged_data)}"
                ),
                "collected_data": merged_data,
                "is_sufficient": is_sufficient,
                "next_phase": next_phase,
                "approved_updates": approved_updates,
                "rejected_updates": rejected_updates,
                "rejected_reasons": rejected_reasons,
                "followup_fields": followup_fields,
            }
        return {
            "ai_message": (
                f"좋아요. 프로젝트 제목은 '{extracted_title}'로 정리해둘게요. "
                "이제 이 프로젝트가 해결하려는 문제나 만들 결과물을 한두 줄로 말해 주세요."
            ),
            "collected_data": merged_data,
            "is_sufficient": is_sufficient,
            "next_phase": next_phase,
            "approved_updates": approved_updates,
            "rejected_updates": rejected_updates,
            "rejected_reasons": rejected_reasons,
            "followup_fields": followup_fields,
        }
    if turn_type == "provide_problem_area" and problem_area_candidate:
        contextual_data = merge_collected_data(
            merged_data,
            {"problemArea": problem_area_candidate},
        )
        topic_anchor = _get_topic_anchor(
            contextual_data or current_data,
            allow_title_fallback=False,
        )
        ai_message, contextual_phase, contextual_sufficient = _build_problem_area_commit_response(
            contextual_data=contextual_data,
            topic_anchor=topic_anchor,
            problem_area=problem_area_candidate,
            current_phase="TOPIC_SET",
        )
        logger.info(
            "topic_exists problem_area topic=%r problem_area=%r before=%s after=%s next_phase=%s",
            topic_anchor,
            problem_area_candidate,
            current_data,
            contextual_data,
            contextual_phase,
        )
        return {
            "ai_message": ai_message,
            "collected_data": contextual_data,
            "is_sufficient": contextual_sufficient,
            "next_phase": contextual_phase,
        }
    if turn_type in {
        "request_summary",
        "request_fill_missing",
        "request_next_step",
        "request_goal_guidance",
        "request_help_needed",
        "request_refine_topic",
        "request_guided_exploration",
        "meta_request",
    }:
        return {
            "ai_message": _apply_turn_policy_to_message(
                state,
                _finalize_committed_response(
                    turn_type="request_refine_topic"
                    if turn_type == "request_guided_exploration"
                    else turn_type,
                    merged_data=current_data,
                    accepted_updates={},
                    ai_message="",
                    is_sufficient=_is_template_ready(current_data),
                    current_phase="TOPIC_SET",
                    followup_fields=[prompted_slot] if turn_type == "request_help_needed" and prompted_slot else [],
                ),
            ),
            "collected_data": raw_current_data,
            "is_sufficient": _is_template_ready(current_data),
            "next_phase": derive_phase_from_collected_data(
                raw_current_data,
                current_phase="TOPIC_SET",
            ),
            "followup_fields": [prompted_slot] if turn_type == "request_help_needed" and prompted_slot else [],
            "next_question_field": prompted_slot if turn_type == "request_help_needed" else None,
        }
    if not _get_topic_anchor(current_data, allow_title_fallback=False) and user_message:
        return {
            "ai_message": (
                "주제 후보를 이해하려면 한 줄로 더 구체적으로 적어주셔야 합니다. "
                "예를 들면 '대학생 팀플 일정 관리 앱'처럼 적어 주세요."
            ),
            "collected_data": raw_current_data,
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
        {turn_type}

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
        "collected_data": merged_data,
        "is_sufficient": is_sufficient,
        "next_phase": next_phase,
    }


# ----------------------------------------------------
# 2. 아이디어가 있을 때 (YES 선택) : 정보 수집 노드 (자연스러운 HMW)
# ----------------------------------------------------
def gather_information_node(state: AgentState):
    if _is_initial_button_selection(state):
        return _build_initial_button_reset_response(state)

    user_message = _effective_user_message(state)
    turn_policy = _get_turn_policy(state)
    current_phase = str(state.get("current_phase") or "")
    raw_current_data = dict(state.get("collected_data") or {})
    current_data = _prune_collected_data(raw_current_data)
    prefilled_data = dict(current_data)
    was_ready = _is_template_ready(current_data)
    if _is_current_goal_query(user_message):
        next_phase = derive_phase_from_collected_data(prefilled_data, current_phase=current_phase)
        is_sufficient = _is_template_ready(prefilled_data)
        return {
            "ai_message": _apply_turn_policy_to_message(
                state,
                _build_current_goal_response(prefilled_data, current_phase=current_phase),
            ),
            "collected_data": raw_current_data,
            "is_sufficient": is_sufficient,
            "next_phase": next_phase,
        }
    requested_next_step = _is_next_step_request(user_message)
    focus_type = _infer_conversation_focus(state)
    prompted_slot = _get_active_prompted_slot(state, prefilled_data)
    contextual_prompted_slot = _get_contextual_prompted_slot(state, prefilled_data)
    direct_updates_raw = {
        key: value
        for key, value in _extract_direct_fact_updates(user_message).items()
        if key in {"subject", "title", "goal", "targetUser", "teamSize", "roles", "dueDate", "deliverables"}
    }
    if (
        prompted_slot == "targetUser"
        and "targetUser" not in direct_updates_raw
        and user_message
        and not _is_non_storable_freeform_message(user_message)
    ):
        direct_updates_raw["targetUser"] = _normalize_direct_fact_value(user_message)
    choice_goal = _extract_choice_based_goal(state) if prompted_slot == "goal" else ""
    if choice_goal:
        direct_updates_raw["goal"] = choice_goal
    if "title" not in direct_updates_raw:
        confirmed_title = _extract_confirmed_title_from_context(state)
        if confirmed_title:
            direct_updates_raw["title"] = confirmed_title
    preflight_target_facility = _extract_target_facility_candidate(state, prefilled_data)
    preflight_problem_area = _extract_problem_area_candidate(
        state,
        prefilled_data,
        direct_updates=direct_updates_raw,
    )
    contextual_slot_updates = (
        {}
        if preflight_target_facility or preflight_problem_area
        else _infer_prompted_slot_updates(
            prompted_slot=contextual_prompted_slot,
            user_message=user_message,
            current_data=prefilled_data,
        )
    )
    for key, value in contextual_slot_updates.items():
        if key == "roles" and value:
            existing_roles = normalize_roles(direct_updates_raw.get("roles"))
            incoming_roles = normalize_roles(value)
            if incoming_roles and len(incoming_roles) > len(existing_roles or []):
                direct_updates_raw[key] = value
            else:
                direct_updates_raw.setdefault(key, value)
            continue
        direct_updates_raw.setdefault(key, value)
    direct_updates = _shared_sanitize_candidate_updates(
        direct_updates_raw,
        current_data=prefilled_data,
    )
    direct_candidate_sources, direct_candidate_updates = _merge_candidate_sources(
        current_data=prefilled_data,
        user_message=user_message,
        direct_updates=direct_updates,
        llm_updates={},
    )
    turn_type = _interpret_turn_type(state, current_data, direct_updates=direct_updates_raw)
    if prompted_slot and not direct_updates_raw and "잘 모르겠" in user_message:
        turn_type = "request_help_needed"
    (
        direct_approved_updates,
        direct_rejected_updates,
        direct_rejected_reasons,
        direct_decisions,
        direct_followup_fields,
    ) = _evaluate_candidate_updates(
        current_data=prefilled_data,
        candidate_sources=direct_candidate_sources,
        candidate_updates=direct_candidate_updates,
        user_message=user_message,
        current_phase=current_phase,
        turn_type=turn_type,
        recent_messages=state.get("recent_messages", []),
        selected_message=state.get("selected_message"),
    )
    target_facility_candidate = _extract_target_facility_candidate(state, prefilled_data)
    recent_problem_area = _get_recent_problem_area_from_context(state)
    direct_conflict_message = _decision_conflict_message(
        prefilled_data,
        {**prefilled_data, **direct_candidate_updates},
        direct_decisions,
    )
    merged_preview = merge_collected_data(prefilled_data, direct_approved_updates)
    if state.get("current_phase") in {"TOPIC_SET", "PROBLEM_DEFINE"} and _get_topic_anchor(
        prefilled_data,
        allow_title_fallback=False,
    ):
        focus_type = focus_type or "goal"
    logger.info(
        "gather turn_type=%s user_message=%r before=%s direct_updates_raw=%s direct_candidates=%s direct_approved=%s direct_rejected=%s target_facility=%r recent_problem_area=%r",
        turn_type,
        user_message,
        prefilled_data,
        direct_updates_raw,
        direct_updates,
        direct_approved_updates,
        direct_rejected_updates,
        target_facility_candidate,
        recent_problem_area,
    )
    if turn_type == "request_help_needed":
        topic_anchor = _get_topic_anchor(prefilled_data, allow_title_fallback=False)
        if topic_anchor and subject_needs_problem_definition(topic_anchor) and not state.get("recent_messages"):
            help_slot = "subject"
            help_message = _build_topic_refinement_message(prefilled_data)
        else:
            help_slot = prompted_slot or choose_next_question_field(
                prefilled_data,
                current_phase=current_phase,
            )
            help_message = _build_slot_help_message(
                help_slot,
                prefilled_data,
                current_phase=current_phase,
                recent_messages=state.get("recent_messages", []),
                user_message=user_message,
            )
        next_phase = derive_phase_from_collected_data(prefilled_data, current_phase=current_phase)
        is_sufficient = _is_template_ready(prefilled_data)
        return {
            "ai_message": _apply_turn_policy_to_message(
                state,
                help_message,
            ),
            "collected_data": raw_current_data,
            "is_sufficient": is_sufficient,
            "next_phase": next_phase,
            "followup_fields": [help_slot] if help_slot else [],
            "next_question_field": help_slot,
        }
    if turn_type == "provide_target_facility" and target_facility_candidate:
        contextual_data = merge_collected_data(
            prefilled_data,
            {
                "problemArea": recent_problem_area,
                "targetFacility": target_facility_candidate,
            },
        )
        next_phase = "GATHER"
        is_sufficient = _is_template_ready(contextual_data)
        logger.info(
            "gather target_facility topic=%r problem_area=%r target_facility=%r before=%s next_phase=%s",
            _get_topic_anchor(prefilled_data, allow_title_fallback=False),
            recent_problem_area,
            target_facility_candidate,
            prefilled_data,
            next_phase,
        )
        return {
            "ai_message": _apply_turn_policy_to_message(
                state,
                _build_target_facility_follow_up(
                    _get_topic_anchor(prefilled_data, allow_title_fallback=False),
                    recent_problem_area,
                    target_facility_candidate,
                ),
            ),
            "collected_data": contextual_data,
            "is_sufficient": is_sufficient,
            "next_phase": next_phase,
        }
    if turn_type == "request_summary":
        next_phase = derive_phase_from_collected_data(prefilled_data, current_phase=current_phase)
        is_sufficient = _is_template_ready(prefilled_data)
        logger.info(
            "gather summary before=%s next_phase=%s is_sufficient=%s",
            prefilled_data,
            next_phase,
            is_sufficient,
        )
        return {
            "ai_message": _apply_turn_policy_to_message(
                state,
                _finalize_committed_response(
                    turn_type=turn_type,
                    merged_data=prefilled_data,
                    accepted_updates={},
                    ai_message="",
                    is_sufficient=is_sufficient,
                ),
            ),
            "collected_data": raw_current_data,
            "is_sufficient": is_sufficient,
            "next_phase": next_phase,
        }
    if turn_type == "request_fill_missing" and not direct_updates:
        next_phase = derive_phase_from_collected_data(prefilled_data, current_phase=current_phase)
        is_sufficient = _is_template_ready(prefilled_data)
        logger.info(
            "gather fill_missing before=%s next_phase=%s is_sufficient=%s",
            prefilled_data,
            next_phase,
            is_sufficient,
        )
        return {
            "ai_message": _apply_turn_policy_to_message(
                state,
                _finalize_committed_response(
                    turn_type=turn_type,
                    merged_data=prefilled_data,
                    accepted_updates={},
                    ai_message="",
                    is_sufficient=is_sufficient,
                ),
            ),
            "collected_data": raw_current_data,
            "is_sufficient": is_sufficient,
            "next_phase": next_phase,
        }
    if turn_type in {"request_refine_topic", "request_guided_exploration", "request_goal_guidance"} and not direct_updates:
        next_phase = derive_phase_from_collected_data(prefilled_data, current_phase=current_phase)
        is_sufficient = _is_template_ready(prefilled_data)
        logger.info(
            "gather guided_request turn_type=%s before=%s next_phase=%s is_sufficient=%s",
            turn_type,
            prefilled_data,
            next_phase,
            is_sufficient,
        )
        return {
            "ai_message": _apply_turn_policy_to_message(
                state,
                _finalize_committed_response(
                    turn_type=turn_type,
                    merged_data=prefilled_data,
                    accepted_updates={},
                    ai_message="",
                    is_sufficient=is_sufficient,
                ),
            ),
            "collected_data": raw_current_data,
            "is_sufficient": is_sufficient,
            "next_phase": next_phase,
        }
    if turn_type == "provide_problem_area":
        problem_area_candidate = _extract_problem_area_candidate(
            state,
            prefilled_data,
            direct_updates=direct_updates,
        )
        if problem_area_candidate:
            contextual_data = merge_collected_data(
                prefilled_data,
                {"problemArea": problem_area_candidate},
            )
            topic_anchor = _get_topic_anchor(prefilled_data, allow_title_fallback=False)
            ai_message, next_phase, is_sufficient = _build_problem_area_commit_response(
                contextual_data=contextual_data,
                topic_anchor=topic_anchor,
                problem_area=problem_area_candidate,
                current_phase=current_phase,
            )
            logger.info(
                "gather problem_area topic=%r problem_area=%r before=%s after=%s next_phase=%s",
                topic_anchor,
                problem_area_candidate,
                prefilled_data,
                contextual_data,
                next_phase,
            )
            return {
                "ai_message": _apply_turn_policy_to_message(
                    state,
                    ai_message,
                ),
                "collected_data": contextual_data,
                "is_sufficient": is_sufficient,
                "next_phase": next_phase,
            }
    if turn_type == "request_next_step" and not direct_updates:
        next_phase = derive_phase_from_collected_data(prefilled_data, current_phase=current_phase)
        is_sufficient = _is_template_ready(prefilled_data)
        logger.info(
            "gather next_step before=%s next_phase=%s is_sufficient=%s",
            prefilled_data,
            next_phase,
            is_sufficient,
        )
        return {
            "ai_message": _apply_turn_policy_to_message(
                state,
                _finalize_committed_response(
                    turn_type=turn_type,
                    merged_data=prefilled_data,
                    accepted_updates={},
                    ai_message="",
                    is_sufficient=is_sufficient,
                ),
            ),
            "collected_data": raw_current_data,
            "is_sufficient": is_sufficient,
            "next_phase": next_phase,
        }
    if turn_type == "meta_request" and not direct_updates:
        next_phase = derive_phase_from_collected_data(prefilled_data, current_phase=current_phase)
        is_sufficient = _is_template_ready(prefilled_data)
        return {
            "ai_message": _apply_turn_policy_to_message(state, _answer_only_fallback(state, "")),
            "collected_data": raw_current_data,
            "is_sufficient": is_sufficient,
            "next_phase": next_phase,
        }
    if direct_conflict_message:
        next_phase = derive_phase_from_collected_data(merged_preview, current_phase=current_phase)
        is_sufficient = _is_template_ready(merged_preview)
        logger.info(
            "gather role/team mismatch before=%s attempted=%s after=%s",
            prefilled_data,
            direct_candidate_updates,
            merged_preview,
        )
        _log_collected_data_trace(
            state,
            raw_collected_data=raw_current_data,
            normalized_collected_data=prefilled_data,
            direct_updates_raw=direct_updates_raw,
            llm_updates_raw={},
            candidate_updates_merged=direct_candidate_updates,
            approved_updates=direct_approved_updates,
            rejected_updates=direct_rejected_updates,
            rejected_reasons=direct_rejected_reasons,
            decisions=direct_decisions,
        )
        follow_up = _build_next_missing_field_prompt(
            merged_preview,
            current_phase=current_phase,
            followup_fields=direct_followup_fields,
            rejected_updates=direct_rejected_updates,
        )
        ai_message = direct_conflict_message
        if follow_up:
            ai_message = f"{ai_message} {follow_up}"
        return {
            "ai_message": _apply_turn_policy_to_message(state, ai_message),
            "collected_data": merged_preview,
            "is_sufficient": is_sufficient,
            "next_phase": next_phase,
            "approved_updates": direct_approved_updates,
            "rejected_updates": direct_rejected_updates,
            "rejected_reasons": direct_rejected_reasons,
            "followup_fields": direct_followup_fields,
            "next_question_field": choose_next_question_field(
                merged_preview,
                current_phase=current_phase,
                followup_fields=direct_followup_fields,
                rejected_updates=direct_rejected_updates,
            ),
        }
    if direct_updates and (
        turn_type in {"provide_fact", "provide_due_date_candidate"}
        or requested_next_step
        or turn_type == "request_fill_missing"
        or direct_approved_updates
    ):
        next_question_field = choose_next_question_field(
            merged_preview,
            current_phase=current_phase,
            followup_fields=direct_followup_fields,
            rejected_updates=direct_rejected_updates,
        )
        next_phase = derive_phase_from_collected_data(merged_preview, current_phase=current_phase)
        is_sufficient = _is_template_ready(merged_preview)
        logger.info(
            "gather committed direct updates before=%s after=%s next_phase=%s is_sufficient=%s next_question_field=%s",
            prefilled_data,
            merged_preview,
            next_phase,
            is_sufficient,
            next_question_field,
        )
        _log_collected_data_trace(
            state,
            raw_collected_data=raw_current_data,
            normalized_collected_data=prefilled_data,
            direct_updates_raw=direct_updates_raw,
            llm_updates_raw={},
            candidate_updates_merged=direct_candidate_updates,
            approved_updates=direct_approved_updates,
            rejected_updates=direct_rejected_updates,
            rejected_reasons=direct_rejected_reasons,
            decisions=direct_decisions,
        )
        return {
            "ai_message": _apply_turn_policy_to_message(
                state,
                _finalize_committed_response(
                    turn_type="provide_fact",
                    merged_data=merged_preview,
                    accepted_updates=direct_approved_updates,
                    ai_message="",
                    is_sufficient=is_sufficient,
                    current_phase=current_phase,
                    rejected_updates=direct_rejected_updates,
                    rejected_reasons=direct_rejected_reasons,
                    followup_fields=direct_followup_fields,
                ),
            ),
            "collected_data": merged_preview,
            "is_sufficient": is_sufficient,
            "next_phase": next_phase,
            "approved_updates": direct_approved_updates,
            "rejected_updates": direct_rejected_updates,
            "rejected_reasons": direct_rejected_reasons,
            "followup_fields": direct_followup_fields,
            "next_question_field": next_question_field,
        }
    focus_instruction = _build_gather_focus_instruction(focus_type)
    missing_field_summary = _build_missing_field_summary(prefilled_data)
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
    - updated_data stores only candidate facts newly supported in this turn.
    - updated_data must be partial.
    - do not repeat full collected_data.
    - do not overwrite existing values unless explicitly corrected or strongly restated.
    - do not output placeholders, guesses, requests, or summary asks as facts.
    - if the user is asking for summary/help, return updated_data={{}}.
    - Never say the project is ready for template generation unless all collected data fields are filled with valid values.
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
    {turn_type}

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
        next_phase = derive_phase_from_collected_data(prefilled_data, current_phase=current_phase)
        return {
            "ai_message": _apply_turn_policy_to_message(state, _answer_only_fallback(state, "")),
            "collected_data": prefilled_data,
            "is_sufficient": was_ready,
            "next_phase": next_phase,
        }

    try:
        raw_result = json.loads(response.content)
        result = _coerce_gather_llm_result(raw_result)
    except JSONDecodeError as exc:
        logger.exception(
            "failed to parse gather_information_node JSON: %s raw_output=%s",
            exc,
            getattr(response, "content", ""),
        )
        next_phase = derive_phase_from_collected_data(prefilled_data, current_phase=current_phase)
        return {
            "ai_message": _apply_turn_policy_to_message(state, _answer_only_fallback(state, "")),
            "collected_data": prefilled_data,
            "is_sufficient": was_ready,
            "next_phase": next_phase,
        }

    llm_updates_raw = dict(result.get("raw_updated_data", {}))
    normalized_llm_updates = _normalize_contextual_llm_updates(
        raw_updates=llm_updates_raw,
        sanitized_updates=result["updated_data"],
        current_data=prefilled_data,
        user_message=user_message,
    )
    filtered_updates = _filter_gather_updates(
        user_message,
        normalized_llm_updates,
        focus_type=focus_type,
    )
    if _get_topic_anchor(prefilled_data, allow_title_fallback=False):
        filtered_updates.pop("subject", None)
        filtered_updates.pop("title", None)
    candidate_sources, candidate_updates_merged = _merge_candidate_sources(
        current_data=prefilled_data,
        user_message=user_message,
        direct_updates=direct_updates,
        llm_updates=filtered_updates,
    )
    (
        approved_updates,
        rejected_updates,
        rejected_reasons,
        decisions,
        followup_fields,
    ) = _evaluate_candidate_updates(
        current_data=prefilled_data,
        candidate_sources=candidate_sources,
        candidate_updates=candidate_updates_merged,
        user_message=user_message,
        current_phase=current_phase,
        turn_type=turn_type,
        recent_messages=state.get("recent_messages", []),
        selected_message=state.get("selected_message"),
    )
    merged_data = merge_collected_data(prefilled_data, approved_updates)
    accepted_updates = {
        key: value for key, value in merged_data.items() if prefilled_data.get(key) != value
    }
    next_phase = derive_phase_from_collected_data(merged_data, current_phase=current_phase)
    is_sufficient = _is_template_ready(merged_data)
    next_question_field = choose_next_question_field(
        merged_data,
        current_phase=current_phase,
        followup_fields=followup_fields,
        rejected_updates=rejected_updates,
    )
    _log_collected_data_trace(
        state,
        raw_collected_data=raw_current_data,
        normalized_collected_data=prefilled_data,
        direct_updates_raw=direct_updates_raw,
        llm_updates_raw=llm_updates_raw,
        candidate_updates_merged=candidate_updates_merged,
        approved_updates=approved_updates,
        rejected_updates=rejected_updates,
        rejected_reasons=rejected_reasons,
        decisions=decisions,
    )
    logger.info(
        "gather llm_candidates=%s candidate_updates_merged=%s accepted_updates=%s rejected_updates=%s before=%s after=%s next_phase=%s is_sufficient=%s next_question_field=%s",
        filtered_updates,
        candidate_updates_merged,
        accepted_updates,
        rejected_updates,
        prefilled_data,
        merged_data,
        next_phase,
        is_sufficient,
        next_question_field,
    )
    ai_msg = _finalize_committed_response(
        turn_type=turn_type,
        merged_data=merged_data,
        accepted_updates=accepted_updates,
        ai_message=str(result.get("ai_message") or "").strip(),
        is_sufficient=is_sufficient,
        current_phase=current_phase,
        rejected_updates=rejected_updates,
        rejected_reasons=rejected_reasons,
        followup_fields=followup_fields,
    )
    candidate_conflict_message = _decision_conflict_message(
        prefilled_data,
        {**prefilled_data, **candidate_updates_merged},
        decisions,
    )
    if candidate_conflict_message:
        follow_up = _build_next_missing_field_prompt(
            merged_data,
            current_phase=current_phase,
            followup_fields=followup_fields,
            rejected_updates=rejected_updates,
        )
        ai_msg = candidate_conflict_message
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

    return {
        "ai_message": ai_msg,
        "collected_data": merged_data,
        "is_sufficient": is_sufficient,
        "next_phase": next_phase,
        "approved_updates": approved_updates,
        "rejected_updates": rejected_updates,
        "rejected_reasons": rejected_reasons,
        "followup_fields": followup_fields,
        "next_question_field": next_question_field,
    }


_original_extract_topic_candidate = _extract_topic_candidate
_original_extract_direct_fact_updates = _extract_direct_fact_updates
_original_looks_like_commit_confirmation = _looks_like_commit_confirmation


def _extract_topic_candidate(user_message: str) -> str:
    candidate = _original_extract_topic_candidate(user_message)
    if candidate:
        return candidate

    text = _clean_text(_strip_mates_mention(user_message))
    if not text or "?" in text or _is_meta_conversation_message(text):
        return ""

    reverse_match = re.match(
        r"^(.+?)\s*(?:을|를|이|가)?\s*주제로\s*(?:생각(?:하고)?\s*있(?:어|어요)|생각중(?:이야|이에요)?|잡(?:고)?\s*있(?:어|어요)|정했(?:어|어요)|하려(?:고)?(?:\s*해)?|하고\s*싶(?:어|어요))\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if not reverse_match:
        return ""

    candidate = reverse_match.group(1).strip()
    candidate = TRAILING_TOPIC_ENDINGS_PATTERN.sub("", candidate).strip(" .,!?:;\"'")
    if not candidate or len(candidate) > 60:
        return ""
    return candidate


def _normalize_goal_candidate(candidate: str) -> str:
    normalized = _normalize_direct_fact_value(candidate)
    if not normalized:
        return ""

    normalized = re.sub(r"^\s*\d{1,2}[)\.\-:]\s*", "", normalized).strip()
    normalized = re.sub(
        r"\s*(?:을|를)\s*목표(?:로|로는)?\s*(?:할게|할 거야|할래|잡을게|정할게|삼을게|둘게)\s*$",
        "",
        normalized,
        flags=re.IGNORECASE,
    ).strip()
    normalized = re.sub(
        r"\s*(?:이|가)\s*목표(?:야|예요|입니다)\s*$",
        "",
        normalized,
        flags=re.IGNORECASE,
    ).strip()
    normalized = re.sub(
        r"^\s*(?:목표|goal)\s*(?:은|는|이|가|:)\s*",
        "",
        normalized,
        flags=re.IGNORECASE,
    ).strip()
    return normalized.strip(" .,!?:;\"'")


def _extract_goal_candidate_from_message(message: str) -> str:
    normalized_message = _clean_text(message)
    if (
        not normalized_message
        or _is_meta_conversation_message(normalized_message)
        or _is_current_goal_query(normalized_message)
    ):
        return ""

    explicit_patterns = (
        r"(?:목표|goal)\s*(?:은|는|이|가|:)\s*(.+)$",
        r"(.+?)\s*(?:을|를)\s*목표(?:로|로는)?\s*(?:할게|할 거야|할래|잡을게|정할게|삼을게|둘게)\s*$",
        r"(.+?)\s*(?:이|가)\s*목표(?:야|예요|입니다)\s*$",
    )
    for pattern in explicit_patterns:
        match = re.search(pattern, normalized_message, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = _normalize_goal_candidate(match.group(1))
        compact = re.sub(r"\s+", "", candidate)
        if compact in {
            "이거",
            "이걸",
            "그거",
            "그걸",
            "저거",
            "저걸",
            "로하자",
            "로할게",
            "로할게요",
            "로정하자",
            "로정할게",
            "로정할게요",
        }:
            continue
        if (
            candidate
            and not _is_non_storable_freeform_message(candidate)
            and not _looks_like_question_line(candidate)
            and not _looks_like_open_question(candidate)
        ):
            return candidate
    return ""


def _extract_direct_fact_updates(user_message: str) -> dict[str, object]:
    updates = dict(_original_extract_direct_fact_updates(user_message))
    goal_candidate = _extract_goal_candidate_from_message(
        _clean_text(_strip_mates_mention(user_message))
    )
    if goal_candidate:
        updates["goal"] = goal_candidate
    return _sanitize_gather_updates(updates)


def _looks_like_commit_confirmation(message: str) -> bool:
    if _original_looks_like_commit_confirmation(message):
        return True

    normalized = _clean_text(message)
    if not normalized:
        return False

    return bool(
        re.search(r"(?:주제|목표|제목|마감|역할|인원|산출물).{0,6}확인했", normalized)
        or re.search(r"(?:확인했습니다|확인했어요|확인할게요)", normalized)
    )


_prior_extract_due_date_candidate_from_message = _extract_due_date_candidate_from_message
_prior_extract_direct_fact_updates = _extract_direct_fact_updates
_prior_extract_target_facility_candidate = _extract_target_facility_candidate
_prior_extract_problem_area_candidate = _extract_problem_area_candidate
_prior_interpret_turn_type = _interpret_turn_type


def _looks_like_public_facility_topic(topic_anchor: str) -> bool:
    normalized = _clean_text(topic_anchor)
    if not normalized:
        return False
    return any(keyword in normalized for keyword in ("공공시설", "시설", "도서관", "공원", "주민센터"))


def _extract_deliverable_type_candidate(message: str) -> str:
    normalized = _clean_text(message)
    compact = re.sub(r"\s+", "", normalized)
    if not normalized:
        return ""
    if any(token in normalized for token in ("알려줘", "도와줘", "정리해줘", "추천해줘", "예시")):
        return ""
    if any(token in compact for token in ("웹서비스", "웹형태", "웹서비스형태", "웹앱", "웹app")):
        return "웹 서비스"
    if any(token in compact for token in ("모바일앱", "앱형태")):
        return "모바일 앱"
    if "프로토타입" in normalized:
        return "프로토타입"
    return ""


def _normalize_due_date_qualifier(value: str) -> str:
    normalized = _clean_text(value)
    compact = re.sub(r"\s+", "", normalized)
    if compact in {"최종제출", "최종발표"}:
        return "최종 제출"
    if compact in {"중간발표", "중간제출"}:
        return "중간 발표"
    return normalized


def _extract_due_date_candidate_from_message(message: str) -> str:
    candidate = _prior_extract_due_date_candidate_from_message(message)
    if candidate:
        return candidate

    normalized = _clean_text(message)
    if not normalized:
        return ""

    date_chunk = r"((?:20\d{2}|\d{2})\s*년\s*\d{1,2}\s*월(?:\s*\d{1,2}\s*일)?|\d{1,2}\s*월(?:\s*\d{1,2}\s*일)?)"
    patterns = (
        rf"{date_chunk}\s*(?:이|가)?\s*(?:마감(?:일)?|데드라인)",
        rf"(?:마감(?:일)?|데드라인|최종\s*제출|중간\s*발표)\s*(?:은|는|이|가|:)?\s*{date_chunk}",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if not match:
            continue
        for group in match.groups():
            if group:
                return _normalize_due_date_candidate(group)
    return ""


def _extract_direct_fact_updates(user_message: str) -> dict[str, object]:
    updates = dict(_prior_extract_direct_fact_updates(user_message))
    message = _clean_text(_strip_mates_mention(user_message))
    deliverable_candidate = _extract_deliverable_type_candidate(message)
    if deliverable_candidate and "deliverables" not in updates:
        updates["deliverables"] = deliverable_candidate
    return _sanitize_gather_updates(updates)


def _extract_target_facility_candidate(
    state: AgentState,
    current_data: dict[str, object] | None = None,
) -> str:
    candidate = _prior_extract_target_facility_candidate(state, current_data)
    if candidate:
        return candidate

    current_data = current_data or {}
    topic_anchor = _get_topic_anchor(current_data, allow_title_fallback=False)
    if not topic_anchor or not _looks_like_public_facility_topic(topic_anchor):
        return ""

    normalized = _clean_text(_effective_user_message(state))
    if not normalized:
        return ""
    if _is_request_like_value(normalized) or _is_undecided_value(normalized):
        return ""
    if _is_meta_conversation_message(normalized) or _extract_due_date_candidate_from_message(normalized):
        return ""
    if _extract_choice_index(normalized) is not None:
        return ""

    compact = re.sub(r"\s+", "", normalized)
    if len(compact) > 10:
        return ""
    if not TARGET_FACILITY_NOUN_PATTERN.search(normalized):
        return ""
    return _normalize_direct_fact_value(normalized).strip(" '\"")


def _extract_problem_area_candidate(
    state: AgentState,
    current_data: dict[str, object] | None = None,
    *,
    direct_updates: dict[str, object] | None = None,
) -> str:
    candidate = _prior_extract_problem_area_candidate(
        state,
        current_data,
        direct_updates=direct_updates,
    )
    if not candidate:
        normalized_message = _clean_text(_effective_user_message(state))
        if (
            not normalized_message
            or _is_meta_conversation_message(normalized_message)
            or _looks_like_open_question(normalized_message)
        ):
            return ""
        fallback_match = re.search(
            r"(.+?문제)(?:를|가)?\s*(?:해결하고\s*싶(?:어요|다)|해결하려(?:고|는)|줄이고\s*싶(?:어요|다)|개선하고\s*싶(?:어요|다))",
            normalized_message,
        )
        if not fallback_match:
            return ""
        candidate = fallback_match.group(1).strip()

    normalized = _clean_text(candidate)
    compact = re.sub(r"\s+", "", normalized)
    if "형태" in normalized:
        return ""
    if compact.endswith("우선") or compact in {"시민이우선", "관리자가우선"}:
        return ""
    if any(token in normalized for token in ("시민", "관리자", "사용자")) and "우선" in normalized:
        return ""
    if TARGET_FACILITY_NOUN_PATTERN.search(normalized) and len(compact) <= 10:
        return ""
    return candidate


def _looks_like_advice_request(user_message: str) -> bool:
    normalized = _clean_text(_strip_mates_mention(user_message))
    if not normalized:
        return False
    help_tokens = ("알려줘", "도와줘", "정리해줘", "써줘", "만들어줘", "추천해줘")
    advice_topics = ("형태", "문장", "논리", "구조", "예시", "발표용", "산출물")
    return any(token in normalized for token in help_tokens) and any(
        topic in normalized for topic in advice_topics
    )


def _looks_like_planning_request(user_message: str) -> bool:
    normalized = _clean_text(_strip_mates_mention(user_message))
    if not normalized:
        return False
    planning_patterns = (
        re.compile(r"(?:우선순위|priority)\s*정리", re.IGNORECASE),
        re.compile(r"(?:다음|그다음)\s*할\s*일", re.IGNORECASE),
        re.compile(r"(?:action\s*items?|todo|to-do|체크리스트|로드맵)", re.IGNORECASE),
        re.compile(r"(?:실행\s*계획|진행\s*순서|mvp\s*기준)", re.IGNORECASE),
    )
    return any(pattern.search(normalized) for pattern in planning_patterns)


def _interpret_turn_type(
    state: AgentState,
    current_data: dict[str, str] | None = None,
    *,
    direct_updates: dict[str, object] | None = None,
) -> str:
    turn_type = _prior_interpret_turn_type(
        state,
        current_data,
        direct_updates=direct_updates,
    )
    user_message = _effective_user_message(state)
    direct_updates = dict(direct_updates or {})

    if turn_type == "request_next_step" and (
        _looks_like_advice_request(user_message) or _looks_like_planning_request(user_message)
    ):
        return "general"
    if turn_type == "request_next_step" and any(
        token in user_message for token in ("정해줘", "추천", "알려줘", "정리해줘")
    ):
        return "general"

    if turn_type == "provide_problem_area":
        effective_current_data = current_data or state.get("collected_data") or {}
        if not _extract_problem_area_candidate(state, effective_current_data, direct_updates=direct_updates):
            if _extract_target_facility_candidate(state, effective_current_data):
                return "provide_target_facility"
            return "general"

    return turn_type


def _normalize_contextual_llm_updates(
    *,
    raw_updates: dict[str, object],
    sanitized_updates: dict[str, object],
    current_data: dict[str, object],
    user_message: str,
) -> dict[str, object]:
    normalized_updates = dict(sanitized_updates or {})
    due_date = str(current_data.get("dueDate") or "").strip()
    qualifier = _normalize_due_date_qualifier(
        str(raw_updates.get("dueDatePhase") or raw_updates.get("dueDateType") or "").strip()
    )

    if due_date and qualifier:
        normalized_updates["dueDate"] = f"{due_date} ({qualifier})"

    return normalized_updates


_final_extract_direct_fact_updates = _extract_direct_fact_updates
def _normalize_goal_candidate(candidate: str) -> str:
    normalized = _normalize_direct_fact_value(candidate)
    if not normalized:
        return ""

    normalized = re.sub(r"^\s*(?:목표|goal)\s*(?:은|는|이|가|:)\s*", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(
        r"\s*(?:이|가)\s*목표(?:야|예요|입니다)?\s*$",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\s*(?:을|를)\s*목표(?:로|로는)?\s*(?:할게|할 거야|할래|잡을게|정할게|삼을게|둘게)\s*$",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"^\s*\d{1,2}[)\.\-:]\s*", "", normalized).strip()
    normalized = re.sub(r"\s+", " ", normalized).strip(" .,!?:;\"'")
    return normalized


def _extract_goal_candidate_from_message(message: str) -> str:
    normalized_message = _clean_text(message)
    if (
        not normalized_message
        or _is_meta_conversation_message(normalized_message)
        or _is_current_goal_query(normalized_message)
        or _looks_like_question_line(normalized_message)
        or _looks_like_open_question(normalized_message)
    ):
        return ""

    explicit_patterns = (
        r"^\s*목표\s*(?:는|은|이|가|:)\s*(.+)$",
        r"^\s*(.+?)\s*을\s*목표로\s*(?:하자|할게|할게요|정하자|정할게|정할게요)\s*$",
        r"^\s*(.+?)\s*가\s*목표(?:예요|입니다)?\s*$",
        r"^\s*\d{1,2}[)\.\-:]\s*(.+?)\s*(?:이걸|이걸로|그걸|그걸로)?\s*목표(?:로|로는)?\s*(?:하자|할게|할 거야|할래|잡을게|잡자|정할게|정하자|삼을게|삼자|둘게|두자)\s*$",
        r"^(.+?)\s*(?:이걸|이걸로|그걸|그걸로)\s*목표(?:로|로는)?\s*(?:하자|할게|할 거야|할래|잡을게|잡자|정할게|정하자|삼을게|삼자|둘게|두자)\s*$",
        r"^목표(?:는|은)?\s*(.+?)(?:이|가)\s*목표(?:야|예요|입니다)?\s*$",
        r"^(.+?)(?:이|가)\s*목표(?:야|예요|입니다)?\s*$",
        r"^목표(?:는|은)?\s*(.+)$",
        r"^(.+?)(?:을|를)\s*목표(?:로|로는)?\s*(?:할게|할 거야|할래|잡을게|정할게|삼을게|둘게)\s*$",
    )
    for pattern in explicit_patterns:
        match = re.search(pattern, normalized_message, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = _normalize_goal_candidate(match.group(1))
        compact = re.sub(r"\s+", "", candidate)
        if compact in {"이거", "이걸", "그거", "그걸", "저거", "저걸"}:
            continue
        if candidate and not _is_non_storable_freeform_message(candidate):
            return candidate
    return ""


def _extract_direct_fact_updates(user_message: str) -> dict[str, object]:
    updates = dict(_final_extract_direct_fact_updates(user_message))
    normalized_message = _clean_text(_strip_mates_mention(user_message))
    if re.search(r"^\s*(?:이거|이걸|그거|그걸|저거|저걸)\s*목표로\s*(?:하자|할게|할게요|정하자|정할게|정할게요)\s*$", normalized_message):
        updates.pop("goal", None)
    target_user_patterns = (
        r"^\s*(?:타겟|대상)\s*사용자(?:는|은|이|가|:)\s*(.+)$",
        r"^\s*(.+?)\s*(?:이|가)?\s*타겟\s*사용자(?:예요|입니다|야)\s*$",
    )
    for pattern in target_user_patterns:
        match = re.search(pattern, normalized_message, flags=re.IGNORECASE)
        if not match:
            continue
        target_user = _normalize_direct_fact_value(match.group(1))
        if target_user and not _is_non_storable_freeform_message(target_user):
            updates["targetUser"] = target_user
            break
    if _is_meta_conversation_message(normalized_message):
        updates.pop("subject", None)
        updates.pop("goal", None)
        updates.pop("deliverables", None)
        updates.pop("dueDate", None)
    if "목표" in normalized_message and (
        _is_current_goal_query(normalized_message)
        or _looks_like_question_line(normalized_message)
        or _looks_like_open_question(normalized_message)
    ):
        updates.pop("goal", None)
    goal_candidate = _extract_goal_candidate_from_message(normalized_message)
    if goal_candidate:
        if goal_candidate.endswith(("를", "을")):
            goal_candidate = goal_candidate[:-1].rstrip()
        updates["goal"] = goal_candidate
    return _sanitize_gather_updates(updates)


def _is_summary_request(user_message: str) -> bool:
    normalized = _clean_text(_strip_mates_mention(user_message))
    if not normalized:
        return False

    recommendation_topics = ("산출물", "결과물", "제출물")
    recommendation_verbs = ("추천", "알려", "가이드", "예시", "정리해줘")
    if any(topic in normalized for topic in recommendation_topics) and any(
        verb in normalized for verb in recommendation_verbs
    ):
        return False

    return _signal_is_summary_request(user_message)


_final_goal_candidate_extractor = _extract_goal_candidate_from_message
_final_problem_area_candidate_extractor = _extract_problem_area_candidate


def _extract_goal_candidate_from_message(message: str) -> str:
    normalized_message = _clean_text(message)
    if (
        not normalized_message
        or _is_meta_conversation_message(normalized_message)
        or _is_current_goal_query(normalized_message)
    ):
        return ""

    candidate = _final_goal_candidate_extractor(normalized_message)
    if not candidate:
        return ""
    compact = re.sub(r"\s+", "", candidate)
    if compact in {"로하자", "로할게", "로정하자", "로정할게", "이걸", "그걸", "저걸"}:
        return ""
    if _looks_like_question_line(candidate) or _looks_like_open_question(candidate):
        return ""
    return candidate


def _extract_problem_area_candidate(
    state: AgentState,
    current_data: dict[str, object] | None = None,
    *,
    direct_updates: dict[str, object] | None = None,
) -> str:
    normalized_message = _clean_text(_effective_user_message(state))
    candidate = _final_problem_area_candidate_extractor(
        state,
        current_data,
        direct_updates=direct_updates,
    )
    if candidate:
        if normalized_message and (
            _looks_like_question_line(normalized_message)
            or _looks_like_open_question(normalized_message)
        ):
            return ""
        return candidate

    if (
        not normalized_message
        or _is_meta_conversation_message(normalized_message)
        or _looks_like_question_line(normalized_message)
        or _looks_like_open_question(normalized_message)
        or _is_storage_control_message(normalized_message, current_data or {})
    ):
        return ""

    fallback_match = re.search(
        r"(.+?문제)(?:를|가)?\s*(?:해결하고|줄이고|개선하고)",
        normalized_message,
    )
    if not fallback_match:
        return ""

    candidate = _clean_text(fallback_match.group(1))
    if (
        not candidate
        or candidate == _get_topic_anchor(current_data or {}, allow_title_fallback=False)
        or _looks_like_question_line(candidate)
        or _looks_like_open_question(candidate)
        or _is_storage_control_message(candidate, current_data or {})
        or _extract_due_date_candidate_from_message(candidate)
        or _looks_like_title_instruction(candidate)
    ):
        return ""

    compact = re.sub(r"\s+", "", candidate)
    if TARGET_FACILITY_NOUN_PATTERN.search(candidate) and len(compact) <= 10:
        return ""
    return candidate


def _extract_direct_fact_updates(user_message: str) -> dict[str, object]:
    updates = dict(_final_extract_direct_fact_updates(user_message))
    normalized_message = _clean_text(_strip_mates_mention(user_message))
    if re.search(r"^\s*(?:이거|이걸|그거|그걸|저거|저걸)\s*목표로\s*(?:하자|할게|정하자|정할게)$", normalized_message):
        updates.pop("goal", None)

    target_user_patterns = (
        r"^\s*(?:타겟\s*사용자|대상\s*사용자)\s*(?:은|는|가|:)\s*(.+)$",
        r"^\s*(.+?)\s*(?:이|가)?\s*대상\s*사용자(?:예요|입니다)?\s*$",
    )
    for pattern in target_user_patterns:
        match = re.search(pattern, normalized_message, flags=re.IGNORECASE)
        if not match:
            continue
        target_user = _normalize_direct_fact_value(match.group(1))
        if target_user and not _is_non_storable_freeform_message(target_user):
            updates["targetUser"] = target_user
            break

    if _is_meta_conversation_message(normalized_message):
        updates.pop("subject", None)
        updates.pop("goal", None)
        updates.pop("deliverables", None)
        updates.pop("dueDate", None)
    if "목표" in normalized_message and _is_current_goal_query(normalized_message):
        updates.pop("goal", None)
    if (
        _detect_requested_gather_focus(normalized_message) == "deliverables"
        and (_looks_like_question_line(normalized_message) or _looks_like_open_question(normalized_message))
    ):
        updates.pop("deliverables", None)

    goal_candidate = _extract_goal_candidate_from_message(normalized_message)
    if goal_candidate:
        if goal_candidate.endswith(("를", "을")):
            goal_candidate = goal_candidate[:-1].rstrip()
        updates["goal"] = goal_candidate
    if str(updates.get("goal") or "").strip() in {"로 하자", "로 할게", "로 정하자", "로 정할게"}:
        updates.pop("goal", None)
    return _sanitize_gather_updates(updates)


_final_fresh_topic_submission_candidate_extractor = _extract_fresh_topic_submission_candidate


def _extract_fresh_topic_submission_candidate(
    state: AgentState,
    current_data: dict[str, object] | None = None,
    *,
    direct_updates: dict[str, object] | None = None,
) -> str:
    candidate = _final_fresh_topic_submission_candidate_extractor(
        state,
        current_data,
        direct_updates=direct_updates,
    )
    if not candidate:
        return ""

    candidate = _postprocess_explicit_title_value(candidate)
    candidate = re.sub(
        r"\s*(?:으로|로)\s*(?:바꿀게|바꿀래|바꾸자|수정할게|수정하자|변경할게|변경하자)\s*$",
        "",
        candidate,
        flags=re.IGNORECASE,
    ).strip(" .,!?:;\"'")
    return candidate
