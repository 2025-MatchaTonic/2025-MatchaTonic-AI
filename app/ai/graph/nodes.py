# app/ai/graph/nodes.py
import json
import logging
import re
from json import JSONDecodeError
from time import perf_counter

from app.ai.graph.collected_data import (
    CandidateDecision,
    apply_collected_data_updates,
    choose_next_question_field,
    derive_phase_from_collected_data,
    is_template_ready as _shared_is_template_ready,
    is_valid_collected_value as _shared_is_valid_collected_value,
    merge_collected_data,
    sanitize_candidate_updates as _shared_sanitize_candidate_updates,
)
from app.ai.graph.conversation_heuristics import (
    FAST_RAG_PHASES,
    GATHER_FIELD_GUIDE,
    QUESTION_LINE_ENDING_PATTERN,
    RAG_FILTERS_BY_PHASE,
    SHORT_MESSAGE_PATTERN,
    TRIVIAL_MESSAGES,
)
from app.ai.graph.llm_clients import (
    invoke_llm as _invoke_llm,
    structured_llm,
)
from app.ai.schemas.llm_outputs import ConversationLLMDecision
from app.ai.graph.state import AgentState, TurnPolicy
from app.ai.graph.template_support import build_recent_context as _build_recent_context
from app.ai.graph.text_support import (
    MATES_MENTION_PATTERN,
    clean_text as _clean_text,
    strip_mates_mention as _strip_mates_mention,
    truncate_message as _truncate_ai_message,
)
from app.ai.graph.topic_presence import (
    _is_topic_presence_negative_message,
    _matches_initial_button_message,
    _matches_topic_presence_button_message,
)
from app.core.config import settings
from app.core.request_normalization import (
    normalize_collected_data as _normalize_request_collected_data,
)
from app.rag.retriever import get_rag_context

logger = logging.getLogger(__name__)

RAG_EMPTY_CONTEXT = "(참고할 레퍼런스를 찾지 못했습니다.)"
RAG_CACHE_MAX_ITEMS = 128
RAG_CONTEXT_CACHE: dict[tuple[str, str, tuple[str, ...], tuple[str, ...], int], str] = (
    {}
)

_VALID_NEXT_FIELDS = {
    "subject",
    "title",
    "problemArea",
    "targetFacility",
    "goal",
    "targetUser",
    "teamSize",
    "roles",
    "dueDate",
    "deliverables",
}

_EXPLICIT_SLOT_MARKERS: dict[str, re.Pattern[str]] = {
    "subject": re.compile(r"(?:subject|topic|\uc8fc\uc81c)", re.IGNORECASE),
    "title": re.compile(r"(?:title|\uc81c\ubaa9|\uc774\ub984)", re.IGNORECASE),
    "goal": re.compile(r"(?:goal|objective|\ubaa9\ud45c)", re.IGNORECASE),
}

_ACCEPT_RECOMMENDATION_PATTERN = re.compile(
    r"(?:추천(?:하는)?\s*대로|이대로|그걸로|그걸로\s*할게|확정|b\s*로|B\s*로)",
    re.IGNORECASE,
)
_QUOTED_RECOMMENDATION_PATTERN = re.compile(r"'([^']{10,180})'")
_OPTION_B_PATTERN = re.compile(r"B\)\s*([^\n]+)")

_PHASE_CONTEXT: dict[str, str] = {
    "EXPLORE": (
        "사용자가 아직 구체적인 프로젝트 주제를 정하지 못한 단계입니다. "
        "이 단계는 실행 계획 수집이 아니라 주제 발굴 전용입니다. "
        "[EXPLORE 주제 발굴 순서] "
        "▶ 1단계 문제 탐색 — 사용자가 불편함·관심 분야·상황을 말하면 subject/title/goal 업데이트 금지. problemArea로만 기록하고, 어떤 상황에서 가장 불편한지 한 가지만 더 물어보세요. "
        "▶ 2단계 타겟 선정 — problemArea가 있으면 그 문제를 가장 자주 겪는 사람이나 집단을 묻고 targetUser로 기록하세요. "
        "▶ 3단계 주제 후보 — problemArea와 targetUser가 모두 있을 때만 구체적인 서비스/앱/플랫폼 주제 후보 1개를 제안하세요. 이때도 사용자가 동의하기 전까지 subject/title 업데이트 금지. "
        "▶ 동의 확정 — 사용자가 제안에 동의(좋아, 그걸로, 맞아, 그거 좋다 등)한 경우에만 subject를 업데이트하세요. "
        "▶ 금지 — subject가 확정되기 전에는 goal/teamSize/roles/dueDate/deliverables를 묻거나 업데이트하지 마세요."
    ),
    "TOPIC_SET": (
        "프로젝트 주제는 잡혀있지만 아직 구체화가 필요한 단계입니다. "
        "주제를 명확히 하고 핵심 문제 영역을 찾아낼 수 있도록 도와주세요."
    ),
    "PROBLEM_DEFINE": (
        "프로젝트 방향은 있지만 구체적인 문제 정의가 필요한 단계입니다. "
        "어떤 문제를 프로젝트로 풀어낼지 함께 생각해보도록 이끌어주세요."
    ),
    "GATHER": (
        "팀이 실행 계획을 함께 세워나가는 단계입니다. "
        "빈 필드를 채우는 게 목적이 아니라, 팀이 인원·역할·일정·결과물에 대한 결정을 내릴 수 있도록 돕는 게 목적입니다. "
        "질문은 필드가 비어서가 아니라, 팀이 그 결정을 잘 내릴 수 있도록 이끄는 것임을 명심하세요."
    ),
    "READY": (
        "필요한 정보가 거의 다 모인 단계입니다. "
        "빈 필드를 채우거나 바로 템플릿 생성을 제안해주세요."
    ),
}


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def _normalize_message(value: str) -> str:
    lowered = MATES_MENTION_PATTERN.sub(" ", str(value or "").strip().lower())
    compact = SHORT_MESSAGE_PATTERN.sub("", lowered)
    return compact.strip()


def _effective_user_message(state: AgentState) -> str:
    return _strip_mates_mention(state.get("user_message"))


def _is_trivial_message(user_message: str) -> bool:
    return _normalize_message(user_message) in TRIVIAL_MESSAGES


def _is_meaningful_fact(value: object) -> bool:
    cleaned = _clean_text(value)
    if not cleaned:
        return False
    normalized = cleaned.lower()
    if "@mates" in normalized or "?" in cleaned:
        return False
    return not any(
        kw in normalized
        for kw in ("모르겠", "모름", "없음", "없어", "미정", "tbd", "unknown")
    )


def _looks_like_accept_recommendation(message: str) -> bool:
    return bool(_ACCEPT_RECOMMENDATION_PATTERN.search(str(message or "")))


def _extract_recent_goal_recommendation(state: AgentState) -> str:
    recent = [
        _strip_mates_mention(msg)
        for msg in state.get("recent_messages", [])
        if _strip_mates_mention(msg)
    ]
    selected = _strip_mates_mention(state.get("selected_message"))
    context = "\n".join([*recent[-6:], selected]).strip()
    if not context:
        return ""

    option_b_matches = [match.strip() for match in _OPTION_B_PATTERN.findall(context)]
    for candidate in reversed(option_b_matches):
        if 10 <= len(candidate) <= 180:
            return candidate.strip(" .")

    quoted_matches = [match.strip() for match in _QUOTED_RECOMMENDATION_PATTERN.findall(context)]
    for candidate in reversed(quoted_matches):
        if any(keyword in candidate for keyword in ("프로토타입", "앱", "알림", "일정", "추천")):
            return candidate.strip(" .")
    return ""


def _build_acceptance_updates(
    state: AgentState,
    current_data: dict[str, object],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    user_message = _effective_user_message(state)
    if not _looks_like_accept_recommendation(user_message):
        return {}, {}
    if _is_valid_collected_value("goal", current_data.get("goal")):
        return {}, {}

    accepted_goal = _extract_recent_goal_recommendation(state)
    if not accepted_goal:
        return {}, {}

    return {"goal": accepted_goal}, {
        "goal": {
            "source": "accepted_recommendation",
            "raw_evidence": user_message,
            "intent": "correct_info",
        }
    }


def _message_explicitly_mentions_slot(message: str, slot: str) -> bool:
    pattern = _EXPLICIT_SLOT_MARKERS.get(slot)
    return bool(pattern and pattern.search(message or ""))


def _expected_slot_for_turn(
    state: AgentState,
    current_data: dict[str, object],
    current_phase: str,
) -> str:
    subject_missing = not _is_valid_collected_value("subject", current_data.get("subject"))
    problem_area_missing = not _is_meaningful_fact(current_data.get("problemArea"))
    target_user_missing = not _is_meaningful_fact(current_data.get("targetUser"))
    current_slot = str(
        state.get("current_slot") or state.get("next_question_field") or ""
    ).strip()
    if current_phase == "EXPLORE":
        if current_slot in {"problemArea", "targetUser", "subject"}:
            return current_slot
        if problem_area_missing:
            return "problemArea"
        if target_user_missing:
            return "targetUser"
        if subject_missing:
            return "subject"
        return ""
    if (
        current_slot == "subject"
        and subject_missing
        and current_phase in {"EXPLORE", "TOPIC_SET", "PROBLEM_DEFINE"}
    ):
        return current_slot
    if (
        current_phase in {"EXPLORE", "TOPIC_SET", "PROBLEM_DEFINE"}
        and subject_missing
    ):
        return "subject"
    if current_phase == "GATHER" and not _is_valid_collected_value(
        "goal", current_data.get("goal")
    ):
        return "goal"
    return ""


def _align_llm_updates_with_expected_slot(
    state: AgentState,
    *,
    current_data: dict[str, object],
    raw_data_updates: dict[str, object],
    candidate_sources: dict[str, dict[str, object]],
    current_phase: str,
) -> None:
    expected_slot = _expected_slot_for_turn(state, current_data, current_phase)
    if current_phase == "EXPLORE":
        allowed_by_slot = {
            "problemArea": {"problemArea"},
            "targetUser": {"targetUser"},
            "subject": {"subject"},
        }.get(expected_slot, set())
        blocked_fields = {
            key
            for key in raw_data_updates
            if key in _VALID_NEXT_FIELDS and key not in allowed_by_slot
        }
        for key in blocked_fields:
            raw_data_updates.pop(key, None)
            candidate_sources.pop(key, None)
        if blocked_fields:
            logger.info(
                "dropped_explore_premature_updates expected_slot=%s blocked=%s",
                expected_slot,
                sorted(blocked_fields),
            )
        if expected_slot in {"problemArea", "targetUser"}:
            return

    if expected_slot != "subject":
        return

    user_message = _effective_user_message(state)

    # subject가 없으면 goal/title을 subject로 재정렬
    if "subject" not in raw_data_updates:
        for source_field in ("goal", "title"):
            if source_field not in raw_data_updates:
                continue
            if _message_explicitly_mentions_slot(user_message, source_field):
                continue

            raw_data_updates["subject"] = raw_data_updates.pop(source_field)
            source_info = dict(candidate_sources.pop(source_field, {}))
            source_info["original_field"] = source_field
            candidate_sources["subject"] = source_info
            logger.info(
                "realigned_llm_update original_field=%s expected_field=subject phase=%s",
                source_field,
                current_phase,
            )
            return

    # subject가 있어도 같이 딸려온 goal은 제거 — subject 확정 전에 goal이 확정되면
    # _postprocess_ai_message 분기가 모두 빗나가 LLM 원문이 그대로 노출됨
    if current_phase in {"EXPLORE", "TOPIC_SET"} and "goal" in raw_data_updates:
        if not _message_explicitly_mentions_slot(user_message, "goal"):
            raw_data_updates.pop("goal")
            candidate_sources.pop("goal", None)
            logger.info(
                "dropped_premature_goal_extraction phase=%s",
                current_phase,
            )


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
    is_negative = state.get(
        "action_type"
    ) == "BTN_NO" or _is_topic_presence_negative_message(message_candidate)
    next_phase = "EXPLORE" if is_negative else "TOPIC_SET"
    logger.info(
        "initial_button_reset action=%s next_phase=%s",
        state.get("action_type"),
        next_phase,
    )
    return {
        "ai_message": "",
        "collected_data": {},
        "is_sufficient": False,
        "next_phase": next_phase,
    }


# ---------------------------------------------------------------------------
# Turn policy helpers
# ---------------------------------------------------------------------------


def _get_turn_policy(state: AgentState) -> TurnPolicy:
    return state.get("turn_policy", "ANSWER_THEN_ASK")


def _is_answer_only_turn(state: AgentState) -> bool:
    return _get_turn_policy(state) == "ANSWER_ONLY"


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


_CASUAL_CONFIRMATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\s*(?:이대로\s*)?(?:할래|할까|진행할까|확정할까|반영해도\s*될까)\s*,?\s*응\?\s*$"),
    re.compile(r"\s*(?:그걸로|이걸로|그렇게)\?\s*$"),
    re.compile(r"\s*응\?\s*$"),
)


def _remove_casual_confirmation_tail(message: str) -> str:
    cleaned = str(message or "").strip()
    if not cleaned:
        return cleaned
    for pattern in _CASUAL_CONFIRMATION_PATTERNS:
        if not pattern.search(cleaned):
            continue
        boundary = max(
            cleaned.rfind("."),
            cleaned.rfind("!"),
            cleaned.rfind("\n"),
        )
        if boundary >= 0:
            return cleaned[: boundary + 1].strip()
        return ""
    return cleaned


# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------


_HELP_REQUEST_PATTERN = re.compile(
    r"도움|어떻게|방법|모르겠|막막|뭘\s*해야|어디서|무엇부터",
    re.IGNORECASE,
)


def _is_help_request(user_message: str) -> bool:
    return bool(_HELP_REQUEST_PATTERN.search(str(user_message or "")))


def _should_skip_rag(state: AgentState) -> bool:
    user_message = _effective_user_message(state)
    selected = _strip_mates_mention(state.get("selected_message"))
    recent = [msg for msg in state.get("recent_messages", []) if str(msg).strip()]
    return _is_trivial_message(user_message) and not selected.strip() and not recent


def _should_use_rag(state: AgentState, phase: str, query: str) -> bool:
    if _should_skip_rag(state):
        return False
    query_text = str(query or "").strip()
    if not query_text:
        return False
    action = state.get("action_type")
    # 템플릿 생성 버튼: 항상 RAG 사용
    if action in {"BTN_PLAN", "BTN_DEV"}:
        return True
    # 대화(CHAT): GATHER 단계 이상에서만 RAG 사용 (EXPLORE는 너무 이른 단계)
    return phase in {"GATHER", "PROBLEM_DEFINE", "READY"}


def _select_rag_top_k(state: AgentState, phase: str, query: str) -> int:
    base_k = max(1, int(settings.RAG_TOP_K))
    if phase not in FAST_RAG_PHASES:
        return base_k
    fast_k = min(base_k, max(1, int(settings.RAG_CHAT_TOP_K)))
    if "?" in query or _is_help_request(query):
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


def _get_rag_filters(filter_key: str) -> dict[str, list[str]]:
    return RAG_FILTERS_BY_PHASE.get(filter_key, {})


def _fetch_rag_context(
    state: AgentState,
    phase: str,
    *,
    topics: list[str] | None = None,
    doc_types: list[str] | None = None,
) -> str:
    query = _build_rag_query(state)
    if not _should_use_rag(state, phase, query):
        logger.info("rag skipped phase=%s action=%s", phase, state.get("action_type"))
        return RAG_EMPTY_CONTEXT

    top_k = _select_rag_top_k(state, phase, query)
    cache_key = (phase, query, tuple(topics or []), tuple(doc_types or []), top_k)
    cached = RAG_CONTEXT_CACHE.get(cache_key)
    if cached:
        return cached

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
        RAG_CONTEXT_CACHE.pop(next(iter(RAG_CONTEXT_CACHE)), None)

    logger.info(
        "rag fetched %.2fs phase=%s top_k=%d chars=%d",
        perf_counter() - started_at,
        phase,
        top_k,
        len(context or ""),
    )
    return context or RAG_EMPTY_CONTEXT


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _prune_collected_data(data: dict[str, object]) -> dict[str, object]:
    return _normalize_request_collected_data(data)


def _is_valid_collected_value(key: str, value: object) -> bool:
    return _shared_is_valid_collected_value(key, value)


def _is_template_ready(current_data: dict[str, object]) -> bool:
    return _shared_is_template_ready(current_data)


def _build_missing_field_summary(current_data: dict[str, object]) -> str:
    missing = [
        f'- {meta["question"]}'
        for key, meta in GATHER_FIELD_GUIDE.items()
        if not _is_valid_collected_value(key, current_data.get(key))
    ]
    return "\n".join(missing) if missing else "- ?놁쓬"


# ---------------------------------------------------------------------------
# Decision helpers
# ---------------------------------------------------------------------------


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
    normalized_collected_data: dict[str, object],
    llm_updates_raw: dict[str, object],
    candidate_updates_merged: dict[str, object],
    approved_updates: dict[str, object],
    rejected_updates: dict[str, object],
    rejected_reasons: dict[str, str],
    decisions: dict[str, CandidateDecision] | None = None,
) -> None:
    logger.info(
        "collected_data_trace project_id=%s phase=%s before=%s after=%s",
        state.get("project_id"),
        state.get("current_phase"),
        normalized_collected_data,
        merge_collected_data(normalized_collected_data, approved_updates),
    )
    for key in candidate_updates_merged:
        decision = decisions.get(key) if decisions else None
        source = "llm" if key in llm_updates_raw else "merged"
        logger.info(
            "candidate_trace field=%s candidate=%r status=%s reason=%s source=%s",
            key,
            candidate_updates_merged.get(key),
            (
                "approved"
                if key in approved_updates
                else "rejected" if key in rejected_reasons else "ignored"
            ),
            rejected_reasons.get(key) or (decision.reason if decision else ""),
            source,
        )


def _decision_conflict_message(
    current_data: dict[str, object],
    candidate_updates: dict[str, object],
    decisions: dict[str, CandidateDecision],
) -> str:
    if not any(d.conflict_severity != "NONE" for d in decisions.values()):
        return ""
    effective_roles = candidate_updates.get("roles", current_data.get("roles"))
    effective_team_size = candidate_updates.get("teamSize", current_data.get("teamSize"))
    if not effective_roles or not effective_team_size:
        return ""
    return (
        f"역할({effective_roles})과 팀 인원({effective_team_size})이 맞지 않습니다. "
        "확인 후 다시 알려 주세요."
    )


# ---------------------------------------------------------------------------
# LLM decision core
# ---------------------------------------------------------------------------

LLM_DECISION_FALLBACK_MESSAGE = (
    "방금 답변을 처리하지 못했습니다. 말씀하신 내용을 다시 한 번 짧게 적어주세요."
)


def _call_llm_decision(
    state: AgentState,
    *,
    current_data: dict[str, object],
    phase: str,
) -> dict[str, object]:
    user_message = _effective_user_message(state)
    turn_policy = _get_turn_policy(state)
    can_ask = turn_policy in {"ANSWER_THEN_ASK", "ASK_ONLY"}

    rag_context = _fetch_rag_context(state, phase=phase, **_get_rag_filters(phase))
    recent_context = _build_recent_context(state)
    if phase == "EXPLORE":
        discovery_missing: list[str] = []
        if not _is_meaningful_fact(current_data.get("problemArea")):
            discovery_missing.append("- 해결하고 싶은 문제 상황")
        if not _is_meaningful_fact(current_data.get("targetUser")):
            discovery_missing.append("- 그 문제를 가장 자주 겪는 대상 사용자")
        if not _is_meaningful_fact(current_data.get("subject")):
            discovery_missing.append("- 사용자가 동의한 프로젝트 주제")
        missing_summary = "\n".join(discovery_missing) if discovery_missing else "- 없음"
    else:
        missing_summary = _build_missing_field_summary(current_data)
    phase_guidance = _PHASE_CONTEXT.get(phase, _PHASE_CONTEXT["GATHER"])
    ask_rule = (
        "필요하면 다음 질문 하나만 하세요."
        if can_ask
        else "질문은 하지 말고 답변만 하세요."
    )

    prompt = f"""당신은 초보 대학생 팀의 프로젝트 의사결정을 돕는 코치입니다.
팀이 스스로 결론에 이를 수 있도록 자연스럽게 대화를 이끌어 주세요.
데이터 수집은 대화의 부산물입니다. 먼저 팀이 결정을 내릴 수 있도록 도와주세요.

[?꾩옱 ?④퀎: {phase}]
{phase_guidance}

[현재 수집된 정보]
{json.dumps(current_data, ensure_ascii=False, indent=2)}

[팀이 아직 결정하지 않은 것들]
{missing_summary}

[최근 대화]
{recent_context}

[李멸퀬 ?먮즺]
{rag_context}

[사용자 메시지]
{user_message}

응답 규칙:
- 답변은 2~4문장 이내. 한 번에 하나의 항목만 다룬다. 다른 미결정 항목은 이번 턴에 먼저 언급하지 않는다.
- 추천이 필요하면 최대 1개를 기본값으로 제시한다. A/B/C 선택지를 나열하지 않는다.
- 사용자가 명확한 사실을 제공했으면 확인 질문으로 되묻지 말고 바로 정리한 뒤 다음에 필요한 항목을 자연스럽게 요청한다.
- '응?', '그걸로?', '할까?', '확정할까?' 같은 가벼운 확인형 종결은 사용하지 않는다.
- 추천안을 제시할 때도 "이대로 진행해도 됩니다"처럼 안내하고, 사용자가 수락하면 다음 항목으로 넘어간다.
- 고객님 말투가 아닌 친근한 코치 말투를 사용하세요.
- '좋아요', '알겠습니다', '반영하겠습니다' 같은 앞구문으로 시작하지 마세요.
- 이전 메시지와 동일하거나 유사한 내용을 반복하지 마세요.
- {ask_rule}
- updates: 이번 턴에서 사용자가 명시적으로 공유한 사실만 포함하세요.
- value에 사용자 문장을 그대로 복사하지 말고 문서에 어울리는 중립적이고 명확한 표현으로 rephrase하세요.
- dueDate value는 날짜만 넣으세요. 예: "6월 30일", "2026-06-30". "마감일을 ... 확정" 같은 문장은 넣지 마세요.
- teamSize value는 숫자만 넣으세요. 예: 3.
- roles value는 역할명만 짧게 나열하세요. 설명 문장, 괄호 설명, "팀원 N명으로 역할을 분담" 같은 접두어는 넣지 마세요.
- 사용자가 말하지 않은 사실은 추가하지 마세요.
- 질문, 요약 요청, 추천 요청, 확인 요청은 updates에 넣지 마세요.
- raw_evidence에는 실제 사용자 메시지에서 근거가 있는 짧은 인용문을 넣으세요.
- confidence가 높은 사실 공유가 아니면 is_user_provided_fact=false로 하세요.
- next_field: 팀이 다음에 결정해야 할 항목 이름. 없으면 빈 문자열
  (problemArea/targetUser/subject/title/goal/teamSize/roles/dueDate/deliverables 중 하나)
- 필드 이름을 언급하지 말 것. JSON만 출력 (다른 텍스트 없이 JSON만):
{{
  "intent": "provide_info | ask_help | ask_summary | request_next_step | correct_info | meta | other",
  "response_mode": "answer | answer_then_ask | ask_only",
  "ai_message": "응답 메시지",
  "updates": [
    {{
      "field": "goal",
      "value": "문서에 어울리게 정리된 값",
      "raw_evidence": "?ъ슜?먭? ?ㅼ젣濡?留먰븳 洹쇨굅",
      "confidence": 0.0,
      "is_user_provided_fact": false
    }}
  ],
  "next_field": "",
  "needs_clarification": false
}}"""

    response = _invoke_llm(
        structured_llm,
        prompt,
        label=f"llm_decision phase={phase}",
        response_format={"type": "json_object"},
    )

    if response is None:
        logger.warning(
            "llm_decision failed phase=%s user_message=%r", phase, user_message
        )
        return {
            "ai_message": LLM_DECISION_FALLBACK_MESSAGE,
            "data_updates": {},
            "update_sources": {},
            "next_field": choose_next_question_field(current_data, current_phase=phase),
            "llm_failed": True,
        }

    try:
        result = json.loads(str(response.content))
    except (JSONDecodeError, AttributeError):
        logger.warning(
            "llm_decision json parse failed phase=%s content=%r",
            phase,
            getattr(response, "content", ""),
        )
        return {
            "ai_message": str(getattr(response, "content", "")).strip()
            or LLM_DECISION_FALLBACK_MESSAGE,
            "data_updates": {},
            "update_sources": {},
            "next_field": "",
            "llm_failed": False,
        }

    if not result.get("updates") and isinstance(result.get("data_updates"), dict):
        result["updates"] = [
            {
                "field": key,
                "value": value,
                "raw_evidence": "",
                "confidence": 0.75,
                "is_user_provided_fact": True,
            }
            for key, value in dict(result.get("data_updates") or {}).items()
            if key in _VALID_NEXT_FIELDS and str(value or "").strip()
        ]
    if result.get("intent") not in {
        "provide_info",
        "ask_help",
        "ask_summary",
        "request_next_step",
        "correct_info",
        "meta",
        "other",
    }:
        result["intent"] = "other"
    if result.get("response_mode") not in {"answer", "answer_then_ask", "ask_only"}:
        result["response_mode"] = "answer_then_ask"
    if result.get("next_field") is None:
        result["next_field"] = ""

    try:
        decision = ConversationLLMDecision.model_validate(result)
    except Exception:
        logger.warning(
            "llm_decision schema validation failed phase=%s payload=%r",
            phase,
            result,
        )
        return {
            "ai_message": str(result.get("ai_message") or "").strip()
            or LLM_DECISION_FALLBACK_MESSAGE,
            "data_updates": {},
            "update_sources": {},
            "next_field": "",
            "llm_failed": False,
        }
    clean_updates: dict[str, object] = {}
    update_sources: dict[str, dict[str, object]] = {}
    for update in decision.updates:
        if (
            not update.is_user_provided_fact
            or update.field not in _VALID_NEXT_FIELDS
            or not str(update.value or "").strip()
        ):
            continue
        clean_updates[update.field] = update.value
        update_sources[update.field] = {
            "source": "llm_decision",
            "raw_evidence": update.raw_evidence,
            "confidence": update.confidence,
            "intent": decision.intent,
        }

    next_field_raw = str(decision.next_field or "").strip()
    next_field = next_field_raw if next_field_raw in _VALID_NEXT_FIELDS else ""

    return {
        "intent": decision.intent,
        "response_mode": decision.response_mode,
        "ai_message": decision.ai_message.strip() or LLM_DECISION_FALLBACK_MESSAGE,
        "data_updates": clean_updates,
        "update_sources": update_sources,
        "next_field": next_field,
        "needs_clarification": decision.needs_clarification,
        "llm_failed": False,
    }


def _apply_llm_message_policy(state: AgentState, ai_message: str) -> str:
    cleaned = str(ai_message or "").strip()
    if not cleaned:
        return cleaned
    cleaned = _remove_casual_confirmation_tail(cleaned)
    if _is_answer_only_turn(state):
        trimmed = _trim_trailing_question_lines(cleaned)
        return _truncate_ai_message(trimmed or cleaned)
    return _truncate_ai_message(cleaned)


def _apply_llm_updates(
    state: AgentState,
    *,
    current_data: dict[str, object],
    llm_result: dict[str, object],
    current_phase: str,
) -> tuple[
    dict[str, object],  # merged_data
    dict[str, object],  # approved_updates
    dict[str, object],  # rejected_updates
    dict[str, str],  # rejected_reasons
    dict[str, CandidateDecision],  # decisions
    list[str],  # followup_fields
    str,  # conflict_message
]:
    user_message = _effective_user_message(state)
    raw_data_updates = dict(llm_result.get("data_updates") or {})
    candidate_sources = dict(llm_result.get("update_sources") or {})
    _align_llm_updates_with_expected_slot(
        state,
        current_data=current_data,
        raw_data_updates=raw_data_updates,
        candidate_sources=candidate_sources,
        current_phase=current_phase,
    )
    acceptance_updates, acceptance_sources = _build_acceptance_updates(state, current_data)
    for key, value in acceptance_updates.items():
        raw_data_updates.setdefault(key, value)
        candidate_sources.setdefault(key, acceptance_sources.get(key, {}))
    for key, value in raw_data_updates.items():
        candidate_sources.setdefault(
            key,
            {"source": "llm_decision", "value": value},
        )
    (
        approved_updates,
        rejected_updates,
        rejected_reasons,
        decisions,
        followup_fields,
    ) = _evaluate_candidate_updates(
        current_data=current_data,
        candidate_sources=candidate_sources,
        candidate_updates=raw_data_updates,
        user_message=user_message,
        current_phase=current_phase,
        turn_type="llm_decision",
        recent_messages=state.get("recent_messages", []),
        selected_message=state.get("selected_message"),
    )
    merged_data = merge_collected_data(current_data, approved_updates)
    conflict_message = _decision_conflict_message(
        current_data, {**current_data, **approved_updates}, decisions
    )
    return (
        merged_data,
        approved_updates,
        rejected_updates,
        rejected_reasons,
        decisions,
        followup_fields,
        conflict_message,
    )


# ---------------------------------------------------------------------------
# Node: explore_problem_node
# ---------------------------------------------------------------------------


def explore_problem_node(state: AgentState):
    if _is_initial_button_selection(state):
        return _build_initial_button_reset_response(state)

    user_message = _effective_user_message(state)
    raw_current_data = dict(state.get("collected_data") or {})
    current_data = _prune_collected_data(raw_current_data)

    if _is_trivial_message(user_message) and not state.get("recent_messages"):
        return {
            "ai_message": "",
            "collected_data": raw_current_data,
            "is_sufficient": False,
            "next_phase": "EXPLORE",
        }

    llm_result = _call_llm_decision(state, current_data=current_data, phase="EXPLORE")
    (
        merged_data,
        approved_updates,
        rejected_updates,
        rejected_reasons,
        decisions,
        followup_fields,
        conflict_message,
    ) = _apply_llm_updates(
        state, current_data=current_data, llm_result=llm_result, current_phase="EXPLORE"
    )

    next_phase = derive_phase_from_collected_data(merged_data, current_phase="EXPLORE")
    is_sufficient = _is_template_ready(merged_data)
    llm_next_field = llm_result.get("next_field", "")
    next_question_field = llm_next_field or choose_next_question_field(
        merged_data,
        current_phase="EXPLORE",
        followup_fields=followup_fields,
        rejected_updates=rejected_updates,
    )

    logger.info(
        "explore llm_failed=%s approved=%s next_phase=%s",
        llm_result.get("llm_failed"),
        approved_updates,
        next_phase,
    )

    return {
        "ai_message": _apply_llm_message_policy(
            state, conflict_message or llm_result.get("ai_message", "")
        ),
        "collected_data": merged_data,
        "is_sufficient": is_sufficient,
        "next_phase": next_phase,
        "approved_updates": approved_updates,
        "rejected_updates": rejected_updates,
        "rejected_reasons": rejected_reasons,
        "followup_fields": followup_fields,
        "next_question_field": next_question_field,
    }


# ---------------------------------------------------------------------------
# Node: topic_exists_node
# ---------------------------------------------------------------------------


def topic_exists_node(state: AgentState):
    if _is_initial_button_selection(state):
        return _build_initial_button_reset_response(state)

    raw_current_data = dict(state.get("collected_data") or {})
    current_data = _prune_collected_data(raw_current_data)

    llm_result = _call_llm_decision(state, current_data=current_data, phase="TOPIC_SET")
    (
        merged_data,
        approved_updates,
        rejected_updates,
        rejected_reasons,
        decisions,
        followup_fields,
        conflict_message,
    ) = _apply_llm_updates(
        state,
        current_data=current_data,
        llm_result=llm_result,
        current_phase="TOPIC_SET",
    )

    next_phase = derive_phase_from_collected_data(
        merged_data, current_phase="TOPIC_SET"
    )
    is_sufficient = _is_template_ready(merged_data)
    llm_next_field = llm_result.get("next_field", "")
    next_question_field = llm_next_field or choose_next_question_field(
        merged_data,
        current_phase="TOPIC_SET",
        followup_fields=followup_fields,
        rejected_updates=rejected_updates,
    )

    logger.info(
        "topic_exists llm_failed=%s approved=%s rejected=%s next_phase=%s",
        llm_result.get("llm_failed"),
        approved_updates,
        rejected_updates,
        next_phase,
    )

    return {
        "ai_message": _apply_llm_message_policy(
            state, conflict_message or llm_result.get("ai_message", "")
        ),
        "collected_data": merged_data,
        "is_sufficient": is_sufficient,
        "next_phase": next_phase,
        "approved_updates": approved_updates,
        "rejected_updates": rejected_updates,
        "rejected_reasons": rejected_reasons,
        "followup_fields": followup_fields,
        "next_question_field": next_question_field,
    }


# ---------------------------------------------------------------------------
# Node: gather_information_node
# ---------------------------------------------------------------------------


def gather_information_node(state: AgentState):
    if _is_initial_button_selection(state):
        pruned = _prune_collected_data(state.get("collected_data") or {})
        if not _is_meaningful_fact(pruned.get("subject")):
            return _build_initial_button_reset_response(state)

    current_phase = str(state.get("current_phase") or "GATHER")
    raw_current_data = dict(state.get("collected_data") or {})
    current_data = _prune_collected_data(raw_current_data)

    llm_result = _call_llm_decision(
        state, current_data=current_data, phase=current_phase
    )
    (
        merged_data,
        approved_updates,
        rejected_updates,
        rejected_reasons,
        decisions,
        followup_fields,
        conflict_message,
    ) = _apply_llm_updates(
        state,
        current_data=current_data,
        llm_result=llm_result,
        current_phase=current_phase,
    )

    next_phase = derive_phase_from_collected_data(
        merged_data, current_phase=current_phase
    )
    is_sufficient = _is_template_ready(merged_data)
    llm_next_field = llm_result.get("next_field", "")
    next_question_field = llm_next_field or choose_next_question_field(
        merged_data,
        current_phase=current_phase,
        followup_fields=followup_fields,
        rejected_updates=rejected_updates,
    )

    logger.info(
        "gather llm_failed=%s approved=%s rejected=%s next_phase=%s is_sufficient=%s",
        llm_result.get("llm_failed"),
        approved_updates,
        rejected_updates,
        next_phase,
        is_sufficient,
    )
    _log_collected_data_trace(
        state,
        normalized_collected_data=current_data,
        llm_updates_raw=dict(llm_result.get("data_updates") or {}),
        candidate_updates_merged=dict(
            _shared_sanitize_candidate_updates(
                dict(llm_result.get("data_updates") or {}), current_data=current_data
            )
        ),
        approved_updates=approved_updates,
        rejected_updates=rejected_updates,
        rejected_reasons=rejected_reasons,
        decisions=decisions,
    )

    return {
        "ai_message": _apply_llm_message_policy(
            state, conflict_message or llm_result.get("ai_message", "")
        ),
        "collected_data": merged_data,
        "is_sufficient": is_sufficient,
        "next_phase": next_phase,
        "approved_updates": approved_updates,
        "rejected_updates": rejected_updates,
        "rejected_reasons": rejected_reasons,
        "followup_fields": followup_fields,
        "next_question_field": next_question_field,
    }
