# app/ai/graph/nodes.py
import json
import logging
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

RAG_EMPTY_CONTEXT = "(愿???덊띁?곗뒪瑜?李얠? 紐삵뻽?듬땲??)"
RAG_CACHE_MAX_ITEMS = 128
RAG_CONTEXT_CACHE: dict[tuple[str, str, tuple[str, ...], tuple[str, ...], int], str] = (
    {}
)

_VALID_NEXT_FIELDS = {
    "subject",
    "title",
    "goal",
    "targetUser",
    "teamSize",
    "roles",
    "dueDate",
    "deliverables",
}

_PHASE_CONTEXT: dict[str, str] = {
    "EXPLORE": (
        "?ъ슜?먭? ?꾩쭅 援ъ껜?곸씤 二쇱젣媛 ?놁뒿?덈떎. "
        "寃쏀뿕 湲곕컲 遺덊렪?⑥씠???꾩씠?붿뼱瑜??대걣?대궡?몄슂. "
        "?대? 二쇱젣媛 ?덈떎硫?媛꾨왂???곷룄濡??좊룄?섏꽭??"
    ),
    "TOPIC_SET": (
        "二쇱젣媛 ?쒖븞?먯?留??꾩쭅 ?뺤젙?섏? ?딆븯?듬땲?? "
        "二쇱젣瑜?紐낇솗???섍퀬 ?듭떖 臾몄젣 ?곸뿭???≪븘媛?몄슂."
    ),
    "PROBLEM_DEFINE": (
        "二쇱젣???덉쑝??紐⑺몴媛 ?꾩쭅 遺덈텇紐낇빀?덈떎. "
        "???꾨줈?앺듃濡?????ㅼ젣濡?留뚮뱾怨??띠? 寃껋쓣 ??以꾨줈 ?뚯뼱?댁꽭??"
    ),
    "GATHER": (
        "二쇱슂 ?뺣낫瑜??섏쭛?섎뒗 ?④퀎?낅땲?? "
        "teamSize, roles, dueDate, deliverables 以?鍮꾩뼱?덈뒗 ??ぉ???먯뿰?ㅻ읇寃?臾쇱뼱蹂댁꽭??"
    ),
    "READY": (
        "?꾩슂???뺣낫媛 嫄곗쓽 ??紐⑥씤 ?④퀎?낅땲?? "
        "?⑥? 怨듬갚??梨꾩슦嫄곕굹 ?쒗뵆由??앹꽦???덈궡?섏꽭??"
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


# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------


def _is_help_request(user_message: str) -> bool:
    return False


def _should_skip_rag(state: AgentState) -> bool:
    user_message = _effective_user_message(state)
    selected = _strip_mates_mention(state.get("selected_message"))
    recent = [msg for msg in state.get("recent_messages", []) if str(msg).strip()]
    return _is_trivial_message(user_message) and not selected.strip() and not recent


def _should_use_rag(state: AgentState, phase: str, query: str) -> bool:
    if _should_skip_rag(state):
        return False
    action = state.get("action_type")
    if action not in {"BTN_PLAN", "BTN_DEV"}:
        return False
    query_text = str(query or "").strip()
    if not query_text:
        return False
    return True


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
        f'- {key}: {meta["label"]}'
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
        f"??븷({effective_roles})怨?? ?몄썝({effective_team_size})??留욎? ?딆븘?? "
        "?뺤씤 ???ㅼ떆 ?뚮젮 二쇱꽭??"
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
    missing_summary = _build_missing_field_summary(current_data)
    phase_guidance = _PHASE_CONTEXT.get(phase, _PHASE_CONTEXT["GATHER"])
    ask_rule = (
        "?꾩슂?섎㈃ ?꾩냽 吏덈Ц????媛吏留??섏꽭??"
        if can_ask
        else "吏덈Ц? ?섏? 留먭퀬 ?듬?留??섏꽭??"
    )

    prompt = f"""?뱀떊? ?쒓뎅?대? ?곕뒗 ?뚰봽?몄썾?????AI ?꾨줈?앺듃 留ㅻ땲??낅땲??
?ъ슜??硫붿떆吏???먯뿰?ㅻ읇怨??ㅼ슜?곸쑝濡??듯븯?몄슂.

[?꾩옱 ?④퀎: {phase}]
{phase_guidance}

[?꾩옱 ?섏쭛???뺣낫]
{json.dumps(current_data, ensure_ascii=False, indent=2)}

[?꾩쭅 ?꾩슂???뺣낫]
{missing_summary}

[理쒓렐 ???
{recent_context}

[李멸퀬 ?먮즺]
{rag_context}

[?ъ슜??硫붿떆吏]
{user_message}

?듬? 洹쒖튃:
- ?ъ슜???붿껌??癒쇱? 吏곸젒 ?듯븯?몄슂. 200???대궡濡?媛꾧껐?섍쾶.
- 怨좉컼吏??留먰닾媛 ?꾨땶 ?ㅼ슜?곸씤 PM 留먰닾瑜??ъ슜?섏꽭??
- '醫뗭븘??, '?뚭쿋?듬땲??, '諛섏쁺?좉쾶??, '?뺣━?좉쾶??, '?뺤씤?좉쾶?? 媛숈? 愿?⑷뎄濡??쒖옉?섏? 留덉꽭??
- ?댁쟾 硫붿떆吏? ?숈씪?섍굅???좎궗???댁슜??諛섎났?섏? 留덉꽭??
- {ask_rule}
- updates: ?대쾲 ??붿뿉???ъ슜?먭? 紐낆떆?곸쑝濡??쒓났???ъ떎留??ы븿?섏꽭??
- value???ъ슜???먮Ц??洹몃?濡?蹂듭궗?섏? 留먭퀬 臾몄꽌???ㅼ뼱媛????덈뒗 以묐┰?곸씠怨?紐낇솗???쒗쁽?쇰줈 rephrase?섏꽭??
- ?ъ슜?먭? 留먰븯吏 ?딆? ?ъ떎??異붽??섏? 留덉꽭??
- 吏덈Ц, ?꾩? ?붿껌, 異붿쿇 ?붿껌, ?붿빟 ?붿껌? updates???ｌ? 留덉꽭??
- raw_evidence?먮뒗 ?ㅼ젣 ?ъ슜??硫붿떆吏?먯꽌 洹쇨굅媛 ?섎뒗 吏㏃? ?먮Ц???ｌ쑝?몄슂.
- confidence媛 ??굅???ъ떎 ?쒓났???꾨땲硫?is_user_provided_fact=false濡??먯꽭??
- next_field: ?ㅼ쓬???뺤씤???꾨뱶 ?대쫫. ?놁쑝硫?鍮?臾몄옄??
  (subject/title/goal/targetUser/teamSize/roles/dueDate/deliverables 以??섎굹)
- ?꾨뱶 蹂??紐낆쓣 ?멸툒?섏? ?딆쓣 寃?JSON 異쒕젰 (?ㅻⅨ ?띿뒪???놁씠 JSON留?:
{{
  "intent": "provide_info | ask_help | ask_summary | request_next_step | correct_info | meta | other",
  "response_mode": "answer | answer_then_ask | ask_only",
  "ai_message": "?쒓뎅???듬?",
  "updates": [
    {{
      "field": "goal",
      "value": "臾몄꽌???????덇쾶 ?뺣━??媛?,
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
