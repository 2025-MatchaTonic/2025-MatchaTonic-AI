# app/api/endpoints/chat.py
import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, root_validator

from app.ai.graph.topic_presence import (
    _is_topic_presence_negative_message,
    _matches_topic_presence_button_message,
)
from app.ai.graph.text_support import (
    strip_mates_mention as _strip_mates_mention,
    truncate_message as _truncate_content,
)
from app.ai.graph.collected_data import (
    CollectedData,
    build_approved_collected_data_snapshot,
    build_public_update_snapshot,
    build_phase_derivation_trace,
    choose_next_question_field,
    derive_phase_from_collected_data,
)
from app.ai.graph.state import TurnPolicy
from app.ai.graph.workflow import ai_app
from app.core.request_normalization import (
    normalize_action_type,
    normalize_collected_data,
    normalize_optional_string,
    normalize_phase,
    normalize_string_list,
)

router = APIRouter()
logger = logging.getLogger(__name__)

CHAT_RESPONSE_MAX_CHARS = 900
_EXECUTION_FACT_FIELDS = {"goal", "teamSize", "roles", "dueDate", "deliverables"}

_VALID_SLOT_FIELDS = {
    "subject",
    "title",
    "goal",
    "targetUser",
    "teamSize",
    "roles",
    "dueDate",
    "deliverables",
}

_TOP_LEVEL_COLLECTED_FIELDS = _VALID_SLOT_FIELDS | {
    "problemArea",
    "targetFacility",
    "targetUser",
}

_ASSISTANT_MESSAGE_PREFIXES = (
    "좋아요",
    "좋습니다",
    "알겠습니다",
    "현재까지",
    "아직 확정된",
    "방금 말씀은",
    "제목은 명시적으로",
    "목표가 없을 땐",
    "먼저 ",
    "다음으로 ",
)

_ASSISTANT_MESSAGE_SNIPPETS = (
    "반영할게요",
    "정리할게요",
    "확인할게요",
    "가볼게요",
    "해볼까요",
    "말해 주세요",
    "적어주세요",
    "알려주세요",
    "보내주세요",
    "정해볼까요",
    "어떻게 가져갈 생각인가요",
    "한 줄로 말해 주세요",
)


def _normalize_slot_name(value: object) -> str | None:
    cleaned = normalize_optional_string(value)
    if cleaned in _VALID_SLOT_FIELDS:
        return cleaned
    return None


def _looks_like_assistant_authored_message(message: str | None) -> bool:
    cleaned = normalize_optional_string(message)
    if not cleaned:
        return False
    if cleaned.startswith(_ASSISTANT_MESSAGE_PREFIXES):
        return True
    if any(snippet in cleaned for snippet in _ASSISTANT_MESSAGE_SNIPPETS):
        return True
    if "예:" in cleaned and ("\n" in cleaned or len(cleaned) > 30):
        return True
    if re.search(r"(?:입니다|이에요|예요)\.\s*(?:수정하려면|이제|다음으로|아직)", cleaned):
        return True
    return False


def _should_promote_selected_message_to_content(action_type: str, message: str | None) -> bool:
    cleaned = normalize_optional_string(message)
    if not cleaned:
        return False
    if action_type == "CHAT" and _looks_like_assistant_authored_message(cleaned):
        return False
    return True


def _normalize_chat_collected_data(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(normalize_collected_data(data))

    # Spring currently sends the chat room/project display name as collectedData.title.
    # In chat collection, subject is the reliable topic anchor; title-only payloads
    # should not make the AI think the user already has a project topic.
    subject = normalize_optional_string(normalized.get("subject"))
    if subject:
        normalized["title"] = subject
        return normalized

    if "title" in normalized and not any(field in normalized for field in _EXECUTION_FACT_FIELDS):
        normalized.pop("title", None)

    return normalized


def _build_initial_button_message(content: str, next_phase: str) -> str:
    if next_phase == "EXPLORE" or _is_topic_presence_negative_message(content):
        return (
            "괜찮아요. 아직 주제가 없어도 됩니다. 관심 분야나 최근에 불편했던 경험을 "
            "한 줄로 말해 주세요."
        )
    return "좋아요. 준비한 프로젝트 주제를 한 줄로 알려주세요."


def _resolve_next_question_field(
    *,
    phase: str,
    collected_data: Dict[str, Any],
    approved_updates: Dict[str, Any],
    rejected_updates: Dict[str, Any],
    followup_fields: List[str],
    proposed_field: str | None,
) -> str:
    if phase == "GATHER" and "goal" not in collected_data and "goal" not in approved_updates:
        return "goal"
    if "goal" in approved_updates and "roles" not in collected_data:
        return "roles"
    return _normalize_slot_name(proposed_field) or choose_next_question_field(
        collected_data,
        current_phase=phase,
        followup_fields=followup_fields,
        rejected_updates=rejected_updates,
    )


def _postprocess_ai_message(
    *,
    message: str,
    phase: str,
    approved_updates: Dict[str, Any],
    collected_data: Dict[str, Any],
    next_question_field: str,
    content: str,
) -> str:
    cleaned = str(message or "").strip()
    if not cleaned and phase in {"EXPLORE", "TOPIC_SET"} and _matches_topic_presence_button_message(content):
        return _build_initial_button_message(content, phase)

    if "goal" in approved_updates and next_question_field == "roles":
        team_size = collected_data.get("teamSize")
        suffix = (
            f" {team_size}인 기준으로 각자 맡을 역할을 알려주세요."
            if team_size
            else " 다음으로 역할 분담을 알려주세요."
        )
        return "목표를 확정했습니다." + suffix

    if "teamSize" in approved_updates and next_question_field == "goal":
        return f"팀 인원 {approved_updates['teamSize']}명 확인했습니다. 다음으로 프로젝트 목표를 한 줄로 정해 주세요."

    return cleaned


class AIChatRequest(BaseModel):
    roomId: int
    content: str = ""
    actionType: str = "CHAT"
    currentStatus: str = "EXPLORE"
    projectName: Optional[str] = None
    rawCollectedData: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    collectedData: CollectedData = Field(default_factory=dict)
    recentMessages: List[str] = Field(default_factory=list)
    selectedMessage: Optional[str] = None
    selectedAnswers: List[str] = Field(default_factory=list)
    currentSlot: Optional[str] = None
    nextQuestionField: Optional[str] = None

    @root_validator(pre=True)
    def normalize_spring_compatible_payload(
        cls, values: Dict[str, Any]
    ) -> Dict[str, Any]:
        payload = dict(values or {})

        if payload.get("roomId") is None and payload.get("projectId") is not None:
            payload["roomId"] = payload["projectId"]

        raw_collected_data = payload.get("collectedData")
        if not isinstance(raw_collected_data, dict):
            raw_collected_data = {}
        raw_collected_data = dict(raw_collected_data)

        for field in _TOP_LEVEL_COLLECTED_FIELDS:
            if field in payload and field not in raw_collected_data:
                raw_collected_data[field] = payload[field]

        payload["collectedData"] = raw_collected_data
        payload["rawCollectedData"] = (
            dict(raw_collected_data) if isinstance(raw_collected_data, dict) else {}
        )

        payload["actionType"] = normalize_action_type(
            payload.get("actionType"), default="CHAT"
        )
        payload["currentStatus"] = normalize_phase(
            payload.get("currentStatus"), default="EXPLORE"
        )
        payload["collectedData"] = _normalize_chat_collected_data(
            payload.get("collectedData")
        )
        payload["projectName"] = normalize_optional_string(
            payload.get("projectName")
        ) or normalize_optional_string(payload["collectedData"].get("projectName"))

        selected_answers = normalize_string_list(payload.get("selectedAnswers"))
        payload["selectedAnswers"] = selected_answers

        recent_messages = normalize_string_list(payload.get("recentMessages"))
        if not recent_messages and selected_answers:
            # AI가 생성한 버튼 텍스트가 사용자 발화로 오염되지 않도록 필터링.
            # 필터 후 빈 경우 원본 유지(백엔드가 이미 구분해서 보낸 경우).
            user_authored = [a for a in selected_answers if not _looks_like_assistant_authored_message(a)]
            recent_messages = user_authored if user_authored else list(selected_answers)
        payload["recentMessages"] = recent_messages

        selected_message = normalize_optional_string(payload.get("selectedMessage"))
        if selected_message is None and selected_answers:
            # 마지막 선택지를 selectedMessage로 승격할 때도 AI 텍스트 제외.
            user_answer = next(
                (a for a in reversed(selected_answers) if not _looks_like_assistant_authored_message(a)),
                selected_answers[-1],
            )
            selected_message = user_answer
        payload["selectedMessage"] = selected_message
        payload["currentSlot"] = _normalize_slot_name(payload.get("currentSlot"))
        payload["nextQuestionField"] = _normalize_slot_name(payload.get("nextQuestionField"))

        content = normalize_optional_string(payload.get("content")) or ""
        if not content:
            if _should_promote_selected_message_to_content(payload["actionType"], selected_message):
                content = selected_message
            elif selected_answers:
                promoted_answers = [
                    answer
                    for answer in selected_answers
                    if _should_promote_selected_message_to_content(payload["actionType"], answer)
                ]
                if promoted_answers:
                    content = "\n".join(promoted_answers)
        payload["content"] = content

        return payload


class AIChatResponse(BaseModel):
    content: str
    suggestedQuestions: List[str]
    currentStatus: str
    isSufficient: bool
    collectedData: CollectedData


PHASE_FIELD_PRIORITY = {
    "PROBLEM_DEFINE": ["subject", "goal", "title"],
    "GATHER": ["teamSize", "roles", "dueDate", "deliverables", "goal"],
    "READY": ["goal", "teamSize", "roles", "dueDate", "deliverables"],
}

SUGGESTED_QUESTIONS_BY_FIELD = {
    "subject": ["어떤 주제로 진행할까요?"],
    "title": ["프로젝트 제목을 한 줄로 적어주세요."],
    "goal": ["이 프로젝트 목표를 한 줄로 말해 주세요."],
    "targetUser": ["주요 사용자는 누구인가요?"],
    "teamSize": ["현재 팀원은 총 몇 명인가요?", "3명", "4명", "5명 이상"],
    "roles": ["각자 맡을 역할을 짧게 적어주세요."],
    "dueDate": ["마감일이나 발표일은 언제인가요?", "1주 안", "2주 안"],
    "deliverables": ["최종 산출물은 무엇인가요?", "기획서", "MVP", "발표 자료"],
}


def _derive_effective_phase(request: AIChatRequest) -> str:
    return derive_phase_from_collected_data(
        request.collectedData,
        current_phase=request.currentStatus,
    )


def _derive_turn_policy(request: AIChatRequest) -> TurnPolicy:
    action = request.actionType
    phase = _derive_effective_phase(request)
    current_topic = str(request.collectedData.get("subject", "")).strip()
    if not current_topic and phase not in {"TOPIC_SET", "PROBLEM_DEFINE"}:
        current_topic = str(request.collectedData.get("title", "")).strip()
    effective_message = _strip_mates_mention(request.content)

    if action == "BTN_NO":
        return "ASK_ONLY"
    if action in {"BTN_YES", "BTN_GO_DEF"}:
        return "ASK_ONLY"
    if action in {"BTN_PLAN", "BTN_DEV"}:
        return "ANSWER_ONLY"
    if action != "CHAT":
        return "ANSWER_ONLY"
    if effective_message and _matches_topic_presence_button_message(effective_message):
        return "ASK_ONLY"

    if phase == "TOPIC_SET" and not current_topic and not effective_message:
        return "ASK_ONLY"
    if phase == "TOPIC_SET" and not current_topic and effective_message:
        if _matches_topic_presence_button_message(effective_message):
            return "ASK_ONLY"
        return "ANSWER_THEN_ASK"
    if phase in {"EXPLORE", "TOPIC_SET", "PROBLEM_DEFINE", "GATHER", "READY"}:
        return "ANSWER_THEN_ASK"
    return "ANSWER_ONLY"


def _build_suggested_questions(
    *,
    phase: str = "GATHER",
    collected_data: Dict[str, Any] | None = None,
    rejected_updates: Dict[str, Any] | None = None,
    followup_fields: List[str] | None = None,
    next_question_field: str | None = None,
) -> List[str]:
    field = _normalize_slot_name(next_question_field) or choose_next_question_field(
        collected_data or {},
        current_phase=phase,
        followup_fields=followup_fields or [],
        rejected_updates=rejected_updates or {},
    )
    return list(SUGGESTED_QUESTIONS_BY_FIELD.get(field, []))


@router.post("/", response_model=AIChatResponse)
async def process_chat(request: AIChatRequest):
    try:
        effective_phase = _derive_effective_phase(request)
        request_phase_trace = build_phase_derivation_trace(
            request.collectedData,
            current_phase=request.currentStatus,
        )
        logger.info(
            "chat request room=%s project_id=%s action=%s current_status=%s effective_phase=%s content=%r raw_collected_keys=%s normalized_collected_keys=%s collected_data=%s recent_count=%d selected_message=%r",
            request.roomId,
            request.roomId,
            request.actionType,
            request.currentStatus,
            effective_phase,
            request.content,
            sorted(request.rawCollectedData.keys()),
            sorted(request.collectedData.keys()),
            request.collectedData,
            len(request.recentMessages),
            request.selectedMessage,
        )
        logger.info(
            "chat phase_trace room=%s stage=request trace=%s",
            request.roomId,
            request_phase_trace,
        )
        initial_state = {
            "project_id": str(request.roomId),
            "user_message": request.content,
            "action_type": request.actionType,
            "current_phase": effective_phase,
            "turn_policy": _derive_turn_policy(request),
            "project_name": request.projectName,
            "collected_data": request.collectedData,
            "recent_messages": request.recentMessages,
            "selected_message": request.selectedMessage,
            "problem_area": normalize_optional_string(request.collectedData.get("problemArea")),
            "target_facility": normalize_optional_string(request.collectedData.get("targetFacility")),
            "current_slot": request.currentSlot or request.nextQuestionField,
            "is_sufficient": False,
            "ai_message": "",
            "next_phase": effective_phase,
            "next_question_field": request.nextQuestionField,
            "template_payload": None,
        }

        result = await asyncio.to_thread(ai_app.invoke, initial_state)
        response_phase = result.get("next_phase", effective_phase)
        internal_response_collected_data = result.get("collected_data", request.collectedData)
        response_collected_data = build_approved_collected_data_snapshot(
            internal_response_collected_data
        )
        response_subject = response_collected_data.get("subject")
        if "title" not in response_collected_data and isinstance(response_subject, str):
            response_collected_data["title"] = response_subject
        approved_updates = build_public_update_snapshot(
            result.get("approved_updates", {}),
            current_data=request.collectedData,
        )
        rejected_updates = build_public_update_snapshot(
            result.get("rejected_updates", {}),
            current_data=request.collectedData,
        )
        rejected_reasons = {
            key: value
            for key, value in dict(result.get("rejected_reasons", {})).items()
            if key in rejected_updates
        }
        followup_fields = result.get("followup_fields", [])
        next_question_field = _resolve_next_question_field(
            phase=response_phase,
            collected_data=response_collected_data,
            approved_updates=approved_updates,
            rejected_updates=rejected_updates,
            followup_fields=followup_fields,
            proposed_field=result.get("next_question_field"),
        )
        response_message = _postprocess_ai_message(
            message=result.get("ai_message", ""),
            phase=response_phase,
            approved_updates=approved_updates,
            collected_data=response_collected_data,
            next_question_field=next_question_field,
            content=request.content,
        )
        response_phase_trace = build_phase_derivation_trace(
            internal_response_collected_data,
            current_phase=response_phase,
        )
        logger.info(
            "chat decision room=%s phase=%s approved_updates=%s rejected_updates=%s rejected_reasons=%s next_question_field=%s",
            request.roomId,
            response_phase,
            approved_updates,
            rejected_updates,
            rejected_reasons,
            next_question_field,
        )
        logger.info(
            "chat phase_trace room=%s stage=response trace=%s node_phase=%s",
            request.roomId,
            response_phase_trace,
            response_phase,
        )
        logger.info(
            "chat response room=%s next_phase=%s is_sufficient=%s collected_data=%s public_collected_data=%s ai_message=%r",
            request.roomId,
            response_phase,
            result.get("is_sufficient", False),
            internal_response_collected_data,
            response_collected_data,
            response_message,
        )

        return AIChatResponse(
            content=_truncate_content(response_message, max_chars=CHAT_RESPONSE_MAX_CHARS),
            suggestedQuestions=_build_suggested_questions(
                phase=response_phase,
                collected_data=response_collected_data,
                rejected_updates=rejected_updates,
                followup_fields=followup_fields,
                next_question_field=next_question_field,
            ),
            currentStatus=response_phase,
            isSufficient=result.get("is_sufficient", False),
            collectedData=response_collected_data,
        )
    except Exception as exc:
        logger.exception("AI chat processing failed: %s", exc)
        raise HTTPException(status_code=500, detail="AI 처리 중 오류 발생")
