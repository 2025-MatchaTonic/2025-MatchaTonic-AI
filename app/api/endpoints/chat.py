# app/api/endpoints/chat.py
import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, root_validator

from app.ai.graph.topic_presence import _matches_topic_presence_button_message
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
    missing_collected_fields,
)
from app.ai.graph.state import TurnPolicy
from app.ai.graph.workflow import ai_app
from app.api.schemas.template import NotionTemplatePayload
from app.core.request_normalization import (
    normalize_action_type,
    normalize_collected_data,
    normalize_optional_string,
    normalize_phase,
    normalize_string_list,
)

router = APIRouter()
logger = logging.getLogger(__name__)

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
        payload["rawCollectedData"] = (
            dict(raw_collected_data) if isinstance(raw_collected_data, dict) else {}
        )

        payload["actionType"] = normalize_action_type(
            payload.get("actionType"), default="CHAT"
        )
        payload["currentStatus"] = normalize_phase(
            payload.get("currentStatus"), default="EXPLORE"
        )
        payload["collectedData"] = normalize_collected_data(
            payload.get("collectedData")
        )
        payload["projectName"] = normalize_optional_string(
            payload.get("projectName")
        ) or normalize_optional_string(payload["collectedData"].get("projectName"))

        selected_answers = normalize_string_list(payload.get("selectedAnswers"))
        payload["selectedAnswers"] = selected_answers

        recent_messages = normalize_string_list(payload.get("recentMessages"))
        if not recent_messages and selected_answers:
            recent_messages = list(selected_answers)
        payload["recentMessages"] = recent_messages

        selected_message = normalize_optional_string(payload.get("selectedMessage"))
        if selected_message is None and selected_answers:
            selected_message = selected_answers[-1]
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
    nextQuestionField: Optional[str] = None
    currentStatus: str
    isSufficient: bool
    collectedData: CollectedData
    approvedUpdates: Dict[str, Any] = Field(default_factory=dict)
    rejectedUpdates: Dict[str, Any] = Field(default_factory=dict)
    rejectedReasons: Dict[str, Any] = Field(default_factory=dict)
    followupFields: List[str] = Field(default_factory=list)
    notionTemplatePayload: Optional[NotionTemplatePayload] = None


QUESTION_BY_FIELD = {
    "subject": "프로젝트 주제를 한 줄로 적어주세요.",
    "title": "프로젝트 제목은 어떻게 정할까요?",
    "goal": "이 프로젝트 목표를 한 줄로 말해 주세요.",
    "teamSize": "현재 팀원은 총 몇 명인가요?",
    "roles": "각자 맡을 역할을 짧게 적어주세요.",
    "dueDate": "마감일이나 발표일은 언제인가요?",
    "deliverables": "최종 산출물은 무엇인가요?",
}

PHASE_FIELD_PRIORITY = {
    "PROBLEM_DEFINE": ["subject", "goal", "title"],
    "GATHER": ["teamSize", "roles", "dueDate", "deliverables", "goal"],
    "READY": ["goal", "teamSize", "roles", "dueDate", "deliverables"],
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
        return "CAPTURE_TITLE"
    if phase in {"EXPLORE", "TOPIC_SET", "PROBLEM_DEFINE", "GATHER", "READY"}:
        return "ANSWER_THEN_ASK"
    return "ANSWER_ONLY"


def _build_suggested_questions(
    *,
    phase: str,
    collected_data: CollectedData,
    rejected_updates: Dict[str, Any] | None,
    followup_fields: List[str] | None,
    next_field_override: str | None = None,
) -> List[str]:
    suggestions: List[str] = []
    seen_fields: set[str] = set()

    next_field = next_field_override or choose_next_question_field(
        collected_data,
        current_phase=phase,
        followup_fields=followup_fields or [],
        rejected_updates=rejected_updates or {},
    )
    if next_field:
        seen_fields.add(next_field)
        question = QUESTION_BY_FIELD.get(next_field)
        if question:
            suggestions.append(question)

    priority_fields: List[str] = []
    for field in followup_fields or []:
        if field not in seen_fields:
            priority_fields.append(field)
            seen_fields.add(field)
    for field in (rejected_updates or {}).keys():
        if field not in seen_fields:
            priority_fields.append(field)
            seen_fields.add(field)

    missing_fields = missing_collected_fields(collected_data)
    for field in PHASE_FIELD_PRIORITY.get(phase, PHASE_FIELD_PRIORITY.get("GATHER", [])):
        if field in missing_fields and field not in seen_fields:
            priority_fields.append(field)
            seen_fields.add(field)
    if phase == "READY":
        for field in missing_fields:
            if field not in seen_fields:
                priority_fields.append(field)
                seen_fields.add(field)

    for field in priority_fields:
        question = QUESTION_BY_FIELD.get(field)
        if not question:
            continue
        suggestions.append(question)
        if len(suggestions) == 3:
            break

    return suggestions


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
        next_question_field = _normalize_slot_name(result.get("next_question_field")) or choose_next_question_field(
            response_collected_data,
            current_phase=response_phase,
            followup_fields=followup_fields,
            rejected_updates=rejected_updates,
        )
        response_phase_trace = build_phase_derivation_trace(
            internal_response_collected_data,
            current_phase=request.currentStatus,
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
            result.get("ai_message", ""),
        )

        return AIChatResponse(
            content=_truncate_content(result.get("ai_message", "")),
            suggestedQuestions=_build_suggested_questions(
                phase=response_phase,
                collected_data=response_collected_data,
                rejected_updates=rejected_updates,
                followup_fields=followup_fields,
                next_field_override=next_question_field,
            ),
            nextQuestionField=next_question_field or None,
            currentStatus=response_phase,
            isSufficient=result.get("is_sufficient", False),
            collectedData=response_collected_data,
            approvedUpdates=approved_updates,
            rejectedUpdates=rejected_updates,
            rejectedReasons=rejected_reasons,
            followupFields=followup_fields,
            notionTemplatePayload=result.get("template_payload"),
        )
    except Exception as exc:
        logger.exception("AI chat processing failed: %s", exc)
        raise HTTPException(status_code=500, detail="AI 처리 중 오류 발생")
