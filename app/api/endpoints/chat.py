# app/api/endpoints/chat.py
import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, root_validator

from app.ai.graph.nodes import _matches_topic_presence_button_message
from app.ai.graph.text_support import (
    strip_mates_mention as _strip_mates_mention,
    truncate_message as _truncate_content,
)
from app.ai.graph.collected_data import CollectedData, derive_phase_from_collected_data
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


class AIChatRequest(BaseModel):
    roomId: int
    content: str = ""
    actionType: str = "CHAT"
    currentStatus: str = "EXPLORE"
    collectedData: CollectedData = Field(default_factory=dict)
    recentMessages: List[str] = Field(default_factory=list)
    selectedMessage: Optional[str] = None
    selectedAnswers: List[str] = Field(default_factory=list)

    @root_validator(pre=True)
    def normalize_spring_compatible_payload(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(values or {})

        if payload.get("roomId") is None and payload.get("projectId") is not None:
            payload["roomId"] = payload["projectId"]

        payload["actionType"] = normalize_action_type(payload.get("actionType"), default="CHAT")
        payload["currentStatus"] = normalize_phase(payload.get("currentStatus"), default="EXPLORE")
        payload["collectedData"] = normalize_collected_data(payload.get("collectedData"))

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

        content = normalize_optional_string(payload.get("content")) or ""
        if not content:
            if selected_message:
                content = selected_message
            elif selected_answers:
                content = "\n".join(selected_answers)
        payload["content"] = content

        return payload


class AIChatResponse(BaseModel):
    content: str
    suggestedQuestions: List[str]
    currentStatus: str
    isSufficient: bool
    collectedData: CollectedData
    notionTemplatePayload: Optional[NotionTemplatePayload] = None


def _derive_effective_phase(request: AIChatRequest) -> str:
    return derive_phase_from_collected_data(
        request.collectedData,
        current_phase=request.currentStatus,
    )


def _derive_turn_policy(request: AIChatRequest) -> TurnPolicy:
    action = request.actionType
    phase = _derive_effective_phase(request)
    current_title = str(request.collectedData.get("title", "")).strip()
    effective_message = _strip_mates_mention(request.content)

    if action == "BTN_NO":
        return "ASK_ONLY"
    if action in {"BTN_YES", "BTN_GO_DEF"}:
        return "ASK_ONLY"
    if action in {"BTN_PLAN", "BTN_DEV"}:
        return "ANSWER_ONLY"
    if action != "CHAT":
        return "ANSWER_ONLY"

    if phase == "TOPIC_SET" and not current_title and not effective_message:
        return "ASK_ONLY"
    if phase == "TOPIC_SET" and not current_title and effective_message:
        if _matches_topic_presence_button_message(effective_message):
            return "ASK_ONLY"
        return "CAPTURE_TITLE"
    if phase in {"EXPLORE", "TOPIC_SET", "GATHER", "READY"}:
        return "ANSWER_THEN_ASK"
    return "ANSWER_ONLY"


@router.post("/", response_model=AIChatResponse)
async def process_chat(request: AIChatRequest):
    try:
        effective_phase = _derive_effective_phase(request)
        logger.info(
            "chat request room=%s action=%s current_status=%s effective_phase=%s content=%r collected_data=%s recent_count=%d selected_message=%r",
            request.roomId,
            request.actionType,
            request.currentStatus,
            effective_phase,
            request.content,
            request.collectedData,
            len(request.recentMessages),
            request.selectedMessage,
        )
        initial_state = {
            "project_id": str(request.roomId),
            "user_message": request.content,
            "action_type": request.actionType,
            "current_phase": effective_phase,
            "turn_policy": _derive_turn_policy(request),
            "collected_data": request.collectedData,
            "recent_messages": request.recentMessages,
            "selected_message": request.selectedMessage,
            "is_sufficient": False,
            "ai_message": "",
            "next_phase": effective_phase,
            "template_payload": None,
        }

        result = await asyncio.to_thread(ai_app.invoke, initial_state)
        response_phase = result.get("next_phase", effective_phase)
        response_collected_data = result.get("collected_data", request.collectedData)
        logger.info(
            "chat response room=%s next_phase=%s is_sufficient=%s collected_data=%s ai_message=%r",
            request.roomId,
            response_phase,
            result.get("is_sufficient", False),
            response_collected_data,
            result.get("ai_message", ""),
        )

        return AIChatResponse(
            content=_truncate_content(result.get("ai_message", "")),
            suggestedQuestions=[],
            currentStatus=response_phase,
            isSufficient=result.get("is_sufficient", False),
            collectedData=response_collected_data,
            notionTemplatePayload=result.get("template_payload"),
        )

    except Exception as exc:
        logger.exception("AI chat processing failed: %s", exc)
        raise HTTPException(status_code=500, detail="AI 처리 중 오류 발생")
