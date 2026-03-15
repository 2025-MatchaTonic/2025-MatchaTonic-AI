# app/api/endpoints/chat.py
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, root_validator

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
    collectedData: Dict[str, str] = Field(default_factory=dict)
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
    collectedData: Dict[str, str]
    notionTemplatePayload: Optional[NotionTemplatePayload] = None


@router.post("/", response_model=AIChatResponse)
async def process_chat(request: AIChatRequest):
    try:
        initial_state = {
            "project_id": str(request.roomId),
            "user_message": request.content,
            "action_type": request.actionType,
            "current_phase": request.currentStatus,
            "collected_data": request.collectedData,
            "recent_messages": request.recentMessages,
            "selected_message": request.selectedMessage,
            "is_sufficient": False,
            "ai_message": "",
            "next_phase": request.currentStatus,
            "template_payload": None,
        }

        result = ai_app.invoke(initial_state)

        return AIChatResponse(
            content=result.get("ai_message", ""),
            suggestedQuestions=[],
            currentStatus=result.get("next_phase", request.currentStatus),
            isSufficient=result.get("is_sufficient", False),
            collectedData=result.get("collected_data", {}),
            notionTemplatePayload=result.get("template_payload"),
        )

    except Exception as exc:
        logger.exception("AI chat processing failed: %s", exc)
        raise HTTPException(status_code=500, detail="AI 처리 중 오류 발생")
