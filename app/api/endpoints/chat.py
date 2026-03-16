# app/api/endpoints/chat.py
import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, root_validator

from app.ai.graph.state import TurnPolicy
from app.ai.graph.workflow import ai_app
from app.api.schemas.template import NotionTemplatePayload
from app.core.config import settings
from app.core.request_normalization import (
    normalize_action_type,
    normalize_collected_data,
    normalize_optional_string,
    normalize_phase,
    normalize_string_list,
)

router = APIRouter()
logger = logging.getLogger(__name__)
MATES_MENTION_PATTERN = re.compile(r"@mates\b", re.IGNORECASE)
AI_RESPONSE_MAX_CHARS = max(80, int(settings.AI_RESPONSE_MAX_CHARS))


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


def _strip_mates_mention(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    stripped = MATES_MENTION_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", stripped).strip()


def _has_mates_mention(request: AIChatRequest) -> bool:
    candidates = [request.content, request.selectedMessage]
    return any("@mates" in str(candidate or "").lower() for candidate in candidates)


def _truncate_content(content: str) -> str:
    text = str(content or "").strip()
    if len(text) <= AI_RESPONSE_MAX_CHARS:
        return text
    truncated = text[: AI_RESPONSE_MAX_CHARS - 1].rstrip()
    if not truncated:
        return text[:AI_RESPONSE_MAX_CHARS]
    if truncated[-1] in {".", "!", "?", "…"}:
        return truncated
    return f"{truncated}…"


def _derive_turn_policy(request: AIChatRequest) -> TurnPolicy:
    action = request.actionType
    phase = request.currentStatus
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
        return "CAPTURE_TITLE"
    if _has_mates_mention(request):
        return "ANSWER_ONLY"
    if phase in {"EXPLORE", "TOPIC_SET", "GATHER", "READY"}:
        return "ANSWER_THEN_ASK"
    return "ANSWER_ONLY"


@router.post("/", response_model=AIChatResponse)
async def process_chat(request: AIChatRequest):
    try:
        initial_state = {
            "project_id": str(request.roomId),
            "user_message": request.content,
            "action_type": request.actionType,
            "current_phase": request.currentStatus,
            "turn_policy": _derive_turn_policy(request),
            "collected_data": request.collectedData,
            "recent_messages": request.recentMessages,
            "selected_message": request.selectedMessage,
            "is_sufficient": False,
            "ai_message": "",
            "next_phase": request.currentStatus,
            "template_payload": None,
        }

        result = await asyncio.to_thread(ai_app.invoke, initial_state)

        return AIChatResponse(
            content=_truncate_content(result.get("ai_message", "")),
            suggestedQuestions=[],
            currentStatus=result.get("next_phase", request.currentStatus),
            isSufficient=result.get("is_sufficient", False),
            collectedData=result.get("collected_data", {}),
            notionTemplatePayload=result.get("template_payload"),
        )

    except Exception as exc:
        logger.exception("AI chat processing failed: %s", exc)
        raise HTTPException(status_code=500, detail="AI 처리 중 오류 발생")
