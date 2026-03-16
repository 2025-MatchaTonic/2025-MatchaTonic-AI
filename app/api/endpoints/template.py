import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, root_validator

from app.ai.services.template_generation import generate_dev_template, generate_plan_template
from app.api.schemas.template import NotionTemplatePayload
from app.core.request_normalization import (
    normalize_collected_data,
    normalize_optional_string,
    normalize_phase,
    normalize_string_list,
)

router = APIRouter()
logger = logging.getLogger(__name__)


class TemplateGenerateRequest(BaseModel):
    roomId: int
    templateType: Literal["plan", "dev"] = "plan"
    currentStatus: str = "READY"
    content: str = ""
    collectedData: Dict[str, str] = Field(default_factory=dict)
    recentMessages: List[str] = Field(default_factory=list)
    selectedMessage: Optional[str] = None
    selectedAnswers: List[str] = Field(default_factory=list)
    actionType: Optional[str] = None

    @root_validator(pre=True)
    def normalize_spring_compatible_payload(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(values or {})

        if payload.get("roomId") is None and payload.get("projectId") is not None:
            payload["roomId"] = payload["projectId"]

        action_type = str(payload.get("actionType") or "").strip().upper()
        template_type = str(payload.get("templateType") or "").strip().lower()
        if not template_type:
            if action_type == "BTN_DEV":
                payload["templateType"] = "dev"
            elif action_type == "BTN_PLAN":
                payload["templateType"] = "plan"

        payload["currentStatus"] = normalize_phase(payload.get("currentStatus"), default="READY")
        payload["collectedData"] = normalize_collected_data(payload.get("collectedData"))

        selected_answers = normalize_string_list(payload.get("selectedAnswers"))
        payload["selectedAnswers"] = selected_answers

        cleaned_recent_messages = normalize_string_list(payload.get("recentMessages"))
        if not cleaned_recent_messages and selected_answers:
            cleaned_recent_messages = list(selected_answers)
        payload["recentMessages"] = cleaned_recent_messages

        selected_message = normalize_optional_string(payload.get("selectedMessage"))
        if selected_message is None:
            selected_message = selected_answers[-1] if selected_answers else None
        payload["selectedMessage"] = selected_message

        content = normalize_optional_string(payload.get("content")) or ""
        if not content:
            if selected_message:
                content = selected_message
            elif selected_answers:
                content = "\n".join(selected_answers)
        payload["content"] = content

        return payload


class TemplateGenerateResponse(BaseModel):
    content: str
    currentStatus: str
    notionTemplatePayload: NotionTemplatePayload


def _build_template_state(request: TemplateGenerateRequest) -> dict:
    user_message = request.content.strip()
    if not user_message:
        user_message = (
            "개발 템플릿 생성해줘"
            if request.templateType == "dev"
            else "기획 템플릿 생성해줘"
        )

    return {
        "project_id": str(request.roomId),
        "user_message": user_message,
        "action_type": "BTN_DEV" if request.templateType == "dev" else "BTN_PLAN",
        "current_phase": request.currentStatus,
        "turn_policy": "ANSWER_ONLY",
        "collected_data": request.collectedData,
        "recent_messages": request.recentMessages,
        "selected_message": request.selectedMessage,
        "is_sufficient": True,
        "ai_message": "",
        "next_phase": request.currentStatus,
        "template_payload": None,
    }


def _run_template_generation(request: TemplateGenerateRequest) -> tuple[dict, dict]:
    state = _build_template_state(request)
    result = (
        generate_dev_template(state)
        if request.templateType == "dev"
        else generate_plan_template(state)
    )
    payload = result.get("template_payload")
    if payload is None:
        raise RuntimeError("template generation returned no payload")
    return result, payload


@router.post("/", response_model=TemplateGenerateResponse)
async def generate_template(request: TemplateGenerateRequest):
    try:
        result, payload = _run_template_generation(request)
        return TemplateGenerateResponse(
            content=result.get("ai_message", ""),
            currentStatus=result.get("next_phase", request.currentStatus),
            notionTemplatePayload=payload,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Template generation failed: %s", exc)
        raise HTTPException(status_code=500, detail="템플릿 생성 중 오류가 발생했습니다.")


@router.post("/spring", response_model=NotionTemplatePayload)
async def generate_template_for_spring(request: TemplateGenerateRequest):
    try:
        _, payload = _run_template_generation(request)
        return payload
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Spring template generation failed: %s", exc)
        raise HTTPException(status_code=500, detail="템플릿 생성 중 오류가 발생했습니다.")
