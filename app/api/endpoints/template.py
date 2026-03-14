from typing import Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ai.graph.collected_data import COLLECTED_DATA_FIELDS
from app.ai.services.template_generation import generate_dev_template, generate_plan_template
from app.api.schemas.template import NotionTemplatePayload

router = APIRouter()


class TemplateGenerateRequest(BaseModel):
    roomId: int
    templateType: Literal["plan", "dev"]
    currentStatus: str = "READY"
    content: str = ""
    collectedData: Dict[str, str] = Field(default_factory=dict)
    recentMessages: List[str] = Field(default_factory=list)
    selectedMessage: Optional[str] = None


class TemplateGenerateResponse(BaseModel):
    content: str
    currentStatus: str
    notionTemplatePayload: NotionTemplatePayload


def _get_missing_collected_fields(collected_data: Dict[str, str]) -> List[Dict[str, str]]:
    missing_fields: List[Dict[str, str]] = []
    for key, label in COLLECTED_DATA_FIELDS.items():
        value = collected_data.get(key)
        if not isinstance(value, str) or not value.strip():
            missing_fields.append({"key": key, "label": label})
    return missing_fields


def _validate_template_request(request: TemplateGenerateRequest) -> None:
    if request.currentStatus != "READY":
        raise HTTPException(
            status_code=400,
            detail={
                "message": "템플릿 생성은 READY 상태에서만 가능합니다.",
                "currentStatus": request.currentStatus,
            },
        )

    missing_fields = _get_missing_collected_fields(request.collectedData)
    if missing_fields:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "필수 collectedData가 모두 채워져야 템플릿을 생성할 수 있습니다.",
                "missingFields": missing_fields,
            },
        )


def _build_template_state(request: TemplateGenerateRequest) -> dict:
    user_message = request.content.strip()
    if not user_message:
        user_message = (
            "개발용 템플릿 생성해줘"
            if request.templateType == "dev"
            else "기획용 템플릿 생성해줘"
        )

    return {
        "project_id": str(request.roomId),
        "user_message": user_message,
        "action_type": "BTN_DEV" if request.templateType == "dev" else "BTN_PLAN",
        "current_phase": request.currentStatus,
        "collected_data": request.collectedData,
        "recent_messages": request.recentMessages,
        "selected_message": request.selectedMessage,
        "is_sufficient": True,
        "ai_message": "",
        "next_phase": request.currentStatus,
        "template_payload": None,
    }


@router.post("/", response_model=TemplateGenerateResponse)
async def generate_template(request: TemplateGenerateRequest):
    try:
        _validate_template_request(request)
        state = _build_template_state(request)
        result = (
            generate_dev_template(state)
            if request.templateType == "dev"
            else generate_plan_template(state)
        )

        payload = result.get("template_payload")
        if payload is None:
            raise RuntimeError("template generation returned no payload")

        return TemplateGenerateResponse(
            content=result.get("ai_message", ""),
            currentStatus=result.get("next_phase", request.currentStatus),
            notionTemplatePayload=payload,
        )
    except HTTPException:
        raise
    except Exception as exc:
        print(f"Template generation error: {exc}")
        raise HTTPException(status_code=500, detail="템플릿 생성 중 오류가 발생했습니다.")
