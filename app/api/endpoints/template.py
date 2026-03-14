from typing import Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ai.graph.nodes import (
    generate_dev_template_node,
    generate_plan_template_node,
)

router = APIRouter()


class NotionTemplateItem(BaseModel):
    key: str
    parentKey: Optional[str]
    title: str
    content: object


class NotionTemplatePayload(BaseModel):
    projectId: int
    templates: List[NotionTemplateItem]


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
        state = _build_template_state(request)
        result = (
            generate_dev_template_node(state)
            if request.templateType == "dev"
            else generate_plan_template_node(state)
        )

        payload = result.get("template_payload")
        if payload is None:
            raise RuntimeError("template generation returned no payload")

        return TemplateGenerateResponse(
            content=result.get("ai_message", ""),
            currentStatus=result.get("next_phase", request.currentStatus),
            notionTemplatePayload=payload,
        )
    except Exception as exc:
        print(f"Template generation error: {exc}")
        raise HTTPException(status_code=500, detail="템플릿 생성 중 오류가 발생했습니다.")
