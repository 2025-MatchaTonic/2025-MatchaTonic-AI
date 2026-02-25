# app/api/endpoints/chat.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from app.ai.graph.workflow import ai_app

router = APIRouter()


class ChatRequest(BaseModel):
    project_id: str
    message: str
    action_type: str  # "CHAT", "BTN_YES", "BTN_NO", "BTN_GO_DEF", "BTN_MORE", "BTN_PLAN", "BTN_DEV"
    current_phase: str  # "INIT", "EXPLORE", "GATHER", "READY", "DONE"
    collected_data: Dict[str, str] = (
        {}
    )  # 기존에 모인 데이터 (Spring에서 관리하여 넘겨줌)


class ChatResponse(BaseModel):
    ai_message: str
    next_phase: str
    is_sufficient: bool
    collected_data: Dict[str, str]


@router.post("/", response_model=ChatResponse)
async def process_chat(request: ChatRequest):
    try:
        initial_state = {
            "project_id": request.project_id,
            "user_message": request.message,
            "action_type": request.action_type,
            "current_phase": request.current_phase,
            "collected_data": request.collected_data,
            "is_sufficient": False,
            "ai_message": "",
            "next_phase": request.current_phase,
        }

        result = ai_app.invoke(initial_state)

        return ChatResponse(
            ai_message=result["ai_message"],
            next_phase=result["next_phase"],
            is_sufficient=result.get("is_sufficient", False),
            collected_data=result.get("collected_data", {}),
        )

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="AI 처리 중 오류 발생")
