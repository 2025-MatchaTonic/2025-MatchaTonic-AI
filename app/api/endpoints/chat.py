from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from app.ai.graph.workflow import ai_app  # 방금 만든 LangGraph 앱 불러오기

router = APIRouter()


class ChatRequest(BaseModel):
    project_id: str
    message: str
    current_step: int


class ChatResponse(BaseModel):
    ai_message: str
    next_step: int
    extracted_data: Optional[Dict[str, Any]] = None


@router.post("/", response_model=ChatResponse)
async def process_chat(request: ChatRequest):
    try:
        # LangGraph에 주입할 초기 상태
        initial_state = {
            "project_id": request.project_id,
            "current_step": request.current_step,
            "user_message": request.message,
            "ai_message": "",
            "next_step": request.current_step,
            "extracted_data": None,
        }

        # LangGraph 실행!
        result = ai_app.invoke(initial_state)

        return ChatResponse(
            ai_message=result["ai_message"],
            next_step=result["next_step"],
            extracted_data=result.get("extracted_data"),
        )

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="AI 처리 중 오류 발생")
