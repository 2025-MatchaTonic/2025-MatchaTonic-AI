# app/api/endpoints/chat.py
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ai.graph.workflow import ai_app

router = APIRouter()


# ---------------------------------------------------------
# 1. Spring -> FastAPI 로 보내는 요청 (Internal Request)
# ---------------------------------------------------------
class AIChatRequest(BaseModel):
    roomId: int  # Spring 명세의 roomId 반영
    content: str  # 사용자 채팅 내용 (명세서의 content 반영)
    actionType: str  # "CHAT", "BTN_YES", "BTN_NO", "BTN_PLAN", "BTN_DEV" 등
    currentStatus: str  # 명세서의 현재 상태값
    collectedData: Dict[str, str] = Field(
        default_factory=dict
    )  # Spring DB에 저장된 이전 데이터
    recentMessages: List[str] = Field(
        default_factory=list
    )  # @mates 호출 시 최근 팀 대화 목록
    selectedMessage: Optional[str] = None  # 사용자가 핵심 채팅으로 선택한 메시지


class NotionTemplateItem(BaseModel):
    key: str
    parentKey: Optional[str]
    title: str
    content: Any


class NotionTemplatePayload(BaseModel):
    projectId: int
    templates: List[NotionTemplateItem]


# ---------------------------------------------------------
# 2. FastAPI -> Spring 으로 돌려주는 응답 (Internal Response)
# ---------------------------------------------------------
class AIChatResponse(BaseModel):
    content: str  # AI 응답 텍스트 (Spring 명세의 aiResponse.content 에 맵핑)
    suggestedQuestions: List[
        str
    ]  # 프론트엔드 버튼/추천질문용 (Spring 명세의 suggestedQuestions 에 맵핑)
    currentStatus: str  # 변경된 진행 단계 (Spring 명세의 currentStatus 에 맵핑)
    isSufficient: bool  # 템플릿 생성 가능 여부
    collectedData: Dict[str, str]  # 업데이트된 기획 데이터 (Spring이 DB에 저장해야 함)
    notionTemplatePayload: Optional[NotionTemplatePayload] = None


@router.post("/", response_model=AIChatResponse)
async def process_chat(request: AIChatRequest):
    try:
        # LangGraph 상태 초기화 (snake_case로 변환하여 내부 로직에 주입)
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

        # LangGraph 실행
        result = ai_app.invoke(initial_state)

        # 결과를 Spring 명세에 맞게 camelCase로 포장하여 반환
        return AIChatResponse(
            content=result.get("ai_message", ""),
            suggestedQuestions=[],  # 필요시 LangGraph 노드에서 추천 질문을 배열로 뽑아서 여기에 넣으면 됩니다.
            currentStatus=result.get("next_phase", request.currentStatus),
            isSufficient=result.get("is_sufficient", False),
            collectedData=result.get("collected_data", {}),
            notionTemplatePayload=result.get("template_payload"),
        )

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="AI 처리 중 오류 발생")
