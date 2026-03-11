# app/ai/graph/state.py
from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict):
    project_id: str
    user_message: str
    action_type: str  # Spring에서 넘겨주는 이벤트: "CHAT", "BTN_YES", "BTN_NO", "BTN_PLAN", "BTN_DEV"
    current_phase: str  # Spring에서 관리하는 현재 단계 (예: INIT, EXPLORE, TOPIC_SET)
    recent_messages: List[str]
    selected_message: Optional[str]

    # AI가 수집하는 데이터 보관소
    collected_data: Dict[
        str, str
    ]  # exact keys are prompt-configurable
    is_sufficient: bool  # 데이터가 모두 모였는지 여부

    # 결과물
    ai_message: str
    next_phase: str
    template_payload: Dict[str, Any] | None
