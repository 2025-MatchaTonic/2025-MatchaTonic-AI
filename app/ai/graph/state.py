# app/ai/graph/state.py
from typing import TypedDict, Optional, Dict, Any


class AgentState(TypedDict):
    project_id: str
    user_message: str
    action_type: str  # Spring에서 넘겨주는 이벤트: "CHAT", "BTN_YES", "BTN_NO", "BTN_GO_DEF", "BTN_MORE", "BTN_PLAN", "BTN_DEV"
    current_phase: str  # "INIT", "EXPLORE", "GATHER", "READY", "DONE"

    # AI가 수집하는 데이터 보관소
    collected_data: Dict[
        str, str
    ]  # {topic, solution, requirements, impact, deliverables}
    is_sufficient: bool  # 데이터가 모두 모였는지 여부

    # 결과물
    ai_message: str
    next_phase: str
