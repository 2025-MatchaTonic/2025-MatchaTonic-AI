from typing import TypedDict, Optional, Dict, Any


class AgentState(TypedDict):
    project_id: str
    current_step: int
    user_message: str

    # AI가 채워넣을 결과물
    ai_message: str
    next_step: int
    extracted_data: Optional[Dict[str, Any]]
