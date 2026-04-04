from typing import Any, Dict, List, Literal, Optional, TypedDict

from app.ai.graph.collected_data import CollectedData


TurnPolicy = Literal[
    "ASK_ONLY",
    "CAPTURE_TITLE",
    "ANSWER_ONLY",
    "ANSWER_THEN_ASK",
]


class AgentState(TypedDict):
    project_id: str
    user_message: str
    action_type: str
    current_phase: str
    turn_policy: TurnPolicy
    recent_messages: List[str]
    selected_message: Optional[str]
    collected_data: CollectedData
    is_sufficient: bool
    ai_message: str
    next_phase: str
    template_payload: Dict[str, Any] | None
