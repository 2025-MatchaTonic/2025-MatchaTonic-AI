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
    project_name: Optional[str]
    user_message: str
    action_type: str
    current_phase: str
    turn_policy: TurnPolicy
    recent_messages: List[str]
    selected_message: Optional[str]
    collected_data: CollectedData
    problem_area: Optional[str]
    target_facility: Optional[str]
    current_slot: Optional[str]
    next_question_field: Optional[str]
    is_sufficient: bool
    ai_message: str
    next_phase: str
    template_payload: Dict[str, Any] | None
    approved_updates: Dict[str, Any]
    rejected_updates: Dict[str, Any]
    rejected_reasons: Dict[str, Any]
    followup_fields: List[str]
