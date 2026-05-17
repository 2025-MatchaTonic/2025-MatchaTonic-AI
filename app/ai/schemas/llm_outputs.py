from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class ConversationFieldUpdate(BaseModel):
    field: Literal[
        "subject",
        "title",
        "problemArea",
        "targetFacility",
        "goal",
        "targetUser",
        "teamSize",
        "roles",
        "dueDate",
        "deliverables",
    ]
    value: Any = ""
    raw_evidence: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    is_user_provided_fact: bool = False


class ConversationLLMDecision(BaseModel):
    intent: Literal[
        "provide_info",
        "ask_help",
        "ask_summary",
        "request_next_step",
        "correct_info",
        "meta",
        "other",
    ] = "other"
    response_mode: Literal["answer", "answer_then_ask", "ask_only"] = "answer_then_ask"
    ai_message: str = ""
    updates: List[ConversationFieldUpdate] = Field(default_factory=list)
    next_field: str | None = ""
    needs_clarification: bool = False


class TemplateProblemDefinitionItem(BaseModel):
    id: int = 1
    situation: str = ""
    reason: str = ""
    limitation: str = ""


class TemplateProblemSolutionItem(BaseModel):
    problem_id: int = 1
    solution_desc: str = ""


class TemplateSolution(BaseModel):
    core_summary: str = ""
    problem_solutions: List[TemplateProblemSolutionItem] = Field(default_factory=list)
    features: List[str] = Field(default_factory=list)


class TemplateTargetPersona(BaseModel):
    name: str = ""
    age: str = ""
    job_role: str = ""
    main_activities: str = ""
    pain_points: List[str] = Field(default_factory=list)
    needs: List[str] = Field(default_factory=list)


class TemplatePlanning(BaseModel):
    project_intro: str = ""
    problem_definition: List[TemplateProblemDefinitionItem] = Field(default_factory=list)
    solution: TemplateSolution = Field(default_factory=TemplateSolution)
    target_persona: TemplateTargetPersona = Field(default_factory=TemplateTargetPersona)


class TemplateContentLLMResponse(BaseModel):
    summary_message: str = ""
    project_home: Dict[str, str] = Field(default_factory=dict)
    planning: TemplatePlanning = Field(default_factory=TemplatePlanning)
    ground_rules: str = ""

    def to_merged_dict(self) -> dict:
        return {
            "summary_message": self.summary_message,
            "project_home": self.project_home,
            "planning": self.planning.model_dump(),
            "ground_rules": self.ground_rules,
        }
