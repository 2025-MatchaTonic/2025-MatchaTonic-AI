from typing import Dict, List

from pydantic import BaseModel, Field

from app.ai.graph.collected_data import COLLECTED_DATA_FIELDS


class GatherLLMResponse(BaseModel):
    ai_message: str = ""
    updated_data: Dict[str, str] = Field(default_factory=dict)
    is_sufficient: bool = False

    def normalized_updated_data(self) -> Dict[str, str]:
        normalized: Dict[str, str] = {}
        for key in COLLECTED_DATA_FIELDS:
            value = self.updated_data.get(key)
            if isinstance(value, str) and value.strip():
                normalized[key] = value.strip()
        return normalized


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
    ground_rules: str = " "

    def to_merged_dict(self) -> dict:
        return {
            "summary_message": self.summary_message,
            "project_home": self.project_home,
            "planning": self.planning.model_dump(),
            "ground_rules": self.ground_rules,
        }
