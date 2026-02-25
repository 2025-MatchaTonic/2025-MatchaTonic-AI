from typing import Dict, List

from fastapi import APIRouter
from pydantic import BaseModel

from app.ai.prompts.pm import STEP_NAMES

router = APIRouter()


class StepInfo(BaseModel):
    step: int
    key: str
    title: str
    done_criteria: List[str]


@router.get("/steps", response_model=List[StepInfo])
async def get_steps():
    return [
        StepInfo(
            step=1,
            key=STEP_NAMES[1],
            title="Problem Definition",
            done_criteria=[
                "target user identified",
                "core problem sentence validated",
                "success metric draft exists",
            ],
        ),
        StepInfo(
            step=2,
            key=STEP_NAMES[2],
            title="Scope Alignment",
            done_criteria=[
                "MVP in-scope agreed",
                "out-of-scope listed",
                "constraints captured",
            ],
        ),
        StepInfo(
            step=3,
            key=STEP_NAMES[3],
            title="Role Assignment",
            done_criteria=[
                "owner per role assigned",
                "workload is balanced",
                "handoff rule is clear",
            ],
        ),
        StepInfo(
            step=4,
            key=STEP_NAMES[4],
            title="Execution Planning",
            done_criteria=[
                "first sprint tasks listed",
                "priority and dependencies clear",
                "review checkpoint scheduled",
            ],
        ),
    ]


@router.get("/manual", response_model=Dict[str, List[str]])
async def get_manual():
    return {
        "team_common_rules": [
            "Every decision must reference user problem or metric.",
            "Limit one topic per discussion turn.",
            "Track unresolved items explicitly.",
        ],
        "pm_thinking_checklist": [
            "Why now and for whom?",
            "What is the minimum testable value?",
            "What is the riskiest assumption this week?",
        ],
    }


@router.get("/templates", response_model=Dict[str, Dict[str, str]])
async def get_templates():
    return {
        "problem_statement": {
            "template": "For [target user], [current pain] causes [negative impact].",
            "example": "For first-time project teams, unclear scope causes late delivery.",
        },
        "mvp_scope": {
            "template": "In-scope: [3 bullets], Out-of-scope: [3 bullets].",
            "example": "In-scope: login, task board, progress update.",
        },
        "role_card": {
            "template": "[Role] owns [deliverable] and reviews [handoff artifact].",
            "example": "PM owns sprint planning and reviews weekly status report.",
        },
    }
