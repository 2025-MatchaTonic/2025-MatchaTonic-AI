from copy import deepcopy
from typing import Any

from app.ai.graph.collected_data import (
    build_approved_collected_data_snapshot,
    format_collected_value,
)
from app.ai.graph.state import AgentState
from app.ai.graph.text_support import clean_text, strip_mates_mention

UNKNOWN = "추가 논의 필요"


def build_project_snapshot(data: dict) -> dict[str, str]:
    data = build_approved_collected_data_snapshot(data)
    subject = clean_text(data.get("subject")) or "프로젝트 주제 미정"
    title = clean_text(data.get("title")) or subject
    goal = clean_text(data.get("goal")) or UNKNOWN
    team_size = format_collected_value("teamSize", data.get("teamSize")) or UNKNOWN
    roles = format_collected_value("roles", data.get("roles")) or UNKNOWN
    due_date = clean_text(data.get("dueDate")) or UNKNOWN
    deliverables = clean_text(data.get("deliverables")) or UNKNOWN
    target_user = clean_text(data.get("targetUser")) or UNKNOWN
    problem_area = clean_text(data.get("problemArea")) or UNKNOWN
    target_facility = clean_text(data.get("targetFacility")) or UNKNOWN

    return {
        "subject": subject,
        "title": title,
        "goal": goal,
        "team_size": team_size,
        "roles": roles,
        "due_date": due_date,
        "deliverables": deliverables,
        "target_user": target_user,
        "problem_area": problem_area,
        "target_facility": target_facility,
    }


def _join_lines(values: list[str]) -> str:
    return "\n".join(value for value in values if value)


def _default_planning(snapshot: dict[str, str], mode: str) -> dict:
    title = snapshot["title"]
    goal = snapshot["goal"]
    roles = snapshot["roles"]
    team_size = snapshot["team_size"]
    due_date = snapshot["due_date"]
    deliverables = snapshot["deliverables"]
    target_user = snapshot["target_user"]
    problem_area = snapshot["problem_area"]

    mode_label = "개발" if mode == "dev" else "기획"
    return {
        "project_intro": (
            f"{title} 프로젝트의 {mode_label} 초안입니다. 목표는 {goal}이며, "
            f"대상 사용자는 {target_user}, 주요 산출물은 {deliverables}입니다."
        ),
        "problem_definition": [
            {
                "id": 1,
                "situation": (
                    f"{problem_area} 영역에서 {target_user}가 겪는 문제를 중심으로 "
                    "프로젝트 범위와 우선순위를 정리해야 합니다."
                ),
                "reason": (
                    f"현재 팀 규모는 {team_size}, 역할은 {roles}, 마감 기준은 {due_date}입니다. "
                    "이 제약 안에서 먼저 해결할 문제를 좁혀야 합니다."
                ),
                "limitation": (
                    "수집되지 않은 정보는 확정하지 않고 추가 논의가 필요한 항목으로 남깁니다."
                ),
            }
        ],
        "solution": {
            "core_summary": (
                f"{goal}을 기준으로 문제, 기능, 역할, 일정, 산출물을 하나의 실행 문서로 정리합니다."
            ),
            "problem_solutions": [
                {
                    "problem_id": 1,
                    "solution_desc": (
                        f"{deliverables}를 기준 산출물로 두고, {roles} 역할이 바로 실행할 수 있도록 "
                        "기획 범위와 개발 범위를 분리합니다."
                    ),
                }
            ],
            "features": [
                f"목표 정리: {goal}",
                f"대상 사용자/문제 정리: {target_user} / {problem_area}",
                f"산출물과 마감 기준 정리: {deliverables} / {due_date}",
            ],
        },
        "target_persona": {
            "name": UNKNOWN,
            "age": UNKNOWN,
            "job_role": target_user,
            "main_activities": f"{problem_area}와 관련된 과업을 수행합니다.",
            "pain_points": [
                "문제와 해결 범위가 문서로 정리되지 않으면 우선순위 조정이 어렵습니다.",
                "역할과 산출물 기준이 불명확하면 작업 인수인계가 반복됩니다.",
            ],
            "needs": [
                "팀이 같은 기준으로 볼 수 있는 프로젝트 홈이 필요합니다.",
                "기획, 개발, 운영 항목이 하위페이지로 나뉜 실행 문서가 필요합니다.",
            ],
        },
    }


def build_default_template_sections(state: AgentState, mode: str = "plan") -> dict:
    snapshot = build_project_snapshot(state.get("collected_data") or {})
    title = snapshot["title"]
    goal = snapshot["goal"]
    roles = snapshot["roles"]
    due_date = snapshot["due_date"]
    deliverables = snapshot["deliverables"]
    target_user = snapshot["target_user"]
    target_facility = snapshot["target_facility"]

    mode_label = "개발" if mode == "dev" else "기획"
    return {
        "project_home": {
            "프로젝트 한 줄 요약": (
                f"{title}는 {goal}을 목표로 하는 {mode_label} 프로젝트입니다."
            ),
            "현재 확정된 정보": _join_lines(
                [
                    f"대상 사용자: {target_user}",
                    f"대상 공간/환경: {target_facility}",
                    f"팀/역할: {roles}",
                    f"주요 산출물: {deliverables}",
                    f"마감 기준: {due_date}",
                ]
            ),
            "문서 사용 방법": (
                "기획, 개발, 운영 하위페이지를 기준으로 회의 내용과 결정 사항을 계속 업데이트합니다."
            ),
        },
        "planning": _default_planning(snapshot, mode),
        "schedule": {
            "마일스톤 초안": [
                "1단계: 문제와 대상 사용자 정의",
                "2단계: 핵심 기능과 산출물 범위 확정",
                "3단계: 개발 범위, DB/API 초안 정리",
                "4단계: 결과물 검토 및 발표/제출 준비",
            ],
            "일정 기준": f"현재 마감 기준은 {due_date}입니다.",
            "주의 사항": "정확한 날짜가 없는 경우 임의 날짜를 만들지 않고 추가 논의 필요로 남깁니다.",
        },
        "development": {
            "개발 목표": (
                f"{deliverables}를 구현하기 위해 필요한 화면, API, 데이터 구조를 우선 정리합니다."
            ),
            "기능 후보": [
                "사용자 흐름을 기준으로 핵심 화면을 정의합니다.",
                "기획 범위에서 바로 필요한 API를 먼저 식별합니다.",
                "데이터 저장이 필요한 항목을 DB 설계 페이지와 연결합니다.",
            ],
            "기술 결정 필요 항목": [
                "프론트엔드/백엔드 기술 스택",
                "인증 방식",
                "배포 환경",
                "외부 API 또는 AI 연동 범위",
            ],
        },
        "database": {
            "주요 엔티티 후보": [
                "User: 사용자 또는 팀원 정보",
                "Project: 프로젝트 기본 정보",
                "Task: 역할별 작업과 진행 상태",
                "Document: 산출물과 회의 기록",
            ],
            "필드 초안": [
                "id, title, description, status, owner, dueDate, createdAt, updatedAt",
                "필요한 필드는 실제 기능 범위가 확정된 뒤 조정합니다.",
            ],
            "관계 초안": [
                "Project는 여러 Task와 Document를 가질 수 있습니다.",
                "User는 여러 Project에 참여하고 Task를 담당할 수 있습니다.",
            ],
        },
        "role_guide": {
            "공통 원칙": [
                "모든 역할은 결정 사항을 문서에 남깁니다.",
                "불확실한 항목은 확정 표현 대신 추가 논의 필요로 표시합니다.",
                "산출물 기준과 마감 기준을 먼저 맞춘 뒤 세부 작업을 나눕니다.",
            ],
            "역할별 초안": [
                f"PM/기획: 목표, 사용자, 문제, 일정 기준을 관리합니다. 현재 역할 정보: {roles}",
                "Frontend: 핵심 화면과 사용자 흐름을 정리합니다.",
                "Backend: API, DB, 인증, 배포 범위를 정리합니다.",
                "AI/Data: AI 연동, 데이터 수집, 모델 입출력 기준을 정리합니다.",
            ],
        },
        "meeting_notes": {
            "회의록 템플릿": [
                "회의 목적:",
                "참석자:",
                "논의 내용:",
                "결정 사항:",
                "다음 액션:",
            ],
            "첫 회의 안건": [
                "프로젝트 목표와 산출물 범위 확정",
                "역할별 책임 범위 확정",
                "마감 기준과 중간 점검 일정 확정",
            ],
        },
        "ground_rules": (
            "1. 수집된 사실만 확정 정보로 기록합니다.\n"
            "2. 모르는 값은 추가 논의 필요로 남깁니다.\n"
            "3. 회의에서 결정된 내용은 관련 하위페이지에 즉시 반영합니다.\n"
            "4. 역할, 일정, 산출물 변경은 Project Home에도 요약합니다."
        ),
    }


def get_template_mode_config(action_type: str) -> dict[str, str]:
    if action_type == "BTN_DEV":
        return {
            "mode": "dev",
            "mode_label": "개발형",
            "focus": (
                "개발 범위, 기능 우선순위, API/DB 초안, 역할별 구현 책임이 드러나도록 작성하세요."
            ),
            "summary_fallback": "개발형 Notion 템플릿 초안을 만들었습니다.",
        }

    return {
        "mode": "plan",
        "mode_label": "기획형",
        "focus": (
            "사용자 문제, 기획 의도, 핵심 기능, 팀 운영 기준이 드러나도록 작성하세요."
        ),
        "summary_fallback": "기획형 Notion 템플릿 초안을 만들었습니다.",
    }


def _clean_string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _clean_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [_clean_string(value) for value in values if _clean_string(value)]


def _merge_loose_section(base_section: dict, generated_section: Any) -> dict:
    if not isinstance(generated_section, dict):
        return base_section

    merged = deepcopy(base_section)
    for key, value in generated_section.items():
        if isinstance(value, list):
            cleaned = _clean_string_list(value)
            if cleaned:
                merged[key] = cleaned
        elif isinstance(value, dict):
            nested = _merge_loose_section(
                merged.get(key, {}) if isinstance(merged.get(key), dict) else {},
                value,
            )
            if nested:
                merged[key] = nested
        else:
            cleaned = _clean_string(value)
            if cleaned:
                merged[key] = cleaned
    return merged


def _merge_project_home(base: dict, generated: dict) -> dict:
    if not isinstance(generated, dict):
        return base
    merged = deepcopy(base)
    for key, value in generated.items():
        cleaned = _clean_string(value)
        if cleaned:
            merged[key] = cleaned
    return merged


def _merge_planning(base: dict, generated: dict) -> dict:
    if not isinstance(generated, dict):
        return base

    merged = deepcopy(base)
    project_intro = _clean_string(generated.get("project_intro"))
    if project_intro:
        merged["project_intro"] = project_intro

    problem_definition = generated.get("problem_definition")
    if isinstance(problem_definition, list) and problem_definition:
        cleaned_problem_definition = []
        for index, item in enumerate(problem_definition, start=1):
            if not isinstance(item, dict):
                continue
            cleaned_problem_definition.append(
                {
                    "id": item.get("id") or index,
                    "situation": _clean_string(item.get("situation")),
                    "reason": _clean_string(item.get("reason")),
                    "limitation": _clean_string(item.get("limitation")),
                }
            )
        if cleaned_problem_definition:
            merged["problem_definition"] = cleaned_problem_definition

    solution = generated.get("solution")
    if isinstance(solution, dict):
        merged_solution = deepcopy(base["solution"])
        core_summary = _clean_string(solution.get("core_summary"))
        if core_summary:
            merged_solution["core_summary"] = core_summary

        problem_solutions = solution.get("problem_solutions")
        if isinstance(problem_solutions, list) and problem_solutions:
            cleaned_problem_solutions = []
            for item in problem_solutions:
                if not isinstance(item, dict):
                    continue
                solution_desc = _clean_string(item.get("solution_desc"))
                if solution_desc:
                    cleaned_problem_solutions.append(
                        {
                            "problem_id": item.get("problem_id") or 1,
                            "solution_desc": solution_desc,
                        }
                    )
            if cleaned_problem_solutions:
                merged_solution["problem_solutions"] = cleaned_problem_solutions

        features = _clean_string_list(solution.get("features"))
        if features:
            merged_solution["features"] = features

        merged["solution"] = merged_solution

    target_persona = generated.get("target_persona")
    if isinstance(target_persona, dict):
        merged_persona = deepcopy(base["target_persona"])
        for key in ["name", "age", "job_role", "main_activities"]:
            value = _clean_string(target_persona.get(key))
            if value:
                merged_persona[key] = value
        for key in ["pain_points", "needs"]:
            values = _clean_string_list(target_persona.get(key))
            if values:
                merged_persona[key] = values
        merged["target_persona"] = merged_persona

    return merged


def merge_template_sections(base: dict, generated: dict) -> dict:
    merged = deepcopy(base)

    merged["project_home"] = _merge_project_home(
        base["project_home"],
        generated.get("project_home", {}),
    )
    merged["planning"] = _merge_planning(
        base["planning"],
        generated.get("planning", {}),
    )

    for key in ["schedule", "development", "database", "role_guide", "meeting_notes"]:
        merged[key] = _merge_loose_section(base[key], generated.get(key, {}))

    ground_rules = _clean_string(generated.get("ground_rules"))
    if ground_rules:
        merged["ground_rules"] = ground_rules

    return merged


def _planning_problem_page(planning: dict) -> dict:
    items = []
    for item in planning.get("problem_definition", []):
        if not isinstance(item, dict):
            continue
        item_id = item.get("id") or len(items) + 1
        items.extend(
            [
                f"문제 {item_id} - 상황: {item.get('situation', '')}",
                f"문제 {item_id} - 원인: {item.get('reason', '')}",
                f"문제 {item_id} - 한계: {item.get('limitation', '')}",
            ]
        )
    return {"문제 정의": items or [UNKNOWN]}


def _planning_solution_page(planning: dict) -> dict:
    solution = planning.get("solution", {})
    return {
        "해결 방향": solution.get("core_summary", UNKNOWN),
        "문제별 해결안": [
            item.get("solution_desc", "")
            for item in solution.get("problem_solutions", [])
            if isinstance(item, dict) and item.get("solution_desc")
        ]
        or [UNKNOWN],
        "핵심 기능": solution.get("features", []) or [UNKNOWN],
    }


def _planning_persona_page(planning: dict) -> dict:
    persona = planning.get("target_persona", {})
    return {
        "기본 정보": [
            f"이름: {persona.get('name', UNKNOWN)}",
            f"나이: {persona.get('age', UNKNOWN)}",
            f"직업/역할: {persona.get('job_role', UNKNOWN)}",
            f"주요 활동: {persona.get('main_activities', UNKNOWN)}",
        ],
        "불편함": persona.get("pain_points", []) or [UNKNOWN],
        "필요": persona.get("needs", []) or [UNKNOWN],
    }


def _development_api_page(development: dict) -> dict:
    return {
        "API 후보": [
            "GET /projects - 프로젝트 목록 조회",
            "GET /projects/{id} - 프로젝트 상세 조회",
            "POST /projects/{id}/tasks - 작업 생성",
            "PATCH /tasks/{id} - 작업 상태 변경",
        ],
        "확정 필요": development.get("기술 결정 필요 항목", []) or [UNKNOWN],
    }


def build_notion_template_payload(state: AgentState, sections: dict) -> dict:
    snapshot = build_project_snapshot(state.get("collected_data") or {})
    home_title = snapshot["title"] if snapshot["title"] else "Project Home"

    planning = sections["planning"]
    development = sections["development"]

    return {
        "projectId": int(state["project_id"]),
        "templates": [
            {
                "key": "PROJECT_HOME",
                "parentKey": None,
                "title": f"{home_title} Project Home",
                "content": sections["project_home"],
            },
            {
                "key": "PLANNING",
                "parentKey": "PROJECT_HOME",
                "title": "01. 기획",
                "content": {"기획 요약": planning.get("project_intro", UNKNOWN)},
            },
            {
                "key": "PROBLEM_DEFINITION",
                "parentKey": "PLANNING",
                "title": "문제 정의",
                "content": _planning_problem_page(planning),
            },
            {
                "key": "SOLUTION_FEATURES",
                "parentKey": "PLANNING",
                "title": "해결 방향과 핵심 기능",
                "content": _planning_solution_page(planning),
            },
            {
                "key": "TARGET_PERSONA",
                "parentKey": "PLANNING",
                "title": "타겟 사용자",
                "content": _planning_persona_page(planning),
            },
            {
                "key": "DEVELOPMENT",
                "parentKey": "PROJECT_HOME",
                "title": "02. 개발",
                "content": development,
            },
            {
                "key": "API_PLAN",
                "parentKey": "DEVELOPMENT",
                "title": "API 초안",
                "content": _development_api_page(development),
            },
            {
                "key": "DB",
                "parentKey": "DEVELOPMENT",
                "title": "DB 설계",
                "content": sections["database"],
            },
            {
                "key": "OPERATIONS",
                "parentKey": "PROJECT_HOME",
                "title": "03. 운영",
                "content": {
                    "운영 기준": [
                        "일정, 역할, 회의록, 그라운드룰을 이 하위페이지 아래에서 관리합니다.",
                        "변경된 결정 사항은 관련 페이지와 Project Home에 함께 반영합니다.",
                    ]
                },
            },
            {
                "key": "SCHEDULE",
                "parentKey": "OPERATIONS",
                "title": "일정",
                "content": sections["schedule"],
            },
            {
                "key": "ROLE_GUIDE",
                "parentKey": "OPERATIONS",
                "title": "역할별 가이드",
                "content": sections["role_guide"],
            },
            {
                "key": "MEETING_NOTES",
                "parentKey": "OPERATIONS",
                "title": "회의록",
                "content": sections["meeting_notes"],
            },
            {
                "key": "GROUND_RULES",
                "parentKey": "OPERATIONS",
                "title": "그라운드룰",
                "content": sections["ground_rules"],
            },
        ],
    }


def build_recent_context(state: AgentState) -> str:
    selected_message = strip_mates_mention(state.get("selected_message"))
    recent_messages = [
        strip_mates_mention(msg)
        for msg in state.get("recent_messages", [])
        if strip_mates_mention(msg)
    ]

    context_blocks: list[str] = []
    if selected_message:
        context_blocks.append(f"[현재 집중 포인트]\n- {selected_message}")
    if recent_messages:
        latest_messages = recent_messages[-3:]
        flow_summary = "\n".join(f"- {message}" for message in latest_messages)
        context_blocks.append("[최근 대화 요약]\n" + flow_summary)

    return "\n\n".join(context_blocks) if context_blocks else "(전달된 최근 대화 없음)"


def build_template_input_summary(state: AgentState) -> str:
    snapshot = build_project_snapshot(state.get("collected_data") or {})
    return (
        f"- 주제: {snapshot['subject']}\n"
        f"- 제목: {snapshot['title']}\n"
        f"- 목표: {snapshot['goal']}\n"
        f"- 대상 사용자: {snapshot['target_user']}\n"
        f"- 팀 규모: {snapshot['team_size']}\n"
        f"- 역할: {snapshot['roles']}\n"
        f"- 마감일: {snapshot['due_date']}\n"
        f"- 산출물: {snapshot['deliverables']}"
    )


def build_template_content_example() -> dict:
    return {
        "project_home": {
            "프로젝트 한 줄 요약": "프로젝트의 목표와 대상 사용자를 한 문장으로 정리",
            "현재 확정된 정보": "수집된 사실만 요약",
            "문서 사용 방법": "팀이 이 Notion을 어떻게 업데이트할지 설명",
        },
        "planning": {
            "project_intro": "프로젝트 배경과 기획 의도",
            "problem_definition": [
                {
                    "id": 1,
                    "situation": "사용자가 겪는 상황",
                    "reason": "그 문제가 중요한 이유",
                    "limitation": "현재 방식의 한계",
                }
            ],
            "solution": {
                "core_summary": "해결 방향 요약",
                "problem_solutions": [
                    {
                        "problem_id": 1,
                        "solution_desc": "문제에 대응하는 해결 방식",
                    }
                ],
                "features": ["핵심 기능 1", "핵심 기능 2", "핵심 기능 3"],
            },
            "target_persona": {
                "name": "추가 논의 필요",
                "age": "추가 논의 필요",
                "job_role": "사용자 또는 역할",
                "main_activities": "주요 활동",
                "pain_points": ["불편함 1", "불편함 2"],
                "needs": ["필요 1", "필요 2"],
            },
        },
        "schedule": {
            "마일스톤 초안": ["프로젝트 단계별 일정"],
            "일정 기준": "마감 기준",
            "주의 사항": "날짜가 없으면 추가 논의 필요로 표시",
        },
        "development": {
            "개발 목표": "구현해야 하는 핵심 목표",
            "기능 후보": ["기능 1", "기능 2", "기능 3"],
            "기술 결정 필요 항목": ["스택", "인증", "배포"],
        },
        "database": {
            "주요 엔티티 후보": ["엔티티와 설명"],
            "필드 초안": ["필드 후보"],
            "관계 초안": ["관계 후보"],
        },
        "role_guide": {
            "공통 원칙": ["팀 운영 원칙"],
            "역할별 초안": ["역할별 책임과 다음 액션"],
        },
        "meeting_notes": {
            "회의록 템플릿": ["회의 목적", "결정 사항", "다음 액션"],
            "첫 회의 안건": ["처음 논의할 항목"],
        },
        "ground_rules": "팀 운영 규칙을 번호 목록으로 작성",
    }
