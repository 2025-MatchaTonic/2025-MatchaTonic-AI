from copy import deepcopy
from typing import Any

from app.ai.graph.collected_data import (
    build_approved_collected_data_snapshot,
    format_collected_value,
)
from app.ai.graph.state import AgentState
from app.ai.graph.text_support import clean_text, strip_mates_mention

UNKNOWN = "추가 논의 필요"


def _is_known(value: Any) -> bool:
    text = clean_text(value) if not isinstance(value, str) else value.strip()
    if not text:
        return False
    return text not in {
        UNKNOWN,
        "프로젝트 주제 미정",
        "사용자 또는 역할",
        "대상 사용자 미정",
        "미정",
    }


def _clean_string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _clean_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [_clean_string(value) for value in values if _clean_string(value)]


def _join_lines(values: list[str]) -> str:
    return "\n".join(value for value in values if value)


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


def _confirmed_facts(snapshot: dict[str, str]) -> list[str]:
    labels = [
        ("프로젝트", "title"),
        ("주제", "subject"),
        ("목표", "goal"),
        ("대상 사용자", "target_user"),
        ("문제 영역", "problem_area"),
        ("대상 공간/환경", "target_facility"),
        ("팀/역할", "roles"),
        ("마감 기준", "due_date"),
        ("산출물", "deliverables"),
    ]
    facts = [
        f"{label}: {snapshot[key]}"
        for label, key in labels
        if _is_known(snapshot.get(key))
    ]
    return facts or ["아직 확정된 정보가 충분하지 않습니다."]


def _default_planning(snapshot: dict[str, str], mode: str) -> dict:
    goal = snapshot["goal"]
    deliverables = snapshot["deliverables"]
    target_user = snapshot["target_user"]
    problem_area = snapshot["problem_area"]

    has_problem_context = _is_known(target_user) or _is_known(problem_area)
    has_solution_context = _is_known(goal) or _is_known(deliverables)

    problem_situation = (
        f"{problem_area if _is_known(problem_area) else '정의할 문제 영역'}에서 "
        f"{target_user if _is_known(target_user) else '정의할 대상 사용자'}가 겪는 불편함을 확인해야 합니다."
        if has_problem_context
        else "대상 사용자와 실제 불편함이 아직 명확하지 않습니다."
    )
    problem_reason = (
        "현재 확정된 목표와 산출물을 실행 가능한 범위로 좁히려면 먼저 사용자와 문제 상황을 분리해야 합니다."
        if has_solution_context
        else "프로젝트 목표, 대상 사용자, 핵심 사용 상황이 부족해 문제 우선순위를 판단하기 어렵습니다."
    )

    solution_summary = (
        f"목표({goal if _is_known(goal) else '정의할 목표'})를 기준으로 최소 실행 범위를 정리합니다."
        if has_solution_context
        else "목표와 산출물이 확정되기 전까지 솔루션은 방향성만 정리합니다."
    )
    solution_desc = (
        f"{deliverables if _is_known(deliverables) else '정의할 산출물'} 기준으로 핵심 기능을 먼저 좁히고, "
        "역할별 실행 작업으로 나눕니다."
        if has_solution_context
        else "추가 회의에서 목표, 산출물, 핵심 기능을 확정한 뒤 실행 작업으로 나눕니다."
    )

    return {
        "project_intro": "수집된 채팅 내용을 기준으로 기획 하위 페이지 초안을 구성합니다.",
        "problem_definition": [
            {
                "id": 1,
                "situation": problem_situation,
                "reason": problem_reason,
                "limitation": (
                    "확정되지 않은 대상 사용자, KPI, 세부 기능은 임의로 만들지 않고 추가 확인 항목으로 남깁니다."
                ),
            }
        ],
        "solution": {
            "core_summary": solution_summary,
            "problem_solutions": [
                {
                    "problem_id": 1,
                    "solution_desc": solution_desc,
                }
            ],
            "features": [
                feature
                for feature in [
                    f"목표 기준 정리: {goal}" if _is_known(goal) else "",
                    f"산출물 기준 정리: {deliverables}" if _is_known(deliverables) else "",
                    "대상 사용자와 핵심 사용 흐름은 추가 논의에서 확정",
                ]
                if feature
            ],
        },
        "target_persona": {
            "name": UNKNOWN,
            "age": UNKNOWN,
            "job_role": target_user if _is_known(target_user) else UNKNOWN,
            "main_activities": UNKNOWN,
            "pain_points": [],
            "needs": [],
        },
    }


def build_default_template_sections(state: AgentState, mode: str = "plan") -> dict:
    snapshot = build_project_snapshot(state.get("collected_data") or {})
    mode_label = "개발" if mode == "dev" else "기획"

    return {
        "project_home": {
            "프로젝트 한 줄 요약": (
                f"{snapshot['title']}는 {snapshot['goal']}을 목표로 하는 {mode_label} 프로젝트입니다."
                if _is_known(snapshot["goal"])
                else f"{snapshot['title']} 프로젝트입니다. 목표는 추가 논의가 필요합니다."
            ),
            "현재 확정된 정보": _join_lines(_confirmed_facts(snapshot)),
            "문서 사용 방법": (
                "Project Home은 확정 정보만 요약하고, 기획 하위 페이지는 AI가 작성한 "
                "문제 정의/솔루션/타겟 페르소나 내용을 백엔드가 우선 반영합니다."
            ),
        },
        "planning": _default_planning(snapshot, mode),
        "development": {},
        "database": {},
        "role_guide": {},
        "meeting_notes": {},
        "schedule": {},
        "ground_rules": (
            "1. 수집된 사실만 확정 정보로 기록합니다.\n"
            "2. 모르는 값은 추가 논의 필요로 남깁니다.\n"
            "3. 결정된 내용은 관련 하위페이지에 즉시 반영합니다.\n"
            "4. 역할, 일정, 산출물 변경은 Project Home에도 요약합니다."
        ),
    }


def get_template_mode_config(action_type: str) -> dict[str, str]:
    if action_type == "BTN_DEV":
        return {
            "mode": "dev",
            "mode_label": "개발형",
            "focus": (
                "개발 범위보다 기획 하위 페이지에 들어갈 문제 정의, 솔루션, "
                "타겟 페르소나가 채팅 내용에 맞게 구체화되도록 작성하세요."
            ),
            "summary_fallback": "개발형 Notion 템플릿 초안을 만들었습니다.",
        }

    return {
        "mode": "plan",
        "mode_label": "기획형",
        "focus": (
            "사용자 문제, 기획 의도, 핵심 기능, 타겟 사용자가 드러나도록 작성하세요."
        ),
        "summary_fallback": "기획형 Notion 템플릿 초안을 만들었습니다.",
    }


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

    ground_rules = _clean_string(generated.get("ground_rules"))
    if ground_rules:
        merged["ground_rules"] = ground_rules

    return merged


def _format_problem_content(planning: dict, snapshot: dict[str, str]) -> str:
    lines: list[str] = ["## 현재 확정된 정보"]
    lines.extend(f"- {fact}" for fact in _confirmed_facts(snapshot))

    for item in planning.get("problem_definition", []):
        if not isinstance(item, dict):
            continue
        situation = _clean_string(item.get("situation"))
        reason = _clean_string(item.get("reason"))
        limitation = _clean_string(item.get("limitation"))
        if situation:
            lines.extend(["", "## 상황", situation])
        if reason:
            lines.extend(["", "## 문제 원인", reason])
        if limitation:
            lines.extend(["", "## 기존 방식의 한계", limitation])

    if not (_is_known(snapshot["target_user"]) and _is_known(snapshot["problem_area"])):
        lines.extend(
            [
                "",
                "## 추가 확인 필요",
                "- 이 프로젝트의 주요 사용자는 누구인가?",
                "- 사용자는 어떤 상황에서 불편함을 겪는가?",
                "- 해결해야 할 문제를 한 문장으로 정의하면 무엇인가?",
            ]
        )
    return "\n".join(lines)


def _format_solution_content(planning: dict, snapshot: dict[str, str]) -> str:
    solution = planning.get("solution", {})
    lines: list[str] = ["## 현재 확정된 정보"]
    lines.extend(f"- {fact}" for fact in _confirmed_facts(snapshot))

    core_summary = _clean_string(solution.get("core_summary"))
    if core_summary:
        lines.extend(["", "## 해결 방향", core_summary])

    problem_solutions = [
        _clean_string(item.get("solution_desc"))
        for item in solution.get("problem_solutions", [])
        if isinstance(item, dict) and _clean_string(item.get("solution_desc"))
    ]
    if problem_solutions:
        lines.extend(["", "## 문제별 해결안"])
        lines.extend(f"- {item}" for item in problem_solutions)

    features = _clean_string_list(solution.get("features"))
    if features:
        lines.extend(["", "## 핵심 기능 후보"])
        lines.extend(f"- {feature}" for feature in features)

    if not (_is_known(snapshot["goal"]) and _is_known(snapshot["deliverables"])):
        lines.extend(
            [
                "",
                "## 추가 확인 필요",
                "- 최종 산출물은 무엇인가?",
                "- 반드시 포함해야 하는 최소 기능은 무엇인가?",
                "- 이번 마감 안에 제외할 기능은 무엇인가?",
            ]
        )
    return "\n".join(lines)


def _format_persona_content(planning: dict, snapshot: dict[str, str]) -> str:
    persona = planning.get("target_persona", {})
    lines = ["## 기본 정보"]

    target_user = _clean_string(persona.get("job_role"))
    main_activities = _clean_string(persona.get("main_activities"))
    if not _is_known(target_user):
        target_user = snapshot["target_user"] if _is_known(snapshot["target_user"]) else UNKNOWN
    if not _is_known(main_activities):
        main_activities = UNKNOWN

    lines.extend(
        [
            f"- 이름: {_clean_string(persona.get('name')) or UNKNOWN}",
            f"- 나이: {_clean_string(persona.get('age')) or UNKNOWN}",
            f"- 직업/역할: {target_user}",
            f"- 주요 활동: {main_activities}",
        ]
    )

    has_target_user = _is_known(snapshot["target_user"])
    if has_target_user:
        pain_points = _clean_string_list(persona.get("pain_points"))
        if pain_points:
            lines.extend(["", "## 불편함"])
            lines.extend(f"- {item}" for item in pain_points)

        needs = _clean_string_list(persona.get("needs"))
        if needs:
            lines.extend(["", "## 필요"])
            lines.extend(f"- {item}" for item in needs)
    else:
        lines.extend(
            [
                "",
                "## 추가 확인 필요",
                "- 이 앱/서비스를 가장 자주 사용할 사람은 누구인가?",
                "- 사용자의 역할, 상황, 기대 행동은 무엇인가?",
                "- 사용자가 현재 겪는 불편함은 무엇인가?",
            ]
        )
    return "\n".join(lines)


def build_notion_template_payload(state: AgentState, sections: dict) -> dict:
    snapshot = build_project_snapshot(state.get("collected_data") or {})
    home_title = snapshot["title"] if _is_known(snapshot["title"]) else "Project Home"
    planning = sections["planning"]

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
                "title": "기획",
                "content": None,
            },
            {
                "key": "PROBLEM_DEFINITION",
                "parentKey": "PLANNING",
                "title": "문제 정의",
                "content": _format_problem_content(planning, snapshot),
            },
            {
                "key": "SOLUTION",
                "parentKey": "PLANNING",
                "title": "솔루션",
                "content": _format_solution_content(planning, snapshot),
            },
            {
                "key": "TARGET_PERSONA",
                "parentKey": "PLANNING",
                "title": "타겟 페르소나",
                "content": _format_persona_content(planning, snapshot),
            },
            {
                "key": "DEVELOPMENT",
                "parentKey": "PROJECT_HOME",
                "title": "개발",
                "content": None,
            },
            {
                "key": "DB",
                "parentKey": "PROJECT_HOME",
                "title": "DB",
                "content": None,
            },
            {
                "key": "ROLE_GUIDE",
                "parentKey": "PROJECT_HOME",
                "title": "역할별 가이드",
                "content": None,
            },
            {
                "key": "GROUND_RULES",
                "parentKey": "PROJECT_HOME",
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
                    "situation": "사용자가 겪는 상황. 부족하면 추가 확인 필요라고 작성",
                    "reason": "그 문제가 중요한 이유. 부족하면 판단 근거가 부족하다고 작성",
                    "limitation": "현재 방식의 한계. 부족하면 확정하지 말고 확인 필요라고 작성",
                }
            ],
            "solution": {
                "core_summary": "해결 방향 요약. 부족하면 방향성만 작성",
                "problem_solutions": [
                    {
                        "problem_id": 1,
                        "solution_desc": "문제에 대응하는 해결 방식. 부족하면 추가 논의 필요라고 작성",
                    }
                ],
                "features": ["핵심 기능 후보 1", "핵심 기능 후보 2", "핵심 기능 후보 3"],
            },
            "target_persona": {
                "name": UNKNOWN,
                "age": UNKNOWN,
                "job_role": "사용자 또는 역할. 부족하면 추가 논의 필요",
                "main_activities": "주요 활동. 부족하면 추가 논의 필요",
                "pain_points": [],
                "needs": [],
            },
        },
        "ground_rules": "팀 운영 규칙을 번호 목록으로 작성",
    }
