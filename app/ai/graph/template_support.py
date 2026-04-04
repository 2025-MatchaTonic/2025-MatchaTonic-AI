from app.ai.graph.collected_data import format_collected_value
from app.ai.graph.state import AgentState
from app.ai.graph.text_support import clean_text, strip_mates_mention


def build_project_snapshot(data: dict) -> dict[str, str]:
    subject = clean_text(data.get("subject")) or "프로젝트 주제 미정"
    title = clean_text(data.get("title")) or subject
    goal = clean_text(data.get("goal")) or "프로젝트 목표는 추가 논의가 필요합니다."
    team_size = format_collected_value("teamSize", data.get("teamSize")) or "팀 규모 미정"
    roles = format_collected_value("roles", data.get("roles")) or "역할 분담 추가 논의 필요"
    due_date = clean_text(data.get("dueDate")) or "마감 일정 미정"
    deliverables = clean_text(data.get("deliverables")) or "산출물 범위 추가 논의 필요"

    return {
        "subject": subject,
        "title": title,
        "goal": goal,
        "team_size": team_size,
        "roles": roles,
        "due_date": due_date,
        "deliverables": deliverables,
    }


def build_default_template_sections(state: AgentState, mode: str = "plan") -> dict:
    snapshot = build_project_snapshot(state.get("collected_data") or {})
    title = snapshot["title"]
    goal = snapshot["goal"]
    team_size = snapshot["team_size"]
    roles = snapshot["roles"]
    due_date = snapshot["due_date"]
    deliverables = snapshot["deliverables"]

    if mode == "dev":
        return {
            "project_home": {
                "project_overview": (
                    f"{title} 프로젝트의 개발 실행 초안입니다. 목표는 {goal} "
                    f"현재 기준 팀 규모는 {team_size}, 역할은 {roles}, "
                    f"주요 산출물은 {deliverables}, 마감 기준은 {due_date}입니다."
                ),
            },
            "planning": {
                "project_intro": f"{title} 개발 실행 초안: {goal}",
                "problem_definition": [
                    {
                        "id": 1,
                        "situation": (
                            "개발 범위, 책임 분담, 산출물 연결 방식이 한 문서에 정리되지 않으면 "
                            "구현 우선순위와 핸드오프가 흔들릴 수 있습니다."
                        ),
                        "reason": (
                            f"팀 규모 {team_size}에서 역할({roles})과 마감({due_date})이 있는 만큼 "
                            "실행 기준을 미리 고정해야 개발 속도와 품질을 맞출 수 있습니다."
                        ),
                        "limitation": (
                            "현재 수집된 정보만으로는 세부 기술 스택, API 명세, 데이터 모델, "
                            "세부 태스크 단위까지는 확정할 수 없습니다."
                        ),
                    }
                ],
                "solution": {
                    "core_summary": (
                        "목표, 역할, 산출물, 마감 기준을 바탕으로 개발 범위와 실행 기준을 단일 문서로 정리합니다."
                    ),
                    "problem_solutions": [
                        {
                            "problem_id": 1,
                            "solution_desc": (
                                f"역할은 {roles}, 산출물은 {deliverables}, 마감은 {due_date} 기준으로 "
                                "개발 범위와 책임을 정리하고 구현 전에 공통 기준을 맞춥니다."
                            ),
                        }
                    ],
                    "features": [
                        f"개발 목표 정리: {goal}",
                        f"역할 및 책임 기준: {roles} / {team_size}",
                        f"구현 산출물 및 일정 기준: {deliverables} / {due_date}",
                    ],
                },
                "target_persona": {
                    "name": "추가 논의 필요",
                    "age": "추가 논의 필요",
                    "job_role": "개발 실행 담당 팀",
                    "main_activities": deliverables,
                    "pain_points": [
                        "구현 범위와 책임 분담이 문서로 정리되지 않으면 실행 속도가 떨어집니다.",
                        "산출물 기준이 분산되면 개발과 검토 과정에서 반복 조율이 발생합니다.",
                    ],
                    "needs": [
                        f"{due_date}까지 바로 실행 가능한 개발 기준 문서가 필요합니다.",
                        "역할, 산출물, 우선순위를 한 번에 볼 수 있는 정리된 실행 초안이 필요합니다.",
                    ],
                },
            },
            "ground_rules": (
                "1. 개발 범위와 우선순위는 수집된 목표, 역할, 산출물, 마감 기준을 우선 근거로 삼습니다.\n"
                "2. 역할별 책임과 산출물 연결 관계를 문서에 명시한 뒤 작업을 시작합니다.\n"
                "3. 세부 기술 선택과 구현 방식은 팀 합의 전까지 확정 표현으로 쓰지 않습니다.\n"
                "4. 마감에 영향을 주는 변경 사항은 문서에 즉시 반영하고 공유합니다."
            ),
        }

    return {
        "project_home": {
            "project_overview": (
                f"{title} 프로젝트입니다. 목표는 {goal} "
                f"현재 기준 팀 규모는 {team_size}, 주요 산출물은 {deliverables}, "
                f"마감 기준은 {due_date}입니다."
            ),
        },
        "planning": {
            "project_intro": f"{title}: {goal}",
            "problem_definition": [
                {
                    "id": 1,
                    "situation": (
                        f"{title} 프로젝트는 목표가 정해져 있지만 실행 기준을 한 문서로 "
                        "정리할 필요가 있습니다."
                    ),
                    "reason": (
                        f"팀 규모 {team_size} 기준으로 역할, 산출물, 일정이 분산되면 "
                        "실행 우선순위가 흔들릴 수 있습니다."
                    ),
                    "limitation": (
                        "현재 수집된 최소 정보만으로는 세부 사용자 시나리오와 세부 요구사항을 "
                        "확정하기 어렵습니다."
                    ),
                }
            ],
            "solution": {
                "core_summary": (
                    "목표, 역할, 산출물, 일정 기준을 하나의 템플릿에 정리해 팀의 실행 기준을 맞춥니다."
                ),
                "problem_solutions": [
                    {
                        "problem_id": 1,
                        "solution_desc": (
                            f"{goal}에 맞춰 역할은 {roles}, 산출물은 {deliverables}, "
                            f"마감은 {due_date} 기준으로 정리합니다."
                        ),
                    }
                ],
                "features": [
                    f"프로젝트 목표 정리: {goal}",
                    f"팀 운영 기준 정리: {roles} / {team_size}",
                    f"산출물 및 일정 기준 정리: {deliverables} / {due_date}",
                ],
            },
            "target_persona": {
                "name": "추가 논의 필요",
                "age": "추가 논의 필요",
                "job_role": roles,
                "main_activities": deliverables,
                "pain_points": [
                    "목표는 정해졌지만 세부 실행 기준과 우선순위가 문서로 정리되지 않았습니다.",
                    f"팀 규모 {team_size}에서 역할 경계와 일정 기준을 맞춰야 합니다.",
                ],
                "needs": [
                    f"{due_date}까지 공유 가능한 실행 템플릿이 필요합니다.",
                    "산출물과 역할을 한 번에 확인할 수 있는 기획 문서가 필요합니다.",
                ],
            },
        },
        "ground_rules": (
            "1. 프로젝트 목표, 역할, 산출물, 일정은 수집된 데이터를 기준으로 정리합니다.\n"
            "2. 최근 팀 대화는 문서의 톤과 우선순위를 보완하는 참고 문맥으로 활용합니다.\n"
            "3. 역할, 산출물, 일정은 팀이 합의한 표현으로 명시합니다.\n"
            "4. 목표와 직접 연결되는 실행 기준부터 먼저 정리합니다."
        ),
    }


def get_template_mode_config(action_type: str) -> dict[str, str]:
    if action_type == "BTN_DEV":
        return {
            "mode": "dev",
            "mode_label": "개발용 템플릿",
            "focus": (
                "문제 정의보다 개발 범위, 구현 기준, 역할 분담, 산출물 연결, "
                "실행 우선순위가 더 잘 드러나도록 작성하세요."
            ),
            "summary_fallback": "개발용 템플릿 초안을 생성했습니다.",
        }

    return {
        "mode": "plan",
        "mode_label": "기획용 템플릿",
        "focus": (
            "사용자 문제, 기획 의도, 산출물 방향, 팀 정렬 포인트가 더 잘 드러나도록 작성하세요."
        ),
        "summary_fallback": "기획용 템플릿 초안을 생성했습니다.",
    }


def merge_template_sections(base: dict, generated: dict) -> dict:
    merged = dict(base)

    project_home = generated.get("project_home")
    if isinstance(project_home, dict):
        overview = str(project_home.get("project_overview", "")).strip()
        if overview:
            merged["project_home"] = {"project_overview": overview}

    planning = generated.get("planning")
    if isinstance(planning, dict):
        merged_planning = dict(base["planning"])

        project_intro = str(planning.get("project_intro", "")).strip()
        if project_intro:
            merged_planning["project_intro"] = project_intro

        problem_definition = planning.get("problem_definition")
        if isinstance(problem_definition, list) and problem_definition:
            cleaned_problem_definition = []
            for index, item in enumerate(problem_definition, start=1):
                if not isinstance(item, dict):
                    continue
                cleaned_problem_definition.append(
                    {
                        "id": item.get("id") or index,
                        "situation": str(item.get("situation", "")).strip(),
                        "reason": str(item.get("reason", "")).strip(),
                        "limitation": str(item.get("limitation", "")).strip(),
                    }
                )
            if cleaned_problem_definition:
                merged_planning["problem_definition"] = cleaned_problem_definition

        solution = planning.get("solution")
        if isinstance(solution, dict):
            merged_solution = dict(base["planning"]["solution"])

            core_summary = str(solution.get("core_summary", "")).strip()
            if core_summary:
                merged_solution["core_summary"] = core_summary

            problem_solutions = solution.get("problem_solutions")
            if isinstance(problem_solutions, list) and problem_solutions:
                cleaned_problem_solutions = []
                for item in problem_solutions:
                    if not isinstance(item, dict):
                        continue
                    solution_desc = str(item.get("solution_desc", "")).strip()
                    if not solution_desc:
                        continue
                    cleaned_problem_solutions.append(
                        {
                            "problem_id": item.get("problem_id") or 1,
                            "solution_desc": solution_desc,
                        }
                    )
                if cleaned_problem_solutions:
                    merged_solution["problem_solutions"] = cleaned_problem_solutions

            features = solution.get("features")
            if isinstance(features, list):
                cleaned_features = [
                    str(feature).strip() for feature in features if str(feature).strip()
                ]
                if cleaned_features:
                    merged_solution["features"] = cleaned_features

            merged_planning["solution"] = merged_solution

        target_persona = planning.get("target_persona")
        if isinstance(target_persona, dict):
            merged_persona = dict(base["planning"]["target_persona"])
            for key in ["name", "age", "job_role", "main_activities"]:
                value = str(target_persona.get(key, "")).strip()
                if value:
                    merged_persona[key] = value

            for key in ["pain_points", "needs"]:
                values = target_persona.get(key)
                if isinstance(values, list):
                    cleaned_values = [
                        str(value).strip() for value in values if str(value).strip()
                    ]
                    if cleaned_values:
                        merged_persona[key] = cleaned_values

            merged_planning["target_persona"] = merged_persona
        merged["planning"] = merged_planning

    ground_rules = generated.get("ground_rules")
    if isinstance(ground_rules, str) and ground_rules.strip():
        merged["ground_rules"] = ground_rules.strip()

    return merged


def build_notion_template_payload(state: AgentState, sections: dict) -> dict:
    return {
        "projectId": int(state["project_id"]),
        "templates": [
            {
                "key": "PROJECT_HOME",
                "parentKey": None,
                "title": "Project Home",
                "content": sections["project_home"],
            },
            {
                "key": "PLANNING",
                "parentKey": "PROJECT_HOME",
                "title": "기획",
                "content": sections["planning"],
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
        context_blocks.append("[최근 흐름 요약]\n" + flow_summary)

    return "\n\n".join(context_blocks) if context_blocks else "(전달된 최근 대화 없음)"


def build_template_input_summary(state: AgentState) -> str:
    snapshot = build_project_snapshot(state.get("collected_data") or {})
    return (
        f"- 주제: {snapshot['subject']}\n"
        f"- 제목: {snapshot['title']}\n"
        f"- 목표: {snapshot['goal']}\n"
        f"- 팀 규모: {snapshot['team_size']}\n"
        f"- 역할: {snapshot['roles']}\n"
        f"- 마감일: {snapshot['due_date']}\n"
        f"- 산출물: {snapshot['deliverables']}"
    )


def build_template_content_example() -> dict:
    return {
        "project_home": {
            "project_overview": "내용",
        },
        "planning": {
            "project_intro": "프로젝트 한 줄 소개",
            "problem_definition": [
                {
                    "id": 1,
                    "situation": "불편한 상황 (상황·경험 중심)",
                    "reason": "왜 문제인가?",
                    "limitation": "기존 해결 방식의 한계",
                }
            ],
            "solution": {
                "core_summary": "핵심 솔루션 한 줄 요약",
                "problem_solutions": [
                    {
                        "problem_id": 1,
                        "solution_desc": "문제 1에 대한 우리 서비스의 해결 방식",
                    }
                ],
                "features": [
                    "우리 솔루션의 특징 1",
                    "우리 솔루션의 특징 2",
                    "우리 솔루션의 특징 3",
                ],
            },
            "target_persona": {
                "name": "이름(가명)",
                "age": "나이",
                "job_role": "직업 / 역할",
                "main_activities": "주요 활동",
                "pain_points": [
                    "불편함 1",
                    "불편함 2",
                ],
                "needs": [
                    "니즈 1",
                    "니즈 2",
                ],
            },
        },
        "ground_rules": " ",
    }
