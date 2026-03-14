# app/ai/graph/nodes.py
import json
from json import JSONDecodeError

from langchain_openai import ChatOpenAI

from app.ai.graph.collected_data import (
    build_collected_data_guide,
    build_collected_data_json_example,
    merge_collected_data,
)
from app.ai.graph.state import AgentState
from app.core.config import settings
from app.rag.retriever import get_rag_context

# NOTE
# The `collected_data` structure was recently extended with a new key
# "roles" (담당 역할).  All of the helper functions imported from
# `collected_data.py` are dynamic, so new fields are included
# automatically in prompts, examples and merging logic.  We also add
# some extra safety checks below so that nodes remain robust as the
# schema evolves.

LLM_MODEL = "gpt-5-mini-2025-08-07"

conversation_llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0.7,
    openai_api_key=settings.OPENAI_API_KEY,
)

structured_llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0.2,
    openai_api_key=settings.OPENAI_API_KEY,
)

RAG_EMPTY_CONTEXT = "(관련 레퍼런스를 찾지 못했습니다.)"


PLAIN_LANGUAGE_RULES = """
- 참고 레퍼런스의 전문용어를 그대로 복붙하지 말고, 초보 팀도 이해할 수 있는 쉬운 한국어로 풀어서 설명하세요.
- 꼭 필요한 전문용어를 써야 하면 한 번만 쓰고, 바로 뒤에 괄호나 짧은 설명으로 뜻을 덧붙이세요.
- 문장은 짧고 분명하게 쓰고, 현업자가 아닌 사람도 바로 이해할 수 있는 표현을 우선하세요.
""".strip()


def _wants_detailed_answer(user_message: str) -> bool:
    detail_keywords = (
        "자세히",
        "구체",
        "상세",
        "예시",
        "템플릿",
        "단계",
        "체크리스트",
        "플랜",
    )
    return any(keyword in user_message for keyword in detail_keywords)


def _build_rag_query(state: AgentState) -> str:
    user_message = (state.get("user_message") or "").strip()
    selected = (state.get("selected_message") or "").strip()
    recent = [msg.strip() for msg in state.get("recent_messages", []) if msg.strip()]

    parts = [user_message]
    if selected:
        parts.append(selected)
    if recent:
        parts.append(" | ".join(recent[-5:]))
    return "\n".join(part for part in parts if part).strip()


def _fetch_rag_context(state: AgentState, phase: str) -> str:
    query = _build_rag_query(state)
    if not query:
        return RAG_EMPTY_CONTEXT

    context = get_rag_context(query=query, current_phase=phase)
    return context or RAG_EMPTY_CONTEXT


def _clean_text(value: object) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _build_project_snapshot(data: dict) -> dict[str, str]:
    title = _clean_text(data.get("title")) or "프로젝트 제목 미정"
    goal = _clean_text(data.get("goal")) or "프로젝트 목표는 추가 논의가 필요합니다."
    team_size = _clean_text(data.get("teamSize")) or "팀 규모 미정"
    roles = _clean_text(data.get("roles")) or "역할 분담 추가 논의 필요"
    due_date = _clean_text(data.get("dueDate")) or "마감 일정 미정"
    deliverables = _clean_text(data.get("deliverables")) or "산출물 범위 추가 논의 필요"

    return {
        "title": title,
        "goal": goal,
        "team_size": team_size,
        "roles": roles,
        "due_date": due_date,
        "deliverables": deliverables,
    }


def _build_default_template_sections(state: AgentState, mode: str = "plan") -> dict:
    snapshot = _build_project_snapshot(state.get("collected_data") or {})
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


def _get_template_mode_config(action_type: str) -> dict[str, str]:
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


def _merge_template_sections(base: dict, generated: dict) -> dict:
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


def _build_notion_template_payload(state: AgentState, sections: dict) -> dict:
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


def _build_recent_context(state: AgentState) -> str:
    selected_message = (state.get("selected_message") or "").strip()
    recent_messages = [msg.strip() for msg in state.get("recent_messages", []) if msg.strip()]

    context_blocks: list[str] = []
    if selected_message:
        context_blocks.append(f"[현재 집중 포인트]\n- {selected_message}")
    if recent_messages:
        latest_messages = recent_messages[-5:]
        flow_summary = "\n".join(f"- {message}" for message in latest_messages)
        context_blocks.append("[최근 흐름 요약]\n" + flow_summary)

    return "\n\n".join(context_blocks) if context_blocks else "(전달된 최근 대화 없음)"


def _build_template_input_summary(state: AgentState) -> str:
    snapshot = _build_project_snapshot(state.get("collected_data") or {})
    return (
        f"- 제목: {snapshot['title']}\n"
        f"- 목표: {snapshot['goal']}\n"
        f"- 팀 규모: {snapshot['team_size']}\n"
        f"- 역할: {snapshot['roles']}\n"
        f"- 마감일: {snapshot['due_date']}\n"
        f"- 산출물: {snapshot['deliverables']}"
    )


def _build_template_content_example() -> dict:
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


def _build_topic_exists_fallback_message() -> str:
    return (
        "좋아요. 이미 프로젝트 주제가 있다면 이제 팀원끼리 자유롭게 논의하면 됩니다. "
        "막히는 지점이 생기면 최근 대화와 함께 @mates 를 호출해 주세요. "
        "제가 문맥을 읽고 방향 정리, 우선순위, 역할 분담, 요구사항 정리까지 도와드릴게요."
    )


# ----------------------------------------------------
# 1. 아이디어가 없을 때 (NO 선택) : 탐색 노드
def explore_problem_node(state: AgentState):
    rag_context = _fetch_rag_context(state, phase="EXPLORE")
    recent_context = _build_recent_context(state)
    prompt = f"""
    당신은 친절하고 센스 있는 브레인스토밍 파트너입니다. 
    사용자가 아직 프로젝트 주제가 없거나, 불편함을 탐색하는 중입니다.

    [참고 레퍼런스]
    {rag_context}

    [최근 대화 문맥]
    {recent_context}

    [중요 지시사항]
    1. 무조건 **한 번에 딱 1개의 질문**만 던지세요. 절대 여러 개를 동시에 묻지 마세요!
    2. 사용자가 대답을 했다면, 먼저 그 대답에 깊이 공감해 준 뒤에 꼬리를 무는 질문을 1개 던지세요.
    3. 첫 시작이라면 이렇게 가볍게 물어보세요: "최근 일주일 동안 '아, 이거 진짜 귀찮다' 했던 적이 있나요?"
    4. 대화가 자연스럽게 이어지도록 친구처럼 편안한 말투를 사용하세요.
    5. 아래 원칙을 반드시 지키세요.
    {PLAIN_LANGUAGE_RULES}
    
    [사용자 입력]
    {state['user_message']}
    """

    response = conversation_llm.invoke(prompt)

    return {
        "ai_message": response.content,
        "next_phase": "EXPLORE",  # 계속 탐색 단계 유지
    }


# ----------------------------------------------------
# 1-2. 아이디어가 있을 때 (YES 선택) : 팀 대화 안내 노드
# ----------------------------------------------------
def topic_exists_node(state: AgentState):
    user_message = (state.get("user_message") or "").strip()
    recent_context = _build_recent_context(state)
    has_recent_context = recent_context != "(전달된 최근 대화 없음)"

    if not user_message and not has_recent_context:
        ai_message = _build_topic_exists_fallback_message()
    else:
        prompt = f"""
        당신은 프로젝트 주제가 이미 정해진 팀을 돕는 AI PM입니다.
        사용자의 최신 입력과 최근 팀 대화를 읽고, 지금 상황에 맞는 짧은 안내를 한국어로 작성하세요.
        아래 규칙을 지키세요.

        규칙:
        - 2~3문장으로만 답하세요.
        - 먼저 사용자의 현재 논의 주제나 관심사를 짧게 짚어 주세요.
        - 그 다음 팀원끼리 이어서 논의해도 된다는 점과, 막히면 @mates로 도움받을 수 있다는 점을 자연스럽게 안내하세요.
        - "무엇을 도와드릴까요?", "다음 중 선택하세요" 같은 고객센터식 문장은 금지합니다.
        - 입력 정보가 적으면 과장하지 말고 일반적인 표현으로 답하세요.
        - 아래 원칙을 반드시 지키세요.
        {PLAIN_LANGUAGE_RULES}

        [최근 대화 문맥]
        {recent_context}

        [사용자 입력]
        {user_message}
        """
        response = conversation_llm.invoke(prompt)
        ai_message = str(response.content).strip() or _build_topic_exists_fallback_message()

    return {
        "ai_message": ai_message,
        "next_phase": "TOPIC_SET",
    }


# ----------------------------------------------------
# 2. 아이디어가 있을 때 (YES 선택) : 정보 수집 노드 (자연스러운 HMW)
# ----------------------------------------------------
def gather_information_node(state: AgentState):
    # make sure we always have a dict for collected_data; older states
    # may not include the new "roles" key yet.
    current_data = state.get("collected_data") or {}
    rag_context = _fetch_rag_context(state, phase=state.get("current_phase", "GATHER"))
    recent_context = _build_recent_context(state)

    # 시스템 몰래 데이터를 평가하는 프롬프트 (JSON 반환 강제)
    eval_prompt = f"""
    당신은 사용자와 대화하며 프로젝트 기획에 필요한 핵심 정보를 모으는 PM입니다.
    딱딱한 학술적 HMW가 아니라, "그럼 우리가 어떻게 하면 ~할 수 있을까요?" 처럼 자연스럽고 친근하게 질문하세요.
    한 번에 너무 많은 걸 묻지 말고, 비어있는 정보를 하나씩 유도하세요.
    이미 들어온 정보는 유지하고, 새로 확인된 항목만 보강하세요.
    필요한 정보 항목은 다음 키를 기준으로 정리하세요: {build_collected_data_guide()}  
    (최근에 "roles"(담당 역할) 필드가 추가되었습니다. 빠뜨리지 말고 물어보세요.)
    아래 원칙을 반드시 지키세요.
    {PLAIN_LANGUAGE_RULES}

    [참고할 소프트웨어 엔지니어링 레퍼런스]
    {rag_context}

    [최근 대화 문맥]
    {recent_context}
    
    [현재까지 모인 정보]
    {json.dumps(current_data, ensure_ascii=False)}
    
    [사용자 대답]
    {state['user_message']}
    
    [출력 형식 강제 (반드시 JSON 포맷으로 응답하세요)]
    {{
        "ai_message": "사용자에게 할 자연스러운 챗봇 응답 텍스트",
        "updated_data": {build_collected_data_json_example()},
        "is_sufficient": false
    }}

    [판단 기준]
    - updated_data에는 이번 대화로 명확해진 값만 채우세요.
    - 정보가 부족하면 ai_message에는 딱 한 가지 후속 질문만 넣으세요.
    - 모든 필수 항목이 충분히 구체적일 때만 is_sufficient를 true로 바꾸세요.
    """

    # JSON 형태로만 응답하도록 강제
    response = structured_llm.invoke(
        eval_prompt,
        response_format={"type": "json_object"},
    )
    try:
        # The LLM is expected to return JSON that conforms to the example
        # above.  If the newly added "roles" field was mishandled we want
        # to see the raw output for debugging.
        result = json.loads(response.content)
    except JSONDecodeError as exc:
        # include the raw content in the exception so callers can inspect
        # what the model produced.
        raise RuntimeError(
            f"failed to parse JSON from LLM response: {exc}\n"
            f"raw output:\n{response.content}"
        )

    ai_msg = result.get("ai_message", "")
    is_sufficient = result.get("is_sufficient", False)
    merged_data = merge_collected_data(current_data, result.get("updated_data"))

    # 데이터가 다 모였다면 템플릿 생성 안내 멘트 추가
    if is_sufficient:
        ai_msg += (
            "\n\n필수 정보가 모두 모였습니다. "
            "원하면 지금 템플릿 생성 단계로 넘어갈 수 있어요."
        )
        next_phase = "READY"
    else:
        next_phase = "GATHER"

    return {
        "ai_message": ai_msg,
        "collected_data": merged_data,
        "is_sufficient": is_sufficient,
        "next_phase": next_phase,
    }


# ----------------------------------------------------
# 3. 템플릿 생성 노드
# ----------------------------------------------------
def _generate_template_node(state: AgentState, action_type: str):
    mode_config = _get_template_mode_config(action_type)
    default_sections = _build_default_template_sections(state, mode=mode_config["mode"])
    template_content_example = _build_template_content_example()
    rag_context = _fetch_rag_context(state, phase="READY")
    recent_context = _build_recent_context(state)
    template_input_summary = _build_template_input_summary(state)
    prompt = f"""
    당신은 {mode_config["mode_label"]}용 노션 프로젝트 템플릿 문구를 작성하는 PM입니다.
    아래 수집된 데이터를 바탕으로 사용자가 바로 노션에 옮길 수 있는 한국어 텍스트를 생성하세요.
    반드시 JSON 객체로만 응답하고, 지정된 키 이름을 그대로 사용하세요.
    projectId, key, parentKey, title은 Python 코드에서 고정 조립하므로 생성하지 마세요.
    collected_data에 들어 있는 값은 모두 확정 정보로 보고 적극 반영하세요.
    모르는 정보는 과장하지 말고, 현재 데이터 범위 안에서만 보완하세요.
    일정, 스프린트, 마일스톤 같은 정보는 명시적으로 주어지지 않으면 넣지 마세요.
    사용자가 제공한 최소 정보로도 바로 쓸 수 있는 초안을 만드는 것이 목적입니다.
    세부 사용자 문제, 시장 정보, 구체 연령, 정량 지표처럼 collected_data에서 직접 확인되지 않는 값은
    사실처럼 단정하지 말고 중립적인 표현으로 작성하세요.
    `problem_definition`과 `problem_solutions`는 각각 1개만 작성하세요.
    `features`는 현재 확정된 정보를 반영한 3개 항목으로 작성하세요.
    최근 대화는 팀이 어떤 방향을 고민하는지 파악하는 참고 문맥입니다.
    문서의 톤과 강조점은 최근 대화를 반영해도 되지만, collected_data에 없는 값을
    최근 대화만으로 확정 사실처럼 쓰지 마세요.
    특히 `target_persona.name`, `target_persona.age`, 세부 일정, 구체 수치, 특정 사용자군은
    근거가 없으면 채워 넣지 말고 중립적으로 작성하세요.
    `ground_rules`는 수집된 데이터를 기준으로 팀이 실행할 때 지킬 원칙 중심으로 작성하세요.
    {mode_config["focus"]}
    최종 payload의 템플릿 구조는 백엔드에서 고정 조립됩니다. key, parentKey, title, content 구조는 절대 바꾸지 않습니다.
    당신은 아래 content 부분만 생성하며, 필드 이름과 자료형은 예시와 정확히 같아야 합니다.
    전문용어를 그대로 옮기기보다 초보 팀도 바로 이해할 수 있게 쉬운 표현으로 다시 써주세요.
    참고 레퍼런스의 용어를 꼭 써야 하면 짧은 설명을 덧붙이거나 더 쉬운 말로 바꾸세요.

    [참고할 소프트웨어 엔지니어링 레퍼런스]
    {rag_context}

    [최근 대화 문맥]
    {recent_context}

    [수집된 데이터]
    {json.dumps(state.get("collected_data", {}), ensure_ascii=False)}

    [현재 확정 정보 요약]
    {template_input_summary}

    [기본 초안 가이드]
    {json.dumps(default_sections, ensure_ascii=False, indent=2)}

    [반드시 따라야 하는 content JSON 형식]
    {{
      "summary_message": "스프링으로 전달할 때 함께 보여줄 한 줄 안내",
      "project_home": {json.dumps(template_content_example["project_home"], ensure_ascii=False)},
      "planning": {json.dumps(template_content_example["planning"], ensure_ascii=False)},
      "ground_rules": " "
    }}

    출력 전 자체 점검:
    - collected_data의 핵심 값이 빠지지 않고 반영되었는가?
    - 최근 대화에서 읽은 맥락은 설명 보완용으로만 사용했는가?
    - collected_data에 없는 값을 확정형 문장으로 과장하지 않았는가?
    - project_home, planning, ground_rules 외의 최상위 키를 추가하지 않았는가?
    - planning 내부 구조를 예시와 다르게 바꾸지 않았는가?
    """

    response = structured_llm.invoke(prompt, response_format={"type": "json_object"})
    try:
        result = json.loads(response.content)
    except JSONDecodeError as exc:
        raise RuntimeError(
            f"failed to parse template JSON from LLM response: {exc}\n"
            f"raw output:\n{response.content}"
        )

    template_sections = _merge_template_sections(default_sections, result)
    template_payload = _build_notion_template_payload(state, template_sections)
    summary_message = str(result.get("summary_message", "")).strip()
    if not summary_message:
        summary_message = mode_config["summary_fallback"]

    return {
        "ai_message": summary_message,
        "template_payload": template_payload,
        "next_phase": "DONE",
        "is_sufficient": True,
    }


def generate_plan_template_node(state: AgentState):
    return _generate_template_node(state, action_type="BTN_PLAN")


def generate_dev_template_node(state: AgentState):
    return _generate_template_node(state, action_type="BTN_DEV")


# ----------------------------------------------------
# 4. @mates 멘션 호출 노드
# ----------------------------------------------------
def mates_helper_node(state: AgentState):
    user_message = str(state.get("user_message", ""))
    detailed_mode = _wants_detailed_answer(user_message)
    response_style = (
        "- 상세 모드(사용자가 상세 요청 시): 최대 6개 bullet, 각 bullet 1~2문장, 완결된 문장으로만 답하세요.\n"
        if detailed_mode
        else "- 기본 모드(기본값): 2~4문장 또는 bullet 최대 3개, 모든 문장은 끝까지 완결되게 작성하세요.\n"
    )
    recent_context = _build_recent_context(state)
    rag_context = _fetch_rag_context(
        state,
        phase=state.get("current_phase", "TOPIC_SET"),
    )
    helper_prompt = f"""
    당신은 팀 대화방에 호출된 AI 팀메이트 @mates 입니다.
    사용자는 평소에는 팀원끼리만 대화하고, 막히는 시점에만 당신을 호출합니다.
    당신의 역할은 최근 팀 대화를 읽고 지금 프로젝트 진행에 실질적으로 도움이 되는 답을 주는 것입니다.

    규칙:
    - 사용자가 이미 질문했으면 바로 답하세요. 다시 "무엇을 도와드릴까요?"라고 묻지 마세요.
    - 최근 팀 대화와 핵심 채팅이 있으면 반드시 반영하세요.
    - @mates 멘션 자체는 무시하고, 실질적인 요청에 집중하세요.
    - 한국어로 답하고, 팀 협업에 바로 쓸 수 있게 구체적으로 제안하세요.
    - 프로젝트 진행에 도움이 되도록 다음 중 "필요한 것만" 포함하세요: 방향 정리, 우선순위 제안, 역할 분담, 요구사항 구조화, 다음 액션.
    - 사용자가 개수나 형식을 요청했다면 그대로 맞추세요.
    - 요청이 모호할 때만 한 가지 짧은 확인 질문을 하세요.
    - 장황한 설명, 섹션 헤더, 과도한 단계별 매뉴얼, "원하시면 더 해드릴게요" 같은 꼬리말은 금지합니다.
    - 답변은 처음부터 길이를 맞춰 작성하고, 문장 중간에 끊기지 않게 완결된 형태로 끝내세요.
    - 아래 원칙을 반드시 지키세요.
    {PLAIN_LANGUAGE_RULES}

    출력 길이 규칙:
    {response_style}

    최근 대화 문맥:
    {recent_context}

    참고 레퍼런스:
    {rag_context}

    사용자 메시지:
    {user_message}
    """
    response = conversation_llm.invoke(helper_prompt)
    ai_message = str(response.content).strip()
    return {"ai_message": ai_message, "next_phase": state["current_phase"]}
