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

llm = ChatOpenAI(
    model="gpt-5-mini-2025-08-07",
    temperature=0.7,
    openai_api_key=settings.OPENAI_API_KEY,
)

RAG_EMPTY_CONTEXT = "(관련 레퍼런스를 찾지 못했습니다.)"
MATES_DEFAULT_MAX_CHARS = 280
MATES_DEFAULT_MAX_LINES = 4
MATES_DETAILED_MAX_CHARS = 700
MATES_DETAILED_MAX_LINES = 10


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


def _compact_mates_answer(text: str, *, max_chars: int, max_lines: int) -> str:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return ""

    compact = "\n".join(lines[:max_lines])
    if len(compact) <= max_chars:
        return compact

    clipped = compact[:max_chars].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0].rstrip()
    return f"{clipped}..."


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


def _build_default_template_sections(state: AgentState) -> dict:
    data = state.get("collected_data") or {}
    topic = data.get("topic") or "프로젝트 방향을 정리하는 초안"
    solution = data.get("solution") or "사용자 문제를 해결할 핵심 접근 방식을 구체화합니다."
    requirements = data.get("requirements") or "필수 요구사항과 우선순위를 팀과 함께 정리합니다."
    impact = data.get("impact") or "사용자에게 줄 변화와 기대 효과를 명확히 정의합니다."
    deliverables = data.get("deliverables") or "노션 기반 프로젝트 문서와 실행 초안을 정리합니다."
    roles = data.get("roles") or data.get("role") or "프로젝트를 기획하고 실행 방향을 정리하는 팀원"

    return {
        "project_home": {
            "project_overview": f"{topic}. {impact}",
        },
        "planning": {
            "project_intro": topic,
            "problem_definition": [
                {
                    "id": 1,
                    "situation": f"{topic}와 관련된 사용자가 현재 겪는 불편이나 비효율이 존재합니다.",
                    "reason": impact,
                    "limitation": requirements,
                }
            ],
            "solution": {
                "core_summary": solution,
                "problem_solutions": [
                    {
                        "problem_id": 1,
                        "solution_desc": solution,
                    }
                ],
                "features": [
                    "핵심 사용자 문제를 빠르게 정리할 수 있는 구조",
                    "팀 역할과 요구사항을 한 번에 정렬하는 방식",
                    "실행 전 기대 효과를 검토할 수 있는 문서 흐름",
                ],
            },
            "target_persona": {
                "name": "핵심 사용자(가명)",
                "age": "20대 후반~30대 후반",
                "job_role": roles,
                "main_activities": deliverables,
                "pain_points": [
                    "문제 정의와 우선순위가 자주 흔들립니다.",
                    "팀 내 역할과 기대 결과물이 명확하지 않습니다.",
                ],
                "needs": [
                    "빠르게 합의 가능한 프로젝트 문서가 필요합니다.",
                    "실행 전에 요구사항과 기대 효과를 정리하고 싶습니다.",
                ],
            },
        },
        "ground_rules": (
            "1. 모든 결정은 사용자 문제와 기대 효과를 기준으로 정리합니다.\n"
            "2. 미확정 항목은 추정으로 단정하지 않고 후속 확인 대상으로 표시합니다.\n"
            "3. 역할, 요구사항, 산출물은 문서에 명시적으로 남깁니다."
        ),
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
        context_blocks.append(f"[핵심 채팅]\n{selected_message}")
    if recent_messages:
        context_blocks.append("[최근 팀 대화]\n" + "\n".join(recent_messages[-12:]))

    return "\n\n".join(context_blocks) if context_blocks else "(전달된 최근 대화 없음)"


# ----------------------------------------------------
# 1. 아이디어가 없을 때 (NO 선택) : 탐색 노드
def explore_problem_node(state: AgentState):
    rag_context = _fetch_rag_context(state, phase="EXPLORE")
    prompt = f"""
    당신은 친절하고 센스 있는 브레인스토밍 파트너입니다. 
    사용자가 아직 프로젝트 주제가 없거나, 불편함을 탐색하는 중입니다.

    [참고 레퍼런스]
    {rag_context}

    [중요 지시사항]
    1. 무조건 **한 번에 딱 1개의 질문**만 던지세요. 절대 여러 개를 동시에 묻지 마세요!
    2. 사용자가 대답을 했다면, 먼저 그 대답에 깊이 공감해 준 뒤에 꼬리를 무는 질문을 1개 던지세요.
    3. 첫 시작이라면 이렇게 가볍게 물어보세요: "최근 일주일 동안 '아, 이거 진짜 귀찮다' 했던 적이 있나요?"
    4. 대화가 자연스럽게 이어지도록 친구처럼 편안한 말투를 사용하세요.
    
    [사용자 입력]
    {state['user_message']}
    """

    response = llm.invoke(prompt)

    return {
        "ai_message": response.content,
        "next_phase": "EXPLORE",  # 계속 탐색 단계 유지
    }


# ----------------------------------------------------
# 1-2. 아이디어가 있을 때 (YES 선택) : 팀 대화 안내 노드
# ----------------------------------------------------
def topic_exists_node(state: AgentState):
    return {
        "ai_message": (
            "좋아요. 이미 프로젝트 주제가 있다면 이제 팀원끼리 자유롭게 논의하면 됩니다. "
            "막히는 지점이 생기면 최근 대화와 함께 @mates 를 호출해 주세요. "
            "제가 문맥을 읽고 방향 정리, 우선순위, 역할 분담, 요구사항 정리까지 도와드릴게요."
        ),
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

    # 시스템 몰래 데이터를 평가하는 프롬프트 (JSON 반환 강제)
    eval_prompt = f"""
    당신은 사용자와 대화하며 프로젝트 기획에 필요한 핵심 정보를 모으는 PM입니다.
    딱딱한 학술적 HMW가 아니라, "그럼 우리가 어떻게 하면 ~할 수 있을까요?" 처럼 자연스럽고 친근하게 질문하세요.
    한 번에 너무 많은 걸 묻지 말고, 비어있는 정보를 하나씩 유도하세요.
    이미 들어온 정보는 유지하고, 새로 확인된 항목만 보강하세요.
    필요한 정보 항목은 다음 키를 기준으로 정리하세요: {build_collected_data_guide()}  
    (최근에 "roles"(담당 역할) 필드가 추가되었습니다. 빠뜨리지 말고 물어보세요.)

    [참고할 소프트웨어 엔지니어링 레퍼런스]
    {rag_context}
    
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
    response = llm.invoke(eval_prompt, response_format={"type": "json_object"})
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
        ai_msg += "\n\n🎉 완벽해요! 프로젝트에 필요한 핵심 정보가 모두 모였습니다. 이제 기획용 템플릿을 만들까요, 개발용 템플릿을 만들까요?"
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
def generate_template_node(state: AgentState):
    default_sections = _build_default_template_sections(state)
    rag_context = _fetch_rag_context(state, phase="READY")
    prompt = f"""
    당신은 노션 프로젝트 템플릿용 문구를 작성하는 PM입니다.
    아래 수집된 데이터를 바탕으로 사용자가 바로 노션에 옮길 수 있는 한국어 텍스트를 생성하세요.
    반드시 JSON 객체로만 응답하고, 지정된 키 이름을 그대로 사용하세요.
    projectId, key, parentKey, title은 Python 코드에서 고정 조립하므로 생성하지 마세요.
    모르는 정보는 과장하지 말고, 현재 데이터 범위 안에서 자연스럽고 구체적인 초안으로 보완하세요.
    일정, 스프린트, 마일스톤 같은 정보는 명시적으로 주어지지 않으면 넣지 마세요.

    [참고할 소프트웨어 엔지니어링 레퍼런스]
    {rag_context}

    [수집된 데이터]
    {json.dumps(state.get("collected_data", {}), ensure_ascii=False)}

    [응답 JSON 형식]
    {{
      "summary_message": "스프링으로 전달할 때 함께 보여줄 한 줄 안내",
      "project_home": {{
        "project_overview": "프로젝트 전체 개요"
      }},
      "planning": {{
        "project_intro": "프로젝트 한 줄 소개",
        "problem_definition": [
          {{
            "id": 1,
            "situation": "불편한 상황",
            "reason": "왜 문제인지",
            "limitation": "기존 해결 방식의 한계"
          }}
        ],
        "solution": {{
          "core_summary": "핵심 솔루션 요약",
          "problem_solutions": [
            {{
              "problem_id": 1,
              "solution_desc": "문제 1 해결 방식"
            }}
          ],
          "features": ["특징 1", "특징 2", "특징 3"]
        }},
        "target_persona": {{
          "name": "이름(가명)",
          "age": "나이",
          "job_role": "직업/역할",
          "main_activities": "주요 활동",
          "pain_points": ["불편함 1", "불편함 2"],
          "needs": ["니즈 1", "니즈 2"]
        }}
      }},
      "ground_rules": "그라운드룰 본문"
    }}
    """

    response = llm.invoke(prompt, response_format={"type": "json_object"})
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
        summary_message = "노션 템플릿 초안을 생성했습니다. Spring에서 이 payload를 조립해 사용하면 됩니다."

    return {
        "ai_message": summary_message,
        "template_payload": template_payload,
        "next_phase": "DONE",
        "is_sufficient": True,
    }


# ----------------------------------------------------
# 4. @mates 멘션 호출 노드
# ----------------------------------------------------
def mates_helper_node(state: AgentState):
    user_message = str(state.get("user_message", ""))
    detailed_mode = _wants_detailed_answer(user_message)
    response_style = (
        "- 상세 모드(사용자가 상세 요청 시): 최대 6개 bullet, 각 bullet 1~2문장, 총 700자 이내.\n"
        if detailed_mode
        else "- 기본 모드(기본값): 2~4문장 또는 bullet 최대 3개, 총 280자 이내.\n"
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

    출력 길이 규칙:
    {response_style}

    최근 대화 문맥:
    {recent_context}

    참고 레퍼런스:
    {rag_context}

    사용자 메시지:
    {user_message}
    """
    response = llm.invoke(helper_prompt)
    max_chars = MATES_DETAILED_MAX_CHARS if detailed_mode else MATES_DEFAULT_MAX_CHARS
    max_lines = MATES_DETAILED_MAX_LINES if detailed_mode else MATES_DEFAULT_MAX_LINES
    ai_message = _compact_mates_answer(
        response.content,
        max_chars=max_chars,
        max_lines=max_lines,
    )
    return {"ai_message": ai_message, "next_phase": state["current_phase"]}
