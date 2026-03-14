import json
from json import JSONDecodeError

from app.ai.schemas.llm_outputs import TemplateContentLLMResponse
from pydantic import ValidationError
from app.ai.graph.nodes import (
    PLAIN_LANGUAGE_RULES,
    _build_default_template_sections,
    _build_notion_template_payload,
    _build_recent_context,
    _build_template_content_example,
    _build_template_input_summary,
    _fetch_rag_context,
    _get_rag_filters,
    _get_template_mode_config,
    _merge_template_sections,
    structured_llm,
)
from app.ai.graph.state import AgentState


def generate_template_from_state(state: AgentState, *, action_type: str) -> dict:
    mode_config = _get_template_mode_config(action_type)
    default_sections = _build_default_template_sections(state, mode=mode_config["mode"])
    template_content_example = _build_template_content_example()
    rag_filter_key = "READY_DEV" if action_type == "BTN_DEV" else "READY_PLAN"
    rag_context = _fetch_rag_context(state, phase="READY", **_get_rag_filters(rag_filter_key))
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
    아래 원칙을 반드시 지키세요.
    {PLAIN_LANGUAGE_RULES}

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
        raw_result = json.loads(response.content)
        result = TemplateContentLLMResponse.model_validate(raw_result)
    except (JSONDecodeError, ValidationError) as exc:
        raise RuntimeError(
            f"failed to parse template JSON from LLM response: {exc}\n"
            f"raw output:\n{response.content}"
        )

    template_sections = _merge_template_sections(default_sections, result.to_merged_dict())
    template_payload = _build_notion_template_payload(state, template_sections)
    summary_message = result.summary_message.strip()
    if not summary_message:
        summary_message = mode_config["summary_fallback"]

    return {
        "ai_message": summary_message,
        "template_payload": template_payload,
        "next_phase": "DONE",
        "is_sufficient": True,
    }


def generate_plan_template(state: AgentState) -> dict:
    return generate_template_from_state(state, action_type="BTN_PLAN")


def generate_dev_template(state: AgentState) -> dict:
    return generate_template_from_state(state, action_type="BTN_DEV")



