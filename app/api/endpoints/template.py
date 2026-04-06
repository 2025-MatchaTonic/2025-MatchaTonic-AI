import logging
import json
from json import JSONDecodeError
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, root_validator
from pydantic import ValidationError

from app.ai.graph.collected_data import (
    CollectedData,
    derive_phase_from_collected_data,
    missing_collected_fields,
    sanitize_collected_data,
)
from app.ai.graph.llm_clients import structured_llm
from app.ai.graph.nodes import _fetch_rag_context, _get_rag_filters
from app.ai.graph.state import AgentState
from app.ai.graph.template_support import (
    build_default_template_sections,
    build_notion_template_payload,
    build_recent_context,
    build_template_content_example,
    build_template_input_summary,
    get_template_mode_config,
    merge_template_sections,
)
from app.ai.graph.text_support import PLAIN_LANGUAGE_RULES
from app.ai.schemas.llm_outputs import TemplateContentLLMResponse
from app.api.schemas.template import NotionTemplatePayload
from app.core.request_normalization import (
    normalize_collected_data,
    normalize_optional_string,
    normalize_phase,
    normalize_string_list,
)

router = APIRouter()
logger = logging.getLogger(__name__)


class TemplateGenerateRequest(BaseModel):
    roomId: int
    templateType: Literal["plan", "dev"] = "plan"
    currentStatus: str = "READY"
    content: str = ""
    collectedData: CollectedData = Field(default_factory=dict)
    recentMessages: List[str] = Field(default_factory=list)
    selectedMessage: Optional[str] = None
    selectedAnswers: List[str] = Field(default_factory=list)
    actionType: Optional[str] = None

    @root_validator(pre=True)
    def normalize_spring_compatible_payload(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(values or {})

        if payload.get("roomId") is None and payload.get("projectId") is not None:
            payload["roomId"] = payload["projectId"]

        action_type = str(payload.get("actionType") or "").strip().upper()
        template_type = str(payload.get("templateType") or "").strip().lower()
        if not template_type:
            if action_type == "BTN_DEV":
                payload["templateType"] = "dev"
            elif action_type == "BTN_PLAN":
                payload["templateType"] = "plan"

        payload["currentStatus"] = normalize_phase(payload.get("currentStatus"), default="READY")
        payload["collectedData"] = normalize_collected_data(payload.get("collectedData"))

        selected_answers = normalize_string_list(payload.get("selectedAnswers"))
        payload["selectedAnswers"] = selected_answers

        cleaned_recent_messages = normalize_string_list(payload.get("recentMessages"))
        if not cleaned_recent_messages and selected_answers:
            cleaned_recent_messages = list(selected_answers)
        payload["recentMessages"] = cleaned_recent_messages

        selected_message = normalize_optional_string(payload.get("selectedMessage"))
        if selected_message is None:
            selected_message = selected_answers[-1] if selected_answers else None
        payload["selectedMessage"] = selected_message

        content = normalize_optional_string(payload.get("content")) or ""
        if not content:
            if selected_message:
                content = selected_message
            elif selected_answers:
                content = "\n".join(selected_answers)
        payload["content"] = content

        return payload


class TemplateGenerateResponse(BaseModel):
    content: str
    currentStatus: str
    notionTemplatePayload: NotionTemplatePayload


def _build_template_state(request: TemplateGenerateRequest) -> dict:
    user_message = request.content.strip()
    if not user_message:
        user_message = (
            "개발 템플릿 생성해줘"
            if request.templateType == "dev"
            else "기획 템플릿 생성해줘"
        )

    return {
        "project_id": str(request.roomId),
        "user_message": user_message,
        "action_type": "BTN_DEV" if request.templateType == "dev" else "BTN_PLAN",
        "current_phase": request.currentStatus,
        "turn_policy": "ANSWER_ONLY",
        "collected_data": request.collectedData,
        "recent_messages": request.recentMessages,
        "selected_message": request.selectedMessage,
        "is_sufficient": True,
        "ai_message": "",
        "next_phase": request.currentStatus,
        "template_payload": None,
    }


def _run_template_generation(request: TemplateGenerateRequest) -> tuple[dict, dict]:
    state = _build_template_state(request)
    result = (
        generate_dev_template(state)
        if request.templateType == "dev"
        else generate_plan_template(state)
    )
    payload = result.get("template_payload")
    if payload is None:
        raise RuntimeError("template generation returned no payload")
    return result, payload


def _build_template_state(request: TemplateGenerateRequest) -> dict:
    user_message = request.content.strip()
    if not user_message:
        user_message = (
            "개발 템플릿 생성해줘"
            if request.templateType == "dev"
            else "기획 템플릿 생성해줘"
        )

    approved_collected_data = sanitize_collected_data(request.collectedData)
    effective_phase = derive_phase_from_collected_data(
        approved_collected_data,
        current_phase=request.currentStatus,
    )

    return {
        "project_id": str(request.roomId),
        "user_message": user_message,
        "action_type": "BTN_DEV" if request.templateType == "dev" else "BTN_PLAN",
        "current_phase": effective_phase,
        "turn_policy": "ANSWER_ONLY",
        "collected_data": approved_collected_data,
        "recent_messages": request.recentMessages,
        "selected_message": request.selectedMessage,
        "is_sufficient": True,
        "ai_message": "",
        "next_phase": effective_phase,
        "template_payload": None,
    }


def _run_template_generation(request: TemplateGenerateRequest) -> tuple[dict, dict]:
    state = _build_template_state(request)
    logger.info(
        "template request room=%s current_status=%s effective_phase=%s missing_fields=%s collected_data=%s",
        request.roomId,
        request.currentStatus,
        state["current_phase"],
        missing_collected_fields(state["collected_data"]),
        state["collected_data"],
    )
    result = (
        generate_dev_template(state)
        if request.templateType == "dev"
        else generate_plan_template(state)
    )
    payload = result.get("template_payload")
    if payload is None:
        raise RuntimeError("template generation returned no payload")
    return result, payload


def generate_template_from_state(state: AgentState, *, action_type: str) -> dict:
    mode_config = get_template_mode_config(action_type)
    default_sections = build_default_template_sections(state, mode=mode_config["mode"])
    template_content_example = build_template_content_example()
    rag_filter_key = "READY_DEV" if action_type == "BTN_DEV" else "READY_PLAN"
    rag_context = _fetch_rag_context(state, phase="READY", **_get_rag_filters(rag_filter_key))
    recent_context = build_recent_context(state)
    template_input_summary = build_template_input_summary(state)
    collected_data = state.get("collected_data", {})
    missing_fields = missing_collected_fields(collected_data)
    missing_fields_summary = ", ".join(missing_fields) if missing_fields else "none"

    prompt = f"""
    You are a PM writing a Korean {mode_config["mode_label"]} draft.
    Output JSON only. Do not change the JSON shape.

    Core factual rules:
    - collected_data에 있는 값만 사실로 사용하세요.
    - 없는 값은 추측해 단정하지 마세요.
    - 예시는 예시로 표시하고, 사실처럼 쓰지 마세요.
    - missing field는 중립 표현으로 처리하세요.
    - projectId, key, parentKey, title은 생성하지 마세요.
    - content만 작성하세요.
    - target_persona.name, age, 구체 KPI, 특정 대상군, 특정 날짜는 collected_data에 없으면 확정하지 마세요.
    - recent conversation은 문체와 강조점 참고용입니다. collected_data에 없는 사실을 확정하는 근거로 쓰지 마세요.

    Section-level rules:
    - 일정 섹션: dueDate가 없으면 정확한 날짜를 쓰지 마세요.
    - 산출물 섹션: deliverables가 없으면 구체 산출물을 확정하지 마세요.
    - 역할 섹션: roles가 없으면 책임 분배를 확정하지 마세요.
    - 목표 섹션: goal이 없으면 "추가 논의 필요" 수준의 중립 표현만 쓰세요.

    Writing rules:
    - `problem_definition`과 `problem_solutions`는 각각 1개만 작성하세요.
    - `features`는 승인된 정보 기준으로 최대 3개만 작성하세요.
    - missing_fields={missing_fields_summary}
    - {mode_config["focus"]}
    {PLAIN_LANGUAGE_RULES}

    [Reference context]
    {rag_context}

    [Recent conversation]
    {recent_context}

    [Approved collected data]
    {json.dumps(collected_data, ensure_ascii=False)}

    [Missing fields]
    {missing_fields_summary}

    [Approved data summary]
    {template_input_summary}

    [Default draft guide]
    {json.dumps(default_sections, ensure_ascii=False, indent=2)}

    [Required JSON content shape]
    {{
      "summary_message": "short Korean summary",
      "project_home": {json.dumps(template_content_example["project_home"], ensure_ascii=False)},
      "planning": {json.dumps(template_content_example["planning"], ensure_ascii=False)},
      "ground_rules": " "
    }}
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

    template_sections = merge_template_sections(default_sections, result.to_merged_dict())
    template_payload = build_notion_template_payload(state, template_sections)
    summary_message = result.summary_message.strip() or mode_config["summary_fallback"]

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


@router.post("/", response_model=TemplateGenerateResponse)
async def generate_template(request: TemplateGenerateRequest):
    try:
        result, payload = _run_template_generation(request)
        return TemplateGenerateResponse(
            content=result.get("ai_message", ""),
            currentStatus=result.get("next_phase", request.currentStatus),
            notionTemplatePayload=payload,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Template generation failed: %s", exc)
        raise HTTPException(status_code=500, detail="템플릿 생성 중 오류가 발생했습니다.")


@router.post("/spring", response_model=NotionTemplatePayload)
async def generate_template_for_spring(request: TemplateGenerateRequest):
    try:
        _, payload = _run_template_generation(request)
        return payload
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Spring template generation failed: %s", exc)
        raise HTTPException(status_code=500, detail="템플릿 생성 중 오류가 발생했습니다.")
