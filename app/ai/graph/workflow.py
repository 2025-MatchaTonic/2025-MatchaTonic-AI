# app/ai/graph/workflow.py
from langgraph.graph import StateGraph, END
from app.ai.graph.state import AgentState
from app.ai.graph.nodes import (
    _extract_direct_fact_updates,
    _extract_topic_candidate,
    _is_meaningful_fact,
    _normalize_topic_title,
    _prune_collected_data,
    explore_problem_node,
    gather_information_node,
    topic_exists_node,
)
from app.ai.services.template_generation import generate_dev_template, generate_plan_template


def _has_title(state: AgentState) -> bool:
    current_data = _prune_collected_data(state.get("collected_data") or {})
    return _is_meaningful_fact(current_data.get("title"))


def _has_any_collected_fact(state: AgentState) -> bool:
    current_data = _prune_collected_data(state.get("collected_data") or {})
    return bool(current_data)


def _should_promote_explore_to_topic_set(state: AgentState) -> bool:
    if _has_any_collected_fact(state):
        return True

    user_message = str(state.get("user_message") or "").strip()
    if not user_message:
        return False

    if _normalize_topic_title(_extract_topic_candidate(user_message)):
        return True

    return bool(_extract_direct_fact_updates(user_message))


def route_logic(state: AgentState):
    action = state["action_type"]
    phase = state["current_phase"]
    turn_policy = state["turn_policy"]

    # 1. 초기 "프로젝트 주제가 있나요?" 응답 처리
    if action == "BTN_NO":
        return "explore_node"
    elif action in ["BTN_YES", "BTN_GO_DEF"]:
        return "topic_exists_node"

    if phase == "TOPIC_SET" and turn_policy in {"ASK_ONLY", "CAPTURE_TITLE"}:
        return "topic_exists_node"

    # 2. 명시적 AI 기능 호출
    if action == "BTN_PLAN":
        return "generate_plan_node"
    if action == "BTN_DEV":
        return "generate_dev_node"

    # 3. 일반 채팅 흐름
    if action == "CHAT":
        if phase == "EXPLORE":
            if _has_title(state):
                return "gather_node"
            if _should_promote_explore_to_topic_set(state):
                return "topic_exists_node"
            return "explore_node"
        if phase in ["TOPIC_SET", "GATHER", "READY"]:
            return "gather_node"

    return END


workflow = StateGraph(AgentState)

workflow.add_node("explore_node", explore_problem_node)
workflow.add_node("gather_node", gather_information_node)
workflow.add_node("generate_plan_node", generate_plan_template)
workflow.add_node("generate_dev_node", generate_dev_template)
workflow.add_node("topic_exists_node", topic_exists_node)

workflow.set_conditional_entry_point(
    route_logic,
    {
        "explore_node": "explore_node",
        "gather_node": "gather_node",
        "generate_plan_node": "generate_plan_node",
        "generate_dev_node": "generate_dev_node",
        "topic_exists_node": "topic_exists_node",
        END: END,
    },
)

# 모든 노드가 끝나면 종료 (상태는 DB나 클라이언트에 반환)
workflow.add_edge("explore_node", END)
workflow.add_edge("gather_node", END)
workflow.add_edge("generate_plan_node", END)
workflow.add_edge("generate_dev_node", END)
workflow.add_edge("topic_exists_node", END)

ai_app = workflow.compile()
