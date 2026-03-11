# app/ai/graph/workflow.py
from langgraph.graph import StateGraph, END
from app.ai.graph.state import AgentState
from app.ai.graph.nodes import (
    explore_problem_node,
    gather_information_node,
    generate_template_node,
    mates_helper_node,
    topic_exists_node,
)


def route_logic(state: AgentState):
    action = state["action_type"]
    msg = state["user_message"]
    phase = state["current_phase"]

    # 0. 특수 호출 (@mates)
    if "@mates" in msg:
        return "mates_node"

    # 1. 초기 "프로젝트 주제가 있나요?" 응답 처리
    if action == "BTN_NO":
        return "explore_node"
    elif action in ["BTN_YES", "BTN_GO_DEF"]:
        return "topic_exists_node"

    # 2. 명시적 AI 기능 호출
    if action in ["BTN_PLAN", "BTN_DEV"]:
        return "generate_node"

    # 3. 일반 채팅 흐름
    if action == "CHAT":
        if phase == "EXPLORE":
            return "explore_node"
        if phase in ["TOPIC_SET", "GATHER", "READY"]:
            return "gather_node"

    return END


workflow = StateGraph(AgentState)

workflow.add_node("explore_node", explore_problem_node)
workflow.add_node("gather_node", gather_information_node)
workflow.add_node("generate_node", generate_template_node)
workflow.add_node("mates_node", mates_helper_node)
workflow.add_node("topic_exists_node", topic_exists_node)

workflow.set_conditional_entry_point(
    route_logic,
    {
        "explore_node": "explore_node",
        "gather_node": "gather_node",
        "generate_node": "generate_node",
        "mates_node": "mates_node",
        "topic_exists_node": "topic_exists_node",
        END: END,
    },
)

# 모든 노드가 끝나면 종료 (상태는 DB나 클라이언트에 반환)
workflow.add_edge("explore_node", END)
workflow.add_edge("gather_node", END)
workflow.add_edge("generate_node", END)
workflow.add_edge("mates_node", END)
workflow.add_edge("topic_exists_node", END)

ai_app = workflow.compile()
