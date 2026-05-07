from langgraph.graph import StateGraph, END
from app.ai.graph.collected_data import derive_phase_from_collected_data
from app.ai.graph.state import AgentState
from app.ai.graph.nodes import (
    explore_problem_node,
    gather_information_node,
    topic_exists_node,
)
from app.api.endpoints.template import generate_dev_template, generate_plan_template


def route_logic(state: AgentState):
    action = state["action_type"]
    phase = derive_phase_from_collected_data(
        state.get("collected_data") or {},
        current_phase=state["current_phase"],
    )

    if action == "BTN_NO":
        return "explore_node"
    if action in ["BTN_YES", "BTN_GO_DEF"]:
        return "topic_exists_node"
    if action == "BTN_PLAN":
        return "generate_plan_node"
    if action == "BTN_DEV":
        return "generate_dev_node"

    if phase in {"GATHER", "READY", "PROBLEM_DEFINE"}:
        return "gather_node"
    if phase == "TOPIC_SET":
        return "topic_exists_node"
    return "explore_node"


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

workflow.add_edge("explore_node", END)
workflow.add_edge("gather_node", END)
workflow.add_edge("generate_plan_node", END)
workflow.add_edge("generate_dev_node", END)
workflow.add_edge("topic_exists_node", END)

ai_app = workflow.compile()
