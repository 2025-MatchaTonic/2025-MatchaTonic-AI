from langgraph.graph import StateGraph, END
from app.ai.graph.state import AgentState
from app.ai.graph.nodes import process_chat_node

# 그래프 생성
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("chat_node", process_chat_node)

# 흐름 연결 (시작 -> chat_node -> 종료)
workflow.set_entry_point("chat_node")
workflow.add_edge("chat_node", END)

# 앱 컴파일 (API에서 가져다 쓸 객체)
ai_app = workflow.compile()
