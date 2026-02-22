from langchain_openai import ChatOpenAI
from app.ai.graph.state import AgentState
from app.rag.retriever import get_retriever
from app.ai.prompts.validation import PM_SYSTEM_PROMPT
from app.core.config import settings

# LLM 초기화
llm = ChatOpenAI(
    model="gpt-4o-mini", temperature=0.7, openai_api_key=settings.OPENAI_API_KEY
)


def process_chat_node(state: AgentState):
    current_step = state["current_step"]
    user_msg = state["user_message"]

    # 1. RAG 지식 검색
    retriever = get_retriever(current_phase=str(current_step))
    docs = retriever.invoke(user_msg)
    rag_context = "\n".join([doc.page_content for doc in docs])

    # 2. 프롬프트 포매팅
    prompt = PM_SYSTEM_PROMPT.format(
        step=current_step, context=rag_context, user_input=user_msg
    )

    # 3. AI 답변 생성
    response = llm.invoke(prompt)

    # 4. 상태 업데이트 후 반환
    return {
        "ai_message": response.content,
        "next_step": current_step,  # 임시로 단계 유지 (추후 로직 추가)
    }
