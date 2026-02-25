# app/ai/graph/nodes.py
import json
from langchain_openai import ChatOpenAI
from app.ai.graph.state import AgentState
from app.core.config import settings

llm = ChatOpenAI(
    model="gpt-4o-mini", temperature=0.7, openai_api_key=settings.OPENAI_API_KEY
)


# ----------------------------------------------------
# 1. 아이디어가 없을 때 (NO 선택) : 탐색 노드
# ----------------------------------------------------
def explore_problem_node(state: AgentState):
    prompt = f"""
    당신은 친절하고 센스 있는 브레인스토밍 파트너입니다. 
    사용자가 아직 프로젝트 주제가 없거나, 불편함을 탐색하는 중입니다.

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
# 2. 아이디어가 있을 때 (YES 선택) : 정보 수집 노드 (자연스러운 HMW)
# ----------------------------------------------------
def gather_information_node(state: AgentState):
    # 시스템 몰래 데이터를 평가하는 프롬프트 (JSON 반환 강제)
    eval_prompt = f"""
    당신은 사용자와 대화하며 5가지 핵심 정보(주제, 해결방안, 요구사항, 기대효과, 산출물)를 모으는 PM입니다.
    딱딱한 학술적 HMW가 아니라, "그럼 우리가 어떻게 하면 ~할 수 있을까요?" 처럼 자연스럽고 친근하게 질문하세요.
    한 번에 너무 많은 걸 묻지 말고, 비어있는 정보를 하나씩 유도하세요.
    
    [현재까지 모인 정보]
    {json.dumps(state.get('collected_data', {}), ensure_ascii=False)}
    
    [사용자 대답]
    {state['user_message']}
    
    [출력 형식 강제 (반드시 JSON 포맷으로 응답하세요)]
    {{
        "ai_message": "사용자에게 할 자연스러운 챗봇 응답 텍스트",
        "updated_data": {{ "topic": "...", "solution": "...", "requirements": "...", "impact": "...", "deliverables": "..." }},
        "is_sufficient": false // 5개가 다 모였고 구체적이면 true로 변경
    }}
    """

    # JSON 형태로만 응답하도록 강제 (GPT-4o-mini JSON mode)
    response = llm.invoke(eval_prompt, response_format={"type": "json_object"})
    result = json.loads(response.content)

    ai_msg = result["ai_message"]
    is_sufficient = result["is_sufficient"]

    # 데이터가 다 모였다면 템플릿 생성 안내 멘트 추가
    if is_sufficient:
        ai_msg += "\n\n🎉 완벽해요! 프로젝트에 필요한 핵심 정보가 모두 모였습니다. 이제 기획용 템플릿을 만들까요, 개발용 템플릿을 만들까요?"
        next_phase = "READY"
    else:
        next_phase = "GATHER"

    return {
        "ai_message": ai_msg,
        "collected_data": result.get("updated_data", state.get("collected_data")),
        "is_sufficient": is_sufficient,
        "next_phase": next_phase,
    }


# ----------------------------------------------------
# 3. 템플릿 생성 노드
# ----------------------------------------------------
def generate_template_node(state: AgentState):
    template_type = "기획자용" if state["action_type"] == "BTN_PLAN" else "개발자용"

    prompt = f"""
    아래 수집된 데이터를 바탕으로 {template_type} 프로젝트 명세서 템플릿을 마크다운으로 깔끔하게 작성해주세요.
    [수집된 데이터]: {json.dumps(state['collected_data'], ensure_ascii=False)}
    """
    response = llm.invoke(prompt)
    return {"ai_message": response.content, "next_phase": "DONE"}


# ----------------------------------------------------
# 4. @mates 멘션 호출 노드
# ----------------------------------------------------
def mates_helper_node(state: AgentState):
    response = llm.invoke(
        f"사용자가 긴급하게 AI 헬퍼(@mates)를 호출했습니다. 유쾌하고 든든하게 대답하며 무엇을 도와줄지 물어보세요. 사용자 메시지: {state['user_message']}"
    )
    return {"ai_message": response.content, "next_phase": state["current_phase"]}
