import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.ai.graph.nodes import (
    _extract_direct_fact_updates,
    explore_problem_node,
    gather_information_node,
)


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(f"{label}: expected truthy condition")


def make_state(user_message: str, *, phase: str = "GATHER", collected_data: dict | None = None) -> dict:
    return {
        "project_id": "1",
        "user_message": user_message,
        "action_type": "CHAT",
        "current_phase": phase,
        "turn_policy": "ANSWER_THEN_ASK",
        "recent_messages": [],
        "selected_message": None,
        "collected_data": collected_data or {},
        "is_sufficient": False,
        "ai_message": "",
        "next_phase": phase,
        "template_payload": None,
    }


def main() -> None:
    fill_state = make_state(
        "지금 미정인 항목이랑 제대로 정의되지 않은 부분까지 다 채워줘",
        phase="EXPLORE",
        collected_data={},
    )
    fill_result = explore_problem_node(fill_state)
    assert_equal(fill_result["collected_data"], {}, "case1 no mutation")
    assert_true(
        "제목" in fill_result["ai_message"] or "주제" in fill_result["ai_message"],
        "case1 deterministic missing prompt",
    )
    assert_true("플랫폼" not in fill_result["ai_message"], "case1 no invented domain")

    assert_equal(
        _extract_direct_fact_updates("제목은 공공화장실 실시간 혼잡 안내로 할래"),
        {"title": "공공화장실 실시간 혼잡 안내"},
        "case2 clean title extraction",
    )

    raw_summary_data = {
        "title": "공공화장실 실시간 혼잡 안내",
        "goal": "사용자가 혼잡한 공공화장실을 피해서 더 편하게 이용할 수 있도록 돕는다",
        "teamSize": 4,
        "roles": ["개발자", "기획자", "PM"],
    }
    summary_state = make_state("지금까지 모인 정보 요약해줘", collected_data=raw_summary_data)
    summary_result = gather_information_node(summary_state)
    assert_equal(summary_result["collected_data"], raw_summary_data, "case3 read-only shape")
    assert_true("팀 인원 4명" in summary_result["ai_message"], "case3 summary teamSize")
    assert_true("역할 개발자, 기획자, PM" in summary_result["ai_message"], "case3 summary roles")

    mixed_state = make_state(
        "팀원은 4명이고 역할은 개발자, 기획자, PM 정도로 할 것 같아. 다음엔 뭐 정해야 해?",
        collected_data={
            "title": "공공화장실 실시간 혼잡 안내",
            "goal": "사용자가 혼잡한 공공화장실을 피해서 더 편하게 이용할 수 있도록 돕는다",
        },
    )
    mixed_result = gather_information_node(mixed_state)
    assert_equal(mixed_result["collected_data"]["teamSize"], "4", "case4 teamSize merged")
    assert_equal(
        mixed_result["collected_data"]["roles"],
        "개발자, 기획자, PM",
        "case4 roles merged",
    )
    assert_true(
        ("마감" in mixed_result["ai_message"]) or ("산출물" in mixed_result["ai_message"]),
        "case4 next-step guidance",
    )

    correction_state = make_state(
        "아니 팀원은 4명이 아니라 5명이야",
        collected_data={
            "title": "공공화장실 실시간 혼잡 안내",
            "goal": "사용자가 혼잡한 공공화장실을 피해서 더 편하게 이용할 수 있도록 돕는다",
            "teamSize": "4",
            "roles": "개발자, 기획자, PM",
        },
    )
    correction_result = gather_information_node(correction_state)
    assert_equal(correction_result["collected_data"]["teamSize"], "5", "case5 overwrite teamSize")

    print("verify_chat_flow: OK")


if __name__ == "__main__":
    main()
