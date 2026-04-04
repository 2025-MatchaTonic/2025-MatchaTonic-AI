import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.endpoints.chat import AIChatRequest, _derive_turn_policy
from app.ai.graph.collected_data import merge_collected_data
from app.ai.graph.nodes import (
    _extract_direct_fact_updates,
    _extract_title_updates_for_topic_set,
    gather_information_node,
    topic_exists_node,
)
from app.core.request_normalization import normalize_collected_data


def _make_state(*, message: str, collected_data: dict | None = None) -> dict:
    return {
        "project_id": "1",
        "user_message": message,
        "action_type": "CHAT",
        "current_phase": "GATHER",
        "turn_policy": "ANSWER_THEN_ASK",
        "recent_messages": [],
        "selected_message": None,
        "collected_data": collected_data or {},
        "is_sufficient": False,
        "ai_message": "",
        "next_phase": "GATHER",
        "template_payload": None,
    }


def _make_topic_state(
    *,
    message: str,
    action_type: str = "CHAT",
    turn_policy: str = "ASK_ONLY",
    collected_data: dict | None = None,
) -> dict:
    return {
        "project_id": "1",
        "user_message": message,
        "action_type": action_type,
        "current_phase": "TOPIC_SET",
        "turn_policy": turn_policy,
        "recent_messages": [],
        "selected_message": None,
        "collected_data": collected_data or {},
        "is_sufficient": False,
        "ai_message": "",
        "next_phase": "TOPIC_SET",
        "template_payload": None,
    }


def test_request_normalization_returns_int_and_list_shapes():
    normalized = normalize_collected_data(
        {
            "teamSize": "4",
            "roles": "개발자, 기획자, PM",
        }
    )

    assert normalized["teamSize"] == 4
    assert normalized["roles"] == ["개발자", "기획자", "PM"]


def test_request_normalization_preserves_subject_separately_from_goal():
    normalized = normalize_collected_data(
        {
            "subject": "공공시설 이용",
            "goal": "예약 불편 해결",
        }
    )

    assert normalized["subject"] == "공공시설 이용"
    assert normalized["goal"] == "예약 불편 해결"


def test_request_normalization_drops_identifier_like_room_title():
    normalized = normalize_collected_data(
        {
            "title": "0982348ㅅ278045ㅍ",
            "goal": "예약 불편 해결",
        }
    )

    assert "title" not in normalized
    assert normalized["goal"] == "예약 불편 해결"


def test_ai_chat_request_discards_identifier_like_room_title():
    request = AIChatRequest(
        roomId=1030,
        content="@mates 공공시설 이용",
        actionType="CHAT",
        currentStatus="EXPLORE",
        collectedData={"title": "0982348ㅅ278045ㅍ"},
    )

    assert request.collectedData == {}


def test_merge_normalizes_roles_and_team_size():
    merged = merge_collected_data(
        {"title": "공공화장실 실시간 혼잡 안내"},
        {"teamSize": "4", "roles": "개발자, 기획자, PM"},
    )

    assert merged["teamSize"] == 4
    assert merged["roles"] == ["개발자", "기획자", "PM"]


def test_summary_request_preserves_committed_state_shape():
    current_data = {
        "title": "공공화장실 실시간 혼잡 안내",
        "goal": "사용자가 혼잡한 공공화장실을 피해서 더 편하게 이용할 수 있도록 돕는다",
        "teamSize": 4,
        "roles": ["개발자", "기획자", "PM"],
    }

    result = gather_information_node(
        _make_state(message="지금까지 모인 정보 요약해줘", collected_data=current_data)
    )

    assert result["collected_data"] == current_data
    assert isinstance(result["collected_data"]["teamSize"], int)
    assert isinstance(result["collected_data"]["roles"], list)


def test_mixed_next_step_fact_update_keeps_normalized_types():
    result = gather_information_node(
        _make_state(
            message="다음엔 팀원은 4명이고 역할은 개발자, 기획자, PM으로 할게",
            collected_data={"title": "공공화장실 실시간 혼잡 안내"},
        )
    )

    assert result["collected_data"]["teamSize"] == 4
    assert result["collected_data"]["roles"] == ["개발자", "기획자", "PM"]


def test_correction_utterance_keeps_team_size_as_int():
    result = gather_information_node(
        _make_state(
            message="아니 4명이 아니라 5명이야",
            collected_data={
                "title": "공공화장실 실시간 혼잡 안내",
                "teamSize": 4,
                "roles": ["개발자", "기획자", "PM"],
            },
        )
    )

    assert result["collected_data"]["teamSize"] == 5
    assert isinstance(result["collected_data"]["teamSize"], int)
    assert result["collected_data"]["roles"] == ["개발자", "기획자", "PM"]


def test_explicit_role_counts_expand_deterministically():
    updates = _extract_direct_fact_updates("개발자 2명, 기획자 1명, PM 1명으로 하면 좋겠어")
    merged = merge_collected_data({"teamSize": 4}, updates)

    assert merged["roles"] == ["개발자 1", "개발자 2", "기획자", "PM"]


def test_role_count_mismatch_is_not_silently_truncated():
    result = gather_information_node(
        _make_state(
            message="개발자 3명, 기획자 1명, PM 1명",
            collected_data={"teamSize": 4},
        )
    )

    assert result["collected_data"]["teamSize"] == 4
    assert "roles" not in result["collected_data"]
    assert "역할 인원 합계가 5명" in result["ai_message"]


def test_topic_button_label_is_not_extracted_as_title():
    state = _make_topic_state(
        message="예, 프로젝트 주제가 있습니다",
        turn_policy="CAPTURE_TITLE",
    )

    assert _extract_title_updates_for_topic_set(state) == {}


def test_topic_candidate_is_stored_as_subject_first():
    state = _make_topic_state(
        message="공공시설 이용 관련",
        turn_policy="CAPTURE_TITLE",
    )

    assert _extract_title_updates_for_topic_set(state) == {"subject": "공공시설 이용 관련"}


def test_help_request_is_not_extracted_as_title():
    state = _make_topic_state(
        message="어떤걸 하고 싶은지 잘 모르겠어 도와줘",
        turn_policy="CAPTURE_TITLE",
    )

    assert _extract_title_updates_for_topic_set(state) == {}


def test_topic_candidate_typo_is_corrected_with_llm():
    state = _make_topic_state(message="공공시설 ㅇ용", turn_policy="CAPTURE_TITLE")

    with patch("app.ai.graph.nodes.settings.OPENAI_API_KEY", "test-key"), patch(
        "app.ai.graph.nodes._invoke_llm",
        return_value=SimpleNamespace(content='{"normalized":"공공시설 이용"}'),
    ):
        assert _extract_title_updates_for_topic_set(state) == {"subject": "공공시설 이용"}


def test_topic_presence_button_message_is_not_extracted_as_subject():
    assert _extract_direct_fact_updates("예, 프로젝트 주제가 있습니다") == {}


def test_direct_subject_typo_is_corrected_with_llm():
    with patch("app.ai.graph.nodes.settings.OPENAI_API_KEY", "test-key"), patch(
        "app.ai.graph.nodes._invoke_llm",
        return_value=SimpleNamespace(content='{"normalized":"공공시설 이용"}'),
    ):
        assert _extract_direct_fact_updates("주제는 공공시설 ㅇ용") == {"subject": "공공시설 이용"}


def test_direct_subject_with_guidance_tail_keeps_subject_only():
    with patch("app.ai.graph.nodes.settings.OPENAI_API_KEY", "test-key"), patch(
        "app.ai.graph.nodes._invoke_llm",
        return_value=SimpleNamespace(content='{"normalized":"공공시설 예약 효율화"}'),
    ):
        assert _extract_direct_fact_updates(
            "주제는 공공시설 예약 효율화이고 아직 어떤 문제를 풀어야할지 모르겠어"
        ) == {"subject": "공공시설 예약 효율화"}


def test_topic_exists_node_reprompts_after_yes_button_label_chat_message():
    result = topic_exists_node(
        _make_topic_state(
            message="예, 프로젝트 주제가 있습니다",
            turn_policy="ASK_ONLY",
        )
    )

    assert result["collected_data"] == {}
    assert result["next_phase"] == "TOPIC_SET"
    assert "한두 줄로 보내주세요" in result["ai_message"]


def test_non_committal_title_update_does_not_overwrite_existing_title():
    merged = merge_collected_data(
        {"title": "공공시설 이용 관련"},
        {"title": "아니 도와줘"},
    )

    assert merged["title"] == "공공시설 이용 관련"


def test_help_request_with_existing_topic_returns_refinement_options():
    result = gather_information_node(
        _make_state(
            message="어떤걸 하고 싶은지 잘 모르겠어 도와줘",
            collected_data={"subject": "공공시설 이용 관련"},
        )
    )

    assert result["collected_data"] == {"subject": "공공시설 이용 관련"}
    assert result["next_phase"] == "GATHER"
    assert "같이 좁혀볼게요" in result["ai_message"]
    assert "혼잡도 확인" in result["ai_message"]
    assert "예약/대기 관리" in result["ai_message"]


def test_topic_exists_node_commits_subject_and_returns_refinement_for_mixed_message():
    with patch("app.ai.graph.nodes.settings.OPENAI_API_KEY", "test-key"), patch(
        "app.ai.graph.nodes._invoke_llm",
        return_value=SimpleNamespace(content='{"normalized":"공공시설 예약 효율화"}'),
    ):
        result = topic_exists_node(
            _make_topic_state(
                message="주제는 공공시설 예약 효율화이고 아직 어떤 문제를 풀어야할지 모르겠어",
                turn_policy="CAPTURE_TITLE",
            )
        )

    assert result["collected_data"] == {"subject": "공공시설 예약 효율화"}
    assert result["next_phase"] == "TOPIC_SET"
    assert "공공시설 예약 효율화" in result["ai_message"]
    assert "같이 좁혀볼게요" in result["ai_message"]
    assert "예약/대기 관리" in result["ai_message"]


def test_chat_turn_policy_treats_topic_presence_button_label_as_ask_only():
    request = AIChatRequest(
        roomId=1,
        content="예, 프로젝트 주제가 있습니다",
        actionType="CHAT",
        currentStatus="TOPIC_SET",
        collectedData={},
    )

    assert _derive_turn_policy(request) == "ASK_ONLY"
