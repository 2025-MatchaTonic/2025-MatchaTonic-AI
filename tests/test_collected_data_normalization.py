import sys
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.endpoints.chat import (
    AIChatRequest,
    _build_suggested_questions,
    _derive_turn_policy,
    process_chat,
)
from app.ai.graph.collected_data import (
    apply_collected_data_updates,
    build_approved_collected_data_snapshot,
    build_public_update_snapshot,
    choose_next_question_field,
    build_collected_data_json_example,
    classify_role_team_size_conflict,
    detect_meta_or_request_like,
    derive_phase_from_collected_data,
    evaluate_candidate_update,
    has_problem_definition_context,
    merge_collected_data,
    normalize_roles,
    subject_needs_problem_definition,
)
from app.api.endpoints.template import _build_template_state
from app.ai.graph.conversation_signals import (
    SIGNAL_CLASSIFICATION_CACHE,
    classify_signal,
)
from app.ai.graph.nodes import (
    _build_contextual_slot_question,
    _build_gather_focus_instruction,
    _coerce_gather_llm_result,
    _extract_direct_fact_updates,
    _extract_problem_area_candidate,
    _extract_target_facility_candidate,
    _extract_title_updates_for_topic_set,
    _interpret_turn_type,
    gather_information_node,
    topic_exists_node,
)
from app.ai.graph.topic_presence import (
    TOPIC_PRESENCE_CLASSIFICATION_CACHE,
    _is_topic_presence_negative_message,
    _matches_topic_presence_button_message,
)
from app.ai.graph.workflow import route_logic
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


def test_explicit_goal_statement_with_question_mark_is_still_extracted():
    updates = _extract_direct_fact_updates(
        "@mates 목표는 사용자가 예약 시간을 더 빨리 확인하게 하는 거야?"
    )

    assert updates["goal"] == "사용자가 예약 시간을 더 빨리 확인하게 하는 거야"


def test_topic_set_accepts_explicit_subject_change_without_freeform_block():
    state = _make_topic_state(
        message="@mates 청소년 우울증 방지 앱으로 바꿀게",
        collected_data={"subject": "날씨 코디 추천 앱"},
    )

    assert _extract_title_updates_for_topic_set(state, state["collected_data"]) == {
        "subject": "청소년 우울증 방지 앱"
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


def test_subject_broadness_detection_and_phase_derivation():
    assert subject_needs_problem_definition("공공시설") is True
    assert subject_needs_problem_definition("공공시설 예약 효율화") is False
    assert derive_phase_from_collected_data({"subject": "공공시설"}) == "PROBLEM_DEFINE"
    assert derive_phase_from_collected_data({"subject": "공공시설 예약 효율화"}) == "GATHER"
    assert derive_phase_from_collected_data({"title": "promate"}) == "TOPIC_SET"
    assert (
        derive_phase_from_collected_data(
            {"title": "promate"},
            current_phase="TOPIC_SET",
        )
        == "TOPIC_SET"
    )
    assert (
        derive_phase_from_collected_data(
            {"title": "promate", "goal": "팀 협업 문제를 줄이는 서비스로 만든다"},
        )
        == "GATHER"
    )
    assert (
        derive_phase_from_collected_data({"subject": "공공시설"}, current_phase="GATHER")
        == "PROBLEM_DEFINE"
    )


def test_request_normalization_drops_identifier_like_room_title():
    normalized = normalize_collected_data(
        {
            "title": "0982348ㅅ278045ㅍ",
            "goal": "예약 불편 해결",
        }
    )

    assert "title" not in normalized
    assert normalized["goal"] == "예약 불편 해결"


def test_request_normalization_drops_room_title_like_metadata_label():
    normalized = normalize_collected_data(
        {
            "title": "캠퍼스1",
            "goal": "교내 이동 불편 해결",
        }
    )

    assert "title" not in normalized
    assert normalized["goal"] == "교내 이동 불편 해결"


def test_collected_data_json_example_is_partial_update_shape():
    example = build_collected_data_json_example()

    assert '"subject"' in example
    assert '"roles"' not in example
    assert '"teamSize"' not in example


def test_ai_chat_request_discards_identifier_like_room_title():
    request = AIChatRequest(
        roomId=1030,
        content="@mates 공공시설 이용",
        actionType="CHAT",
        currentStatus="EXPLORE",
        collectedData={"title": "0982348ㅅ278045ㅍ"},
    )

    assert request.collectedData == {}


def test_ai_chat_request_keeps_raw_collected_data_for_logging():
    request = AIChatRequest(
        roomId=1032,
        content="@mates 지금 확정된 사항들 알려줘",
        actionType="CHAT",
        currentStatus="TOPIC_SET",
        collectedData={
            "subject": "공공시설",
            "title": "공공시설 도서관 혼잡도 개선",
            "goal": "도서관 혼잡도 확인을 쉽게 한다",
        },
    )

    assert request.rawCollectedData == {
        "subject": "공공시설",
        "title": "공공시설 도서관 혼잡도 개선",
        "goal": "도서관 혼잡도 확인을 쉽게 한다",
    }
    assert request.collectedData == request.rawCollectedData


def test_ai_chat_request_does_not_promote_ai_confirmation_message_to_content():
    request = AIChatRequest(
        roomId=1033,
        content="",
        actionType="CHAT",
        currentStatus="GATHER",
        selectedMessage="좋아요. 목표는 '몰라'로 반영할게요.",
        collectedData={"subject": "공공시설 예약"},
    )

    assert request.content == ""
    assert request.selectedMessage == "좋아요. 목표는 '몰라'로 반영할게요."


def test_ai_chat_request_keeps_user_selected_fact_message_as_content():
    request = AIChatRequest(
        roomId=1034,
        content="",
        actionType="CHAT",
        currentStatus="GATHER",
        selectedMessage="목표는 공공시설 예약 과정을 더 빠르게 만드는 거예요",
        collectedData={"subject": "공공시설 예약"},
    )

    assert request.content == "목표는 공공시설 예약 과정을 더 빠르게 만드는 거예요"


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


def test_summary_request_detects_confirmed_items_phrase():
    current_data = {
        "subject": "공공시설",
        "goal": "도서관 혼잡도 확인을 쉽게 한다",
    }

    result = gather_information_node(
        _make_state(message="지금 확정된 사항들 알려줘", collected_data=current_data)
    )

    assert result["collected_data"] == current_data
    assert "공공시설" in result["ai_message"]
    assert "목표" in result["ai_message"]
    assert "같이 좁혀볼게요" not in result["ai_message"]


def test_interpret_turn_type_detects_goal_guidance_request():
    turn_type = _interpret_turn_type(
        _make_state(
            message="@mates 목표 아직 잘 모르겠어 세우는 거 도와줘",
            collected_data={"subject": "공공시설 예약 효율화"},
        ),
        {"subject": "공공시설 예약 효율화"},
    )

    assert turn_type == "request_goal_guidance"


def test_goal_guidance_returns_examples_instead_of_repeating_goal_question():
    with patch(
        "app.ai.graph.nodes._invoke_llm",
        return_value=SimpleNamespace(
            content='{"slot":"goal","question":"이 주제에서 가장 만들고 싶은 변화는 무엇인가요?","options":["예약 가능한 시간과 절차를 빠르게 확인하게 한다","중복 예약과 대기 불편을 줄인다","모바일에서 예약 변경까지 간편하게 처리하게 한다"],"fallback_question":"이 프로젝트 목표를 한 줄로 말해 주세요.","generation_reason":"subject-grounded"}'
        ),
    ):
        result = gather_information_node(
            _make_state(
                message="@mates 목표 아직 잘 모르겠어 세우는 거 도와줘",
                collected_data={"subject": "공공시설 예약 효율화"},
            )
        )

    assert result["collected_data"] == {"subject": "공공시설 예약 효율화"}
    assert "목표는 이렇게 잡아볼 수 있어요" in result["ai_message"]
    assert "1." in result["ai_message"]
    assert "2." in result["ai_message"]
    assert "번호 하나를 고르" in result["ai_message"]
    assert result["ai_message"] != "이 프로젝트로 팀이 최종적으로 만들고 싶은 결과를 한 줄로 말하면 무엇인가요?"


def test_goal_guidance_fallback_returns_options_for_recommendation_request():
    with patch("app.ai.graph.nodes._invoke_llm", return_value=None):
        result = gather_information_node(
            _make_state(
                message="@mates 목표 못 정하겠어 추천해줘",
                collected_data={"subject": "대학생 일정 관리 앱"},
            )
        )

    assert result["collected_data"] == {"subject": "대학생 일정 관리 앱"}
    assert "목표는 이렇게 잡아볼 수 있어요" in result["ai_message"]
    assert "1." in result["ai_message"]
    assert "2." in result["ai_message"]
    assert "3." in result["ai_message"]
    assert "어떤 문제를 해결" not in result["ai_message"]


def test_goal_guidance_context_turns_problem_material_into_goal_options():
    guidance_message = (
        "목표는 보통 '누가 겪는 어떤 문제를 어떤 상태로 바꿀 것인가'로 잡으면 좋아요. "
        "최근에 불편했던 상황이나 만들고 싶은 방향을 한 줄로 말해 주세요."
    )

    result = topic_exists_node(
        {
            **_make_state(
                message="@mates 회의 일정 잡는게 너무 어려웠어",
            ),
            "current_phase": "TOPIC_SET",
            "recent_messages": [guidance_message],
        }
    )

    assert result["collected_data"] == {}
    assert "주제 방향" not in result["ai_message"]
    assert "어떤 문제를 해결" not in result["ai_message"]
    assert "목표는 이렇게 잡아볼 수 있어요" in result["ai_message"]
    assert result["next_question_field"] == "goal"


def test_goal_guidance_number_choice_is_committed_as_goal():
    guidance_message = (
        "좋아요. '공공시설 예약' 기준이면 목표는 이렇게 잡을 수 있어요.\n"
        "1. 사용자가 예약 가능 시간과 절차를 더 빠르게 확인하고 신청할 수 있게 한다\n"
        "2. 중복 예약과 대기 불편을 줄여 공공시설 예약 과정을 더 효율적으로 만든다\n"
        "3. 전화나 방문 없이 모바일에서 예약과 변경을 간편하게 처리할 수 있게 한다\n"
        "가장 가까운 번호 하나를 고르거나, 원하는 방향으로 한 줄 수정해 주세요."
    )

    result = gather_information_node(
        {
            **_make_state(
                message="@mates 1. 사용자가 예약 가능 시간과 절차를 더 빠르게 확인하고 신청할 수 있게 한다 이걸 목표로 하자",
                collected_data={"subject": "공공시설 예약"},
            ),
            "recent_messages": [guidance_message],
        }
    )

    assert result["collected_data"]["goal"] == "사용자가 예약 가능 시간과 절차를 더 빠르게 확인하고 신청할 수 있게 한다"
    assert "목표는 '사용자가 예약 가능 시간과 절차를 더 빠르게 확인하고 신청할 수 있게 한다'" in result["ai_message"]


def test_broad_subject_goal_commit_does_not_repeat_problem_definition_prompt():
    result = gather_information_node(
        _make_state(
            message="@mates 목표는 청소년 우울증 감소야",
            collected_data={
                "subject": "청소년 우울증 예방",
                "title": "청소년 우울증 예방",
            },
        )
    )

    assert result["collected_data"]["goal"] == "청소년 우울증 감소"
    assert "목표는 '청소년 우울증 감소'" in result["ai_message"]
    assert "어떤 문제를 해결" not in result["ai_message"]
    assert "사용자가 가장 먼저 겪는 문제" not in result["ai_message"]


def test_prompted_target_user_extracts_short_candidate_from_mixed_sentence():
    result = gather_information_node(
        {
            **_make_state(
                message="@mates 문제를 가장 먼저 겪는 대상은 청소년이고 해당 문제를 줄이기 위해 오락 앱을 만들고 싶어",
                collected_data={
                    "subject": "청소년 우울증 예방",
                    "title": "청소년 우울증 예방",
                },
            ),
            "current_slot": "targetUser",
        }
    )

    assert result["collected_data"]["targetUser"] == "청소년"


def test_mixed_next_step_fact_update_keeps_normalized_types():
    with patch(
        "app.ai.graph.nodes._extract_direct_fact_updates",
        return_value={"teamSize": 4, "roles": "개발자, 기획자, PM"},
    ), patch(
        "app.ai.graph.nodes._interpret_turn_type",
        return_value="provide_fact",
    ):
        result = gather_information_node(
            _make_state(
                message="다음엔 팀원은 4명이고 역할은 개발자, 기획자, PM으로 할게",
                collected_data={"title": "공공화장실 실시간 혼잡 안내"},
            )
        )

    assert result["collected_data"]["teamSize"] == 4
    assert result["collected_data"]["roles"] == ["개발자", "기획자", "PM"]


def test_correction_utterance_keeps_team_size_as_int():
    with patch(
        "app.ai.graph.nodes._extract_direct_fact_updates",
        return_value={"teamSize": 5},
    ), patch(
        "app.ai.graph.nodes._interpret_turn_type",
        return_value="provide_fact",
    ):
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


def test_request_like_goal_value_is_not_persisted():
    assert _extract_direct_fact_updates("목표는 추천해달라고") == {}


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


def test_topic_presence_button_message_resets_stale_topic_state():
    result = topic_exists_node(
        _make_topic_state(
            message="예, 프로젝트 주제가 있습니다",
            turn_policy="ASK_ONLY",
            collected_data={"subject": "캠퍼스"},
        )
    )

    assert result["collected_data"] == {}
    assert result["next_phase"] == "TOPIC_SET"
    assert "한두 줄로 보내주세요" in result["ai_message"]


def test_turn_policy_treats_button_like_chat_as_ask_only_even_with_stale_topic():
    request = AIChatRequest(
        roomId=1035,
        content="@mates 예, 프로젝트 주제가 있습니다",
        actionType="CHAT",
        currentStatus="GATHER",
        collectedData={"subject": "캠퍼스"},
    )

    assert _derive_turn_policy(request) == "ASK_ONLY"


def test_llm_topic_presence_message_routes_chat_to_ask_only():
    TOPIC_PRESENCE_CLASSIFICATION_CACHE.clear()
    request = AIChatRequest(
        roomId=1036,
        content="@mates 아직 정한 건 없고 같이 정하고 싶어요",
        actionType="CHAT",
        currentStatus="GATHER",
        collectedData={"subject": "캠퍼스"},
    )

    with patch("app.ai.graph.topic_presence.settings.OPENAI_API_KEY", "test-key"), patch(
        "app.ai.graph.topic_presence._invoke_llm",
        return_value=SimpleNamespace(content='{"label":"no_topic","confidence":0.96}'),
    ):
        assert _derive_turn_policy(request) == "ASK_ONLY"


def test_llm_topic_presence_negative_message_resets_to_explore():
    TOPIC_PRESENCE_CLASSIFICATION_CACHE.clear()
    with patch("app.ai.graph.topic_presence.settings.OPENAI_API_KEY", "test-key"), patch(
        "app.ai.graph.topic_presence._invoke_llm",
        return_value=SimpleNamespace(content='{"label":"no_topic","confidence":0.97}'),
    ):
        assert _matches_topic_presence_button_message("아직 정한 건 없고 같이 정하고 싶어요") is True
        assert _is_topic_presence_negative_message("아직 정한 건 없고 같이 정하고 싶어요") is True
        result = topic_exists_node(
            _make_topic_state(
                message="아직 정한 건 없고 같이 정하고 싶어요",
                turn_policy="ASK_ONLY",
                collected_data={"subject": "캠퍼스"},
            )
        )

    assert result["collected_data"] == {}
    assert result["next_phase"] == "EXPLORE"
    assert "최근 일주일" in result["ai_message"]


def test_apply_collected_data_updates_blocks_request_like_goal_candidate():
    next_data, audit = apply_collected_data_updates(
        {"subject": "공공시설 혼잡도 안내 서비스"},
        {"goal": "추천해달라고"},
        turn_type="provide_fact",
        current_status="GATHER",
        user_message="추천해달라고",
    )

    assert next_data == {"subject": "공공시설 혼잡도 안내 서비스"}
    assert audit["approved"] == {}
    assert audit["rejected"]["goal"] == "추천해달라고"
    assert audit["rejected_reasons"]["goal"] == "goal_request_like_not_storable"


def test_apply_collected_data_updates_allows_prompted_team_size_answer():
    next_data, audit = apply_collected_data_updates(
        {"subject": "공공시설 혼잡도 안내 서비스"},
        {"teamSize": "4명이요"},
        turn_type="request_fill_missing",
        current_status="GATHER",
        recent_messages=["현재 이 프로젝트는 총 몇 명이 함께 진행하나요?"],
        user_message="4명이요",
    )

    assert next_data["teamSize"] == 4
    assert audit["approved"]["teamSize"] == 4
    assert audit["rejected"] == {}


def test_apply_collected_data_updates_requires_title_context_for_confirmation():
    next_data, audit = apply_collected_data_updates(
        {"subject": "공공시설 혼잡도 안내 서비스"},
        {"title": "혼잡도 나침반"},
        turn_type="provide_fact",
        current_status="TOPIC_SET",
        user_message="그 제목으로 할게",
    )

    assert "title" not in next_data
    assert audit["rejected_reasons"]["title"] == "title_requires_recent_context"


def test_apply_collected_data_updates_allows_explicit_goal_overwrite():
    next_data, audit = apply_collected_data_updates(
        {
            "subject": "공공시설 혼잡도 안내 서비스",
            "goal": "현재 혼잡도를 보여주는 서비스",
        },
        {"goal": "공공시설 혼잡도 예측으로 수정할게"},
        turn_type="provide_fact",
        current_status="GATHER",
        user_message="목표는 공공시설 혼잡도 예측으로 수정할게",
    )

    assert next_data["goal"] == "공공시설 혼잡도 예측으로 수정할게"
    assert audit["approved"]["goal"] == "공공시설 혼잡도 예측으로 수정할게"


def test_gather_node_blocks_title_confirmation_without_recent_context():
    with patch(
        "app.ai.graph.nodes._extract_direct_fact_updates",
        return_value={"title": "혼잡도 나침반"},
    ), patch(
        "app.ai.graph.nodes._interpret_turn_type",
        return_value="provide_fact",
    ):
        result = gather_information_node(
            {
                **_make_state(
                    message="그 제목으로 할게",
                    collected_data={"subject": "공공시설 혼잡도 안내 서비스"},
                ),
                "current_phase": "TOPIC_SET",
            }
        )

    assert "title" not in result["collected_data"]
    assert result["rejected_reasons"]["title"] == "title_requires_recent_context"
    assert "제목은 명시적으로 정하거나" in result["ai_message"]


def test_gather_node_allows_title_confirmation_with_recent_context():
    with patch(
        "app.ai.graph.nodes._extract_direct_fact_updates",
        return_value={"title": "혼잡도 나침반"},
    ), patch(
        "app.ai.graph.nodes._interpret_turn_type",
        return_value="provide_fact",
    ):
        result = gather_information_node(
            {
                **_make_state(
                    message="그 제목으로 할게",
                    collected_data={"subject": "공공시설 혼잡도 안내 서비스"},
                ),
                "current_phase": "TOPIC_SET",
                "recent_messages": ["좋아요. 프로젝트 제목 후보는 '혼잡도 나침반'이 어울려 보여요."],
            }
        )

    assert result["collected_data"]["title"] == "혼잡도 나침반"
    assert result["approved_updates"]["title"] == "혼잡도 나침반"


def test_gather_node_preserves_existing_goal_on_recommendation_request():
    with patch(
        "app.ai.graph.nodes._extract_direct_fact_updates",
        return_value={"goal": "추천해달라고"},
    ), patch(
        "app.ai.graph.nodes._interpret_turn_type",
        return_value="provide_fact",
    ):
        result = gather_information_node(
            _make_state(
                message="추천 좀 해줘",
                collected_data={
                    "subject": "공공시설 혼잡도 안내 서비스",
                    "goal": "현재 혼잡도를 빠르게 확인하게 한다",
                },
            )
        )

    assert result["collected_data"]["goal"] == "현재 혼잡도를 빠르게 확인하게 한다"
    assert result["rejected_reasons"]["goal"] == "goal_request_like_not_storable"
    assert "바로 저장하진 않았어요" in result["ai_message"]


def test_build_template_state_uses_approved_snapshot_only():
    request = SimpleNamespace(
        roomId=1,
        templateType="plan",
        currentStatus="READY",
        content="",
        collectedData={
            "subject": "공공시설 혼잡도 안내 서비스",
            "goal": "정리해줘",
            "title": "098234878045",
            "teamSize": 4,
        },
        recentMessages=[],
        selectedMessage=None,
    )

    state = _build_template_state(request)

    assert state["collected_data"] == {
        "subject": "공공시설 혼잡도 안내 서비스",
        "teamSize": 4,
    }


def test_build_approved_collected_data_snapshot_drops_request_like_goal():
    snapshot = build_approved_collected_data_snapshot(
        {
            "subject": "공공시설 혼잡도 안내 서비스",
            "goal": "정리해줘",
            "dueDate": "2026년 말",
        }
    )

    assert snapshot == {
        "subject": "공공시설 혼잡도 안내 서비스",
        "dueDate": "2026년 말",
    }


def test_build_public_update_snapshot_preserves_public_auxiliary_fields():
    snapshot = build_public_update_snapshot(
        {
            "subject": "weather outfit assistant",
            "problemArea": "personalized guidance",
            "targetFacility": "library",
            "goal": "help users choose clothes faster",
        }
    )

    assert snapshot == {
        "subject": "weather outfit assistant",
        "problemArea": "personalized guidance",
        "targetFacility": "library",
        "goal": "help users choose clothes faster",
    }


def test_due_date_statement_with_suffix_is_extracted_directly():
    assert _extract_direct_fact_updates("@mates 2026년 6월이 마감일") == {
        "dueDate": "2026년 6월"
    }


def test_deliverable_shape_statement_is_not_problem_area():
    state = _make_state(
        message="@mates 웹서비스형태",
        collected_data={"subject": "공공시설 효율화 앱"},
    )

    assert _extract_problem_area_candidate(state, state["collected_data"]) == ""
    assert _extract_direct_fact_updates("@mates 웹서비스형태") == {"deliverables": "웹 서비스"}


def test_priority_answer_is_not_problem_area():
    state = _make_state(
        message="@mates 시민이 우선이야",
        collected_data={"subject": "공공시설 효율화 앱"},
    )

    assert _extract_problem_area_candidate(state, state["collected_data"]) == ""


def test_advice_request_about_deliverable_shape_stays_general():
    state = _make_state(
        message="@mates 웹 서비스 형태고 구체적인 산출물 형태 알려줘",
        collected_data={
            "subject": "공공시설 효율화 앱",
            "goal": "불편과 정보 부족 문제 줄이기",
            "dueDate": "2026년 6월",
        },
    )

    assert _interpret_turn_type(state, state["collected_data"]) == "general"


def test_short_facility_answer_is_captured_from_context():
    state = _make_state(
        message="@mates 도서관",
        collected_data={"subject": "공공시설 효율화 앱"},
    )

    assert _extract_target_facility_candidate(state, state["collected_data"]) == "도서관"


def test_llm_due_date_type_is_merged_into_existing_due_date():
    with patch("app.ai.graph.nodes._fetch_rag_context", return_value=""), patch(
        "app.ai.graph.nodes._extract_direct_fact_updates",
        return_value={},
    ), patch(
        "app.ai.graph.nodes._interpret_turn_type",
        return_value="general",
    ), patch(
        "app.ai.graph.nodes._invoke_llm",
        return_value=SimpleNamespace(
            content=(
                '{"intent":"general","ai_message":"최종 제출 기준으로 이해했습니다.",'
                '"updated_data":{"dueDateType":"최종 제출"},"is_sufficient":false}'
            )
        ),
    ):
        result = gather_information_node(
            _make_state(
                message="@mates 최종 제출",
                collected_data={
                    "subject": "공공시설 효율화 앱",
                    "goal": "불편과 정보 부족 문제 줄이기",
                    "dueDate": "2026년 6월",
                },
            )
        )

    assert result["approved_updates"]["dueDate"] == "2026년 6월 (최종 제출)"
    assert result["collected_data"]["dueDate"] == "2026년 6월 (최종 제출)"


def test_process_chat_returns_approved_update_metadata():
    request = AIChatRequest(
        roomId=1,
        content="@mates 목표는 공공시설 이용 불편을 줄이는 거야",
        actionType="CHAT",
        currentStatus="GATHER",
        collectedData={"subject": "공공시설 효율화 앱"},
    )

    with patch(
        "app.api.endpoints.chat.ai_app.invoke",
        return_value={
            "ai_message": "좋아요. 목표를 정리할게요.",
            "collected_data": {
                "subject": "공공시설 효율화 앱",
                "goal": "공공시설 이용 불편을 줄인다",
            },
            "is_sufficient": False,
            "next_phase": "GATHER",
            "approved_updates": {"goal": "공공시설 이용 불편을 줄인다"},
            "rejected_updates": {},
            "rejected_reasons": {},
            "followup_fields": ["dueDate"],
            "template_payload": None,
        },
    ):
        response = asyncio.run(process_chat(request))

    assert response.collectedData["goal"] == "공공시설 이용 불편을 줄인다"
    assert response.suggestedQuestions[0] == "마감일이나 발표일은 언제인가요?"


def test_due_date_prompt_does_not_force_middle_or_final_milestones():
    question = _build_contextual_slot_question(
        {"subject": "대학생 일정 관리 앱", "goal": "과제 시험 마감 관리"},
        "dueDate",
    )
    instruction = _build_gather_focus_instruction("dueDate")

    assert "중간발표" not in question
    assert "최종제출" not in question
    assert "마감 기준" in question
    assert "정해진 날짜" in question
    assert "특정 마일스톤" in instruction


def test_process_chat_preserves_auxiliary_fields_in_public_response():
    request = AIChatRequest(
        roomId=11,
        content="@mates library congestion problem",
        actionType="CHAT",
        currentStatus="GATHER",
        collectedData={"subject": "public facility service"},
    )

    with patch(
        "app.api.endpoints.chat.ai_app.invoke",
        return_value={
            "ai_message": "ok",
            "collected_data": {
                "subject": "public facility service",
                "problemArea": "congestion visibility",
                "goal": "reduce wasted visits",
            },
            "is_sufficient": False,
            "next_phase": "GATHER",
            "approved_updates": {
                "problemArea": "congestion visibility",
                "goal": "reduce wasted visits",
            },
            "rejected_updates": {
                "targetFacility": "library",
            },
            "rejected_reasons": {
                "targetFacility": "turn_type_blocks_storage",
            },
            "followup_fields": ["dueDate"],
            "template_payload": None,
        },
    ):
        response = asyncio.run(process_chat(request))

    assert response.collectedData == {
        "subject": "public facility service",
        "title": "public facility service",
        "problemArea": "congestion visibility",
        "goal": "reduce wasted visits",
    }


def test_process_chat_prefers_node_next_question_field():
    request = AIChatRequest(
        roomId=2,
        content="@mates 잘 모르겠어",
        actionType="CHAT",
        currentStatus="GATHER",
        collectedData={"subject": "공공시설 예약"},
    )

    with patch(
        "app.api.endpoints.chat.ai_app.invoke",
        return_value={
            "ai_message": "이 프로젝트 목표를 한 줄로 말해 주세요. 예: 시민이 공공시설을 빠르게 예약할 수 있도록 돕는다",
            "collected_data": {"subject": "공공시설 예약"},
            "is_sufficient": False,
            "next_phase": "GATHER",
            "approved_updates": {},
            "rejected_updates": {},
            "rejected_reasons": {},
            "followup_fields": [],
            "next_question_field": "goal",
            "template_payload": None,
        },
    ):
        response = asyncio.run(process_chat(request))

    assert response.suggestedQuestions[0] == "이 프로젝트 목표를 한 줄로 말해 주세요."


def test_goal_statement_trims_trailing_goal_phrase():
    assert _extract_direct_fact_updates(
        "@mates 목표는 공공시설 이용 과정에서 반복되는 불편과 정보 부족 문제를 줄인다가 목표야"
    ) == {
        "goal": "공공시설 이용 과정에서 반복되는 불편과 정보 부족 문제를 줄인다"
    }


def test_deliverables_list_from_llm_is_persisted():
    with patch("app.ai.graph.nodes._fetch_rag_context", return_value=""), patch(
        "app.ai.graph.nodes._extract_direct_fact_updates",
        return_value={},
    ), patch(
        "app.ai.graph.nodes._interpret_turn_type",
        return_value="general",
    ), patch(
        "app.ai.graph.nodes._invoke_llm",
        return_value=SimpleNamespace(
            content=(
                '{"intent":"general","ai_message":"산출물 권장을 정리했습니다.","updated_data":'
                '{"deliverables":["작동하는 웹 앱","API 명세서","시연 영상"]},"is_sufficient":false}'
            )
        ),
    ):
        result = gather_information_node(
            _make_state(
                message="@mates 웹 서비스 형태고 구체적인 산출물 형태 알려줘",
                collected_data={
                    "subject": "공공시설 효율화 앱",
                    "goal": "불편과 정보 부족 문제 줄이기",
                    "dueDate": "2026년 6월",
                },
            )
        )

    assert result["approved_updates"]["deliverables"] == "작동하는 웹 앱; API 명세서; 시연 영상"
    assert result["collected_data"]["deliverables"] == "작동하는 웹 앱; API 명세서; 시연 영상"


def test_deliverable_recommendation_request_is_not_classified_as_summary():
    state = _make_state(
        message="@mates 캡스톤디자인 경진대회 출품할 예정이야 그에 맞게 추천하는 최종 산출물 정리해줘",
        collected_data={
            "subject": "공공시설 효율화 앱",
            "goal": "불편과 정보 부족 문제 줄이기",
            "dueDate": "2026년 6월",
        },
    )

    assert _interpret_turn_type(state, state["collected_data"]) == "general"


def test_explicit_deliverables_statement_preserves_full_phrase():
    assert _extract_direct_fact_updates(
        "@mates 최종 산출물은 웹 서비스 프로토타입과 발표 자료예요"
    ) == {
        "deliverables": "웹 서비스 프로토타입과 발표 자료"
    }


def test_problem_statement_is_classified_before_generic_advice():
    state = _make_state(
        message="@mates 대학생 팀플에서 역할이 안 나뉘어서 생기는 문제를 해결하고 싶어요",
        collected_data={"subject": "AI 기반 팀 프로젝트 협업 지원 서비스"},
    )

    assert _extract_problem_area_candidate(state, state["collected_data"]) == (
        "대학생 팀플에서 역할이 안 나뉘어서 생기는 문제"
    )
    assert _interpret_turn_type(state, state["collected_data"]) == "provide_problem_area"


def test_role_recommendation_request_stays_on_general_route():
    state = _make_state(
        message="@mates 너가 필요한 역할 정해줘",
        collected_data={
            "subject": "AI 기반 팀 프로젝트 협업 지원 서비스",
            "goal": "역할 분담과 일정 관리를 더 쉽게 만드는 것",
            "dueDate": "6월 말",
            "deliverables": "웹 서비스",
        },
    )
    state["recent_messages"] = [
        "좋아요. 산출물은 웹 서비스 정리할게요. 팀원 역할 분담은 어떻게 가져갈 생각인가요? 아직 미정이면 필요한 역할만 말해도 됩니다."
    ]

    assert _interpret_turn_type(state, state["collected_data"]) == "general"


def test_target_user_statement_is_extracted_as_structured_fact():
    updates = _extract_direct_fact_updates(
        "@mates 타겟 사용자는 초보 대학생 팀 프로젝트 팀이야"
    )

    assert updates["targetUser"] == "초보 대학생 팀 프로젝트 팀"


def test_primary_user_statement_is_not_misread_as_problem_area():
    state = _make_state(
        message="@mates 주 사용자는 고학년이고 전공은 컴퓨터공학과야 시험, 과제 일정 관리하는 앱 만들고 싶어",
        collected_data={"subject": "대학생 일정 관리 앱"},
    )
    state["recent_messages"] = [
        "주 사용자(예: 학년·전공)와 이 앱으로 해결하려는 핵심 문제를 한 문장으로 알려주세요."
    ]

    updates = _extract_direct_fact_updates(state["user_message"])

    assert updates["targetUser"] == "고학년 컴퓨터공학과"
    assert (
        _extract_problem_area_candidate(
            state,
            state["collected_data"],
            direct_updates=updates,
        )
        == ""
    )
    assert (
        _interpret_turn_type(
            state,
            state["collected_data"],
            direct_updates=updates,
        )
        == "provide_fact"
    )


def test_gather_commits_primary_user_without_echoing_whole_answer_as_problem():
    state = _make_state(
        message="@mates 주 사용자는 고학년이고 전공은 컴퓨터공학과야 시험, 과제 일정 관리하는 앱 만들고 싶어",
        collected_data={"subject": "대학생 일정 관리 앱"},
    )
    state["recent_messages"] = [
        "주 사용자(예: 학년·전공)와 이 앱으로 해결하려는 핵심 문제를 한 문장으로 알려주세요."
    ]

    result = gather_information_node(state)

    assert result["approved_updates"]["targetUser"] == "고학년 컴퓨터공학과"
    assert result["collected_data"]["targetUser"] == "고학년 컴퓨터공학과"
    assert "주 사용자는 '고학년 컴퓨터공학과'" in result["ai_message"]
    assert "주 사용자는 고학년이고 전공은 컴퓨터공학과야" not in result["ai_message"]


def test_normalize_collected_data_prunes_guidance_goal_and_primary_user_problem_area():
    normalized = normalize_collected_data(
        {
            "subject": "대학생 일정 관리 앱",
            "title": "대학생 일정 관리 앱",
            "goal": "이렇게 잡아볼 수 있어요",
            "problemArea": "주 사용자는 고학년이고 전공은 컴퓨터공학과야 시험, 과제 일정 관리하는 앱 만들고 싶어",
        }
    )

    assert normalized == {
        "subject": "대학생 일정 관리 앱",
        "title": "대학생 일정 관리 앱",
    }


def test_goal_choice_after_polluted_session_commits_user_goal():
    state = _make_state(
        message="@mates 개인화된 알림·우선순위로 시험·과제 제출 누락 최소화",
        collected_data={
            "subject": "대학생 일정 관리 앱",
            "title": "대학생 일정 관리 앱",
            "goal": "이렇게 잡아볼 수 있어요",
            "problemArea": "주 사용자는 고학년이고 전공은 컴퓨터공학과야 시험, 과제 일정 관리하는 앱 만들고 싶어",
        },
    )
    state["recent_messages"] = [
        "앱에서 가장 우선할 단기적 목표는 무엇인가요? 아래 중 선택하거나 직접 적어주세요.\n"
        "1. 마감 누락 방지 중심: 개인화된 알림·우선순위로 시험·과제 제출 누락 최소화\n"
        "가장 가까운 번호 하나를 고르거나, 원하는 방향으로 한 줄 수정해서 말해 주세요."
    ]

    result = gather_information_node(state)

    assert result["approved_updates"]["goal"] == "개인화된 알림·우선순위로 시험·과제 제출 누락 최소화"
    assert result["collected_data"]["goal"] == "개인화된 알림·우선순위로 시험·과제 제출 누락 최소화"
    assert "problemArea" not in result["collected_data"]
    assert "이렇게 잡아볼 수 있어요" not in result["collected_data"].values()


def test_target_user_plain_reply_commits_when_recent_prompt_targets_it():
    state = _make_state(
        message="@mates 초보 대학생 팀 프로젝트 팀",
        collected_data={
            "subject": "AI 기반 팀 프로젝트 협업 지원 서비스",
            "goal": "역할 분담과 일정 관리를 더 쉽게 만드는 것",
        },
    )
    state["recent_messages"] = [
        "현재 목표와 가장 잘 맞는 대상은 초보 대학생 팀 프로젝트 팀입니다. 이 타겟으로 확정하고, 다음으로 주요 문제 상황을 더 구체화할까요?"
    ]

    result = gather_information_node(state)

    assert result["approved_updates"]["targetUser"] == "초보 대학생 팀 프로젝트 팀"
    assert result["collected_data"]["targetUser"] == "초보 대학생 팀 프로젝트 팀"


def test_target_user_is_optional_for_ready_phase():
    assert (
        derive_phase_from_collected_data(
            {
                "subject": "AI 기반 팀 프로젝트 협업 지원 서비스",
                "goal": "역할 분담과 일정 관리를 더 쉽게 만드는 것",
                "teamSize": 4,
                "roles": ["PM", "프론트엔드", "백엔드", "디자인"],
                "dueDate": "6월 말",
                "deliverables": "웹 서비스 프로토타입과 발표 자료",
            }
        )
        == "READY"
    )


def test_mvp_priority_request_stays_on_general_route():
    state = _make_state(
        message="@mates MVP 기준으로 우선순위 정리해줘",
        collected_data={
            "subject": "AI 기반 팀 프로젝트 협업 지원 서비스",
            "goal": "역할 분담과 일정 관리를 더 쉽게 만드는 것",
        },
    )

    assert _interpret_turn_type(state, state["collected_data"]) == "general"


def test_next_actions_request_stays_on_general_route():
    state = _make_state(
        message="@mates 다음 할 일 정리해줘",
        collected_data={
            "subject": "AI 기반 팀 프로젝트 협업 지원 서비스",
            "goal": "역할 분담과 일정 관리를 더 쉽게 만드는 것",
            "deliverables": "웹 서비스 프로토타입",
        },
    )

    assert _interpret_turn_type(state, state["collected_data"]) == "general"


def test_conversation_signal_classifier_uses_llm_for_non_pattern_message():
    SIGNAL_CLASSIFICATION_CACHE.clear()
    with patch(
        "app.ai.graph.conversation_signals._should_try_llm_signal_classification",
        return_value=True,
    ), patch(
        "app.ai.graph.conversation_signals._invoke_llm",
        return_value=SimpleNamespace(
            content='{"label":"help_request","confidence":0.93}'
        ),
    ) as invoke_mock:
        assert classify_signal("이 상황에서 어느 쪽부터 잡는 게 맞을까") == "help_request"

    assert invoke_mock.call_count == 1


def test_roles_statement_preserves_parenthetical_labels():
    normalized = normalize_collected_data(
        {
            "roles": [
                "PM(프로젝트 총괄)",
                "프론트엔드 개발자(화면 구현)",
                "백엔드 개발자(서버·API)",
                "디자이너(UX/UI)",
            ]
        }
    )

    assert normalized["roles"] == [
        "PM(프로젝트 총괄)",
        "프론트엔드 개발자(화면 구현)",
        "백엔드 개발자(서버·API)",
        "디자이너(UX/UI)",
    ]


def test_topic_exists_node_returns_subject_approval_metadata():
    result = topic_exists_node(
        _make_topic_state(
            message="@mates 주제는 공공시설 효율화 앱이야",
            collected_data={},
        )
    )

    assert result["approved_updates"] == {"subject": "공공시설 효율화 앱"}
    assert result["rejected_updates"] == {}


def test_ready_title_confirmation_commits_last_suggested_title():
    state = _make_state(
        message="@mates 그 제목으로해줘",
        collected_data={
            "subject": "AI 기반 팀 프로젝트 협업 지원 서비스",
            "goal": "역할 분담과 일정 관리를 더 쉽게 만드는 것",
            "dueDate": "6월 말",
            "deliverables": "웹 서비스",
            "teamSize": 4,
            "roles": [
                "PM(프로젝트 총괄)",
                "프론트엔드 개발자(화면 구현)",
                "백엔드 개발자(서버·API)",
                "디자이너(UX/UI)",
            ],
        },
    )
    state["current_phase"] = "READY"
    state["next_phase"] = "READY"
    state["is_sufficient"] = True
    state["recent_messages"] = [
        "제목 제안: 'AI 팀코디: 역할·일정 자동관리'입니다. 이 제목으로 할까요?"
    ]

    with patch("app.ai.graph.nodes._fetch_rag_context", return_value=""):
        result = gather_information_node(state)

    assert result["approved_updates"] == {"title": "AI 팀코디: 역할·일정 자동관리"}
    assert result["collected_data"]["title"] == "AI 팀코디: 역할·일정 자동관리"


def test_summary_response_does_not_expose_room_title_metadata():
    result = gather_information_node(
        _make_state(
            message="@mates 지금까지 정리된 내용 요약해줘",
            collected_data={
                "subject": "공공시설 효율화 앱",
                "title": "promate4",
                "goal": "불편과 정보 부족 문제 줄이기",
                "dueDate": "2026년 6월",
            },
        )
    )

    assert "promate4" not in result["ai_message"]


def test_topic_exists_node_accepts_reverse_topic_phrasing():
    result = topic_exists_node(
        _make_topic_state(
            message="@mates 공공시설 관련 문제를 해결하는걸 주제로 생각하고 있어",
            turn_policy="CAPTURE_TITLE",
            collected_data={"title": "promate 2"},
        )
    )

    assert result["collected_data"]["subject"]
    assert result["next_phase"] in {"PROBLEM_DEFINE", "GATHER"}
    assert result["collected_data"].get("title") != "promate 2"


def test_direct_goal_suffix_statement_is_extracted():
    assert _extract_direct_fact_updates(
        "공공시설 이용 과정에서 반복되는 불편과 정보 부족 문제를 줄인다가 목표야"
    ) == {
        "goal": "공공시설 이용 과정에서 반복되는 불편과 정보 부족 문제를 줄인다"
    }


def test_numbered_goal_reply_keeps_goal_body_instead_of_tail_fragment():
    assert _extract_direct_fact_updates(
        "2. 공공시설 이용 과정에서 반복되는 불편과 정보 부족 문제를 줄인다를 목표로 할게"
    ) == {
        "goal": "공공시설 이용 과정에서 반복되는 불편과 정보 부족 문제를 줄인다"
    }


def test_goal_change_statement_extracts_body_before_goal_keyword():
    assert _extract_direct_fact_updates(
        "바코드 스캔으로 유통기한 자동 등록(빠른 MVP)으로 목표 바꿀래"
    ) == {
        "goal": "바코드 스캔으로 유통기한 자동 등록(빠른 MVP)"
    }


def test_request_normalization_drops_unconfirmed_goal_help_phrase():
    normalized = normalize_collected_data(
        {
            "subject": "공공시설 효율화 앱",
            "goal": "아직 정하지 못했어 도와줘",
        }
    )

    assert normalized == {"subject": "공공시설 효율화 앱"}


def test_unconfirmed_goal_is_replaced_by_explicit_goal():
    decision = evaluate_candidate_update(
        key="goal",
        current_value="아직 정하지 못했어 도와줘",
        incoming_value="공공시설 이용 과정에서 반복되는 불편과 정보 부족 문제를 줄인다",
        source="llm",
        user_message="공공시설 이용 과정에서 반복되는 불편과 정보 부족 문제를 줄인다가 목표야",
        current_phase="GATHER",
        current_data={"goal": "아직 정하지 못했어 도와줘"},
        candidate_updates={"goal": "공공시설 이용 과정에서 반복되는 불편과 정보 부족 문제를 줄인다"},
    )

    assert decision.approved is True
    assert decision.reason == "replace_unconfirmed_goal"


def test_gather_node_blocks_uncommitted_goal_confirmation_reply():
    with patch("app.ai.graph.nodes._fetch_rag_context", return_value=""), patch(
        "app.ai.graph.nodes._extract_direct_fact_updates",
        return_value={},
    ), patch(
        "app.ai.graph.nodes._interpret_turn_type",
        return_value="general",
    ), patch(
        "app.ai.graph.nodes._invoke_llm",
        return_value=SimpleNamespace(
            content=(
                '{"intent":"general","ai_message":"목표 확인했습니다: \\"새 목표\\". 더 구체화할까요?",'
                '"updated_data":{"goal":"새 목표"},"is_sufficient":false}'
            )
        ),
    ):
        result = gather_information_node(
            _make_state(
                message="@mates 새 목표로 정리해줘",
                collected_data={"subject": "공공시설 효율화 앱", "goal": "기존 목표"},
            )
        )

    assert result["collected_data"]["goal"] == "기존 목표"
    assert "목표 확인했습니다" not in result["ai_message"]


def test_goal_query_answers_current_value_without_rejection():
    result = gather_information_node(
        _make_state(
            message="@mates 기존 목표가 뭔데",
            collected_data={"subject": "공공시설 예약", "goal": "기존 목표"},
        )
    )

    assert result["collected_data"] == {"subject": "공공시설 예약", "goal": "기존 목표"}
    assert "현재 목표는 '기존 목표'입니다." in result["ai_message"]
    assert "수정 의도가 분명할 때만 바꿀게요" not in result["ai_message"]


def test_protected_goal_does_not_overwrite_on_ambiguous_help_request():
    decision = evaluate_candidate_update(
        key="goal",
        current_value="기존 목표",
        incoming_value="새 목표",
        source="llm",
        user_message="목표 추천해줘",
        current_phase="GATHER",
        current_data={"goal": "기존 목표"},
        candidate_updates={"goal": "새 목표"},
    )

    assert decision.approved is False
    assert decision.reason == "protected_field_requires_explicit_or_strong_restatement"


def test_goal_query_message_does_not_become_direct_goal_update():
    updates = _extract_direct_fact_updates("@mates 기존 목표가 뭔데")

    assert "goal" not in updates


def test_due_date_rejects_approximate_overwrite():
    decision = evaluate_candidate_update(
        key="dueDate",
        current_value="2026-06-30",
        incoming_value="6월쯤",
        source="llm",
        user_message="마감은 6월쯤?",
        current_phase="GATHER",
        current_data={"dueDate": "2026-06-30"},
        candidate_updates={"dueDate": "6월쯤"},
    )

    assert decision.approved is False
    assert decision.requires_followup_question is True
    assert decision.reason == "approximate_due_date_cannot_overwrite_confirmed_value"


def test_explicit_correction_overwrite_is_approved():
    decision = evaluate_candidate_update(
        key="goal",
        current_value="기존 목표",
        incoming_value="수정된 목표",
        source="direct_structured",
        user_message="아니, 목표는 수정된 목표야",
        current_phase="GATHER",
        current_data={"goal": "기존 목표"},
        candidate_updates={"goal": "수정된 목표"},
    )

    assert decision.approved is True
    assert decision.overwrite_mode == "EXPLICIT"


def test_goal_change_intent_overwrite_is_approved():
    decision = evaluate_candidate_update(
        key="goal",
        current_value="식재료 소비기한을 자동으로 알려준다",
        incoming_value="바코드 스캔으로 유통기한 자동 등록",
        source="direct_structured",
        user_message="바코드 스캔으로 유통기한 자동 등록으로 목표 바꿀래",
        current_phase="READY",
        current_data={"goal": "식재료 소비기한을 자동으로 알려준다"},
        candidate_updates={"goal": "바코드 스캔으로 유통기한 자동 등록"},
    )

    assert decision.approved is True
    assert decision.overwrite_mode == "EXPLICIT"


def test_strong_restatement_overwrite_is_approved():
    decision = evaluate_candidate_update(
        key="goal",
        current_value="기존 목표",
        incoming_value="최종 목표",
        source="llm",
        user_message="정리하면 목표는 최종 목표입니다",
        current_phase="GATHER",
        current_data={"goal": "기존 목표"},
        candidate_updates={"goal": "최종 목표"},
    )

    assert decision.approved is True
    assert decision.overwrite_mode == "STRONG_RESTATEMENT"


def test_roles_team_size_soft_conflict_is_saved_with_followup():
    decision = evaluate_candidate_update(
        key="roles",
        current_value=None,
        incoming_value=["PM", "디자이너", "개발자 1", "개발자 2"],
        source="direct_structured",
        user_message="역할은 PM, 디자이너, 개발자 2명입니다",
        current_phase="GATHER",
        current_data={"teamSize": 3},
        candidate_updates={"teamSize": 3, "roles": ["PM", "디자이너", "개발자 1", "개발자 2"]},
    )

    assert decision.approved is True
    assert decision.conflict_severity == "SOFT"
    assert decision.requires_followup_question is True


def test_roles_team_size_hard_conflict_is_rejected():
    roles = ["PM", "디자이너", "개발자 1", "개발자 2", "개발자 3", "QA"]
    decision = evaluate_candidate_update(
        key="roles",
        current_value=None,
        incoming_value=roles,
        source="direct_structured",
        user_message="역할은 PM, 디자이너, 개발 3명, QA입니다",
        current_phase="GATHER",
        current_data={"teamSize": 2},
        candidate_updates={"teamSize": 2, "roles": roles},
    )

    assert decision.approved is False
    assert decision.conflict_severity == "HARD"
    assert decision.requires_followup_question is True
    assert classify_role_team_size_conflict(roles, 2).value == "HARD"


def test_suggested_questions_prioritize_rejected_and_phase_fields():
    questions = _build_suggested_questions(
        phase="GATHER",
        collected_data={"subject": "공공시설 예약"},
        rejected_updates={"dueDate": "6월쯤"},
        followup_fields=["roles"],
    )

    assert questions[0] == "각자 맡을 역할을 짧게 적어주세요."
    assert questions[1] == "마감일이나 발표일은 언제인가요?"
    assert "현재 팀원은 총 몇 명인가요?" in questions


def test_choose_next_question_field_skips_filled_slots():
    next_field = choose_next_question_field(
        {
            "subject": "AI 기반 팀 프로젝트 협업 지원 서비스",
            "goal": "역할 분담과 일정 관리를 더 쉽게 만드는 것",
            "dueDate": "6월 말",
            "deliverables": "웹 서비스 프로토타입과 발표 자료",
        },
        current_phase="GATHER",
    )

    assert next_field == "roles"


def test_detect_meta_or_request_like_distinguishes_summary_vs_recommendation():
    summary_flags = detect_meta_or_request_like("지금까지 정리된 내용 요약해줘")
    recommendation_flags = detect_meta_or_request_like(
        "캡스톤디자인 경진대회용 최종 산출물 추천해줘",
        requested_focus="deliverables",
    )

    assert summary_flags["kind"] == "request_summary"
    assert recommendation_flags["kind"] == "request_recommendation"


def test_mixed_next_step_fact_update_keeps_normalized_types():
    with patch(
        "app.ai.graph.nodes._extract_direct_fact_updates",
        return_value={"teamSize": 4, "roles": ["개발자", "기획자", "PM"]},
    ), patch(
        "app.ai.graph.nodes._interpret_turn_type",
        return_value="provide_fact",
    ):
        result = gather_information_node(
            _make_state(
                message="다음엔 팀원은 4명이고 역할은 개발자, 기획자, PM으로 할게",
                collected_data={"title": "공공화장실 실시간 혼잡 안내"},
            )
        )

    assert result["collected_data"]["teamSize"] == 4
    assert result["collected_data"]["roles"] == ["개발자", "기획자", "PM"]


def test_correction_utterance_keeps_team_size_as_int():
    with patch(
        "app.ai.graph.nodes._extract_direct_fact_updates",
        return_value={"teamSize": 5, "roles": ["개발자", "기획자", "PM"]},
    ), patch(
        "app.ai.graph.nodes._interpret_turn_type",
        return_value="provide_fact",
    ):
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


def test_role_count_mismatch_is_not_silently_truncated():
    result = gather_information_node(
        _make_state(
            message="개발자 3명, 기획자 1명, PM 1명",
            collected_data={"teamSize": 4},
        )
    )

    assert result["collected_data"]["teamSize"] == 4
    assert result["collected_data"]["roles"] == ["개발자 1", "개발자 2", "개발자 3", "기획자", "PM"]
    assert "총 인원은 4명" in result["ai_message"]


def test_topic_exists_node_explicit_new_topic_overrides_existing_anchor():
    result = topic_exists_node(
        _make_topic_state(
            message="주제: 빈 강의실 찾기",
            turn_policy="CAPTURE_TITLE",
            collected_data={"subject": "캠퍼스", "title": "임시 제목"},
        )
    )

    assert result["collected_data"]["subject"] == "빈 강의실 찾기"


def test_explicit_role_counts_expand_deterministically():
    merged = merge_collected_data(
        {"teamSize": 4},
        {"roles": ["개발자", "개발자", "기획자", "PM"]},
    )

    assert merged["roles"] == ["개발자 1", "개발자 2", "기획자", "PM"]


def test_gather_node_resets_stale_state_for_button_like_chat_message():
    result = gather_information_node(
        {
            **_make_state(
                message="예, 프로젝트 주제가 있습니다",
                collected_data={"subject": "캠퍼스", "goal": "기존 목표"},
            ),
            "current_phase": "GATHER",
        }
    )

    assert result["collected_data"] == {}
    assert result["next_phase"] == "TOPIC_SET"
    assert "한두 줄로 보내주세요" in result["ai_message"]


def test_topic_presence_button_message_is_not_problem_area_candidate():
    state = _make_topic_state(
        message="예, 프로젝트 주제가 있습니다",
        collected_data={"subject": "캠퍼스"},
    )

    assert _extract_problem_area_candidate(state, {"subject": "캠퍼스"}) == ""


def test_non_committal_title_update_does_not_overwrite_existing_title():
    merged = merge_collected_data(
        {"title": "공공시설 이용 관련"},
        {"title": "아니 도와줘"},
    )

    assert merged["title"] == "공공시설 이용 관련"


def test_help_request_with_existing_topic_returns_refinement_options():
    with patch(
        "app.ai.graph.nodes._invoke_llm",
        return_value=SimpleNamespace(
            content='{"slot":"problemArea","question":"이 주제에서 사용자가 먼저 겪는 문제는 무엇인가요?","options":["방문 전에 혼잡도와 대기시간을 알기 어렵다","예약 가능한 시간과 절차를 찾기 번거롭다","운영시간과 이용 규칙을 한눈에 파악하기 어렵다"],"fallback_question":"가장 먼저 해결할 문제를 한 줄로 말해 주세요.","generation_reason":"subject-grounded"}'
        ),
    ):
        result = gather_information_node(
            _make_state(
                message="어떤걸 하고 싶은지 잘 모르겠어 도와줘",
                collected_data={"subject": "공공시설 이용 관련"},
            )
        )

    assert result["collected_data"] == {"subject": "공공시설 이용 관련"}
    assert result["next_phase"] == "PROBLEM_DEFINE"
    assert "같이 좁혀볼게요" in result["ai_message"]
    assert "방문 전에 혼잡도와 대기시간을 알기 어렵다" in result["ai_message"]
    assert "예약 가능한 시간과 절차를 찾기 번거롭다" in result["ai_message"]


def test_problem_area_message_preserves_existing_subject():
    result = gather_information_node(
        _make_state(
            message="접근성 강화하는 문제",
            collected_data={"subject": "공공시설"},
        )
    )

    assert result["collected_data"]["subject"] == "공공시설"
    assert result["collected_data"]["problemArea"] == "접근성 강화"
    assert "공공시설" in result["ai_message"]
    assert "접근성 강화" in result["ai_message"]


def test_problem_area_refines_broad_subject_during_problem_define_phase():
    result = gather_information_node(
        {
            **_make_state(
                message="예약 불편 해결하는 쪽",
                collected_data={"subject": "공공시설"},
            ),
            "current_phase": "PROBLEM_DEFINE",
        }
    )

    assert result["collected_data"]["subject"] == "공공시설"
    assert "problemArea" in result["collected_data"]
    assert "예약" in result["collected_data"]["problemArea"]
    assert result["next_phase"] == "GATHER"
    assert "어떤 시설을 대상으로 하나요?" in result["ai_message"]


def test_problem_area_candidate_ignores_question_like_deliverables_request():
    state = _make_state(
        message="산출물은 뭐로 해야 하지?",
        collected_data={"subject": "공공시설"},
    )

    assert _extract_problem_area_candidate(state, {"subject": "공공시설"}) == ""
    assert _extract_direct_fact_updates("산출물은 뭐로 해야 하지?") == {}


def test_structured_goal_fact_turn_type_is_prioritized_over_problem_area():
    state = {
        **_make_state(
            message="목표는 공공시설 혼잡도를 빠르게 확인할 수 있도록 돕는 거예요",
            collected_data={"subject": "공공시설"},
        ),
        "current_phase": "PROBLEM_DEFINE",
    }

    assert (
        _interpret_turn_type(
            state,
            {"subject": "공공시설"},
            direct_updates={"goal": "공공시설 혼잡도를 빠르게 확인할 수 있도록 돕는다"},
        )
        == "provide_fact"
    )


def test_goal_statement_updates_goal_without_polluting_subject():
    with patch(
        "app.ai.graph.nodes._extract_direct_fact_updates",
        return_value={"goal": "공공시설 혼잡도를 빠르게 확인할 수 있도록 돕는다"},
    ):
        result = gather_information_node(
            {
                **_make_state(
                    message="목표는 공공시설 혼잡도를 빠르게 확인할 수 있도록 돕는 거예요",
                    collected_data={"subject": "공공시설"},
                ),
                "current_phase": "PROBLEM_DEFINE",
            }
        )

    assert result["collected_data"]["subject"] == "공공시설"
    assert result["collected_data"]["goal"] == "공공시설 혼잡도를 빠르게 확인할 수 있도록 돕는다"
    assert result["next_phase"] == "GATHER"


def test_off_topic_question_does_not_pollute_subject():
    with patch("app.ai.graph.nodes._fetch_rag_context", return_value=""), patch(
        "app.ai.graph.nodes._invoke_llm",
        return_value=SimpleNamespace(
            content='{"intent":"general","ai_message":"프로젝트 관련 정보만 정리할게요.","updated_data":{},"is_sufficient":false}'
        ),
    ):
        result = gather_information_node(
            {
                **_make_state(
                    message="근데 오늘 점심 뭐 먹지?",
                    collected_data={"subject": "공공시설"},
                ),
                "current_phase": "PROBLEM_DEFINE",
            }
        )

    assert result["collected_data"] == {"subject": "공공시설"}
    assert "점심" not in result["collected_data"]["subject"]


def test_problem_area_choice_with_suffix_uses_option_label():
    state = _make_topic_state(
        message="1번문제 해결하는걸로하자",
        collected_data={"subject": "공공시설"},
    )
    state["recent_messages"] = [
        "좋아요. '공공시설'까지는 잡혔고 아직 구체 문제는 미정이에요. 같이 좁혀볼게요.\n"
        "1. 혼잡도 확인\n"
        "2. 예약/대기 관리\n"
        "3. 운영시간·위치 안내\n"
        "4. 불편 신고·접근성 정보\n"
        "5. 아직 모르겠어요. 추천이 더 필요해요\n"
        "번호로 답하거나 더 끌리는 방향을 한 줄로 적어 주세요."
    ]

    result = gather_information_node(state)

    assert "혼잡도 확인" in result["ai_message"]
    assert "1번 문제" not in result["ai_message"]


def test_target_facility_reply_advances_after_problem_area_follow_up():
    state = _make_topic_state(
        message="도서관을 대상으로 한다고",
        collected_data={"subject": "공공시설"},
    )
    state["recent_messages"] = [
        "좋아요. 공공시설의 혼잡도 확인 문제로 좁혀볼게요. 어떤 시설을 대상으로 하나요? 예: 도서관, 공원, 주민센터, 버스터미널"
    ]

    result = gather_information_node(state)

    assert result["next_phase"] == "GATHER"
    assert "도서관" in result["ai_message"]
    assert "혼잡도 확인" in result["ai_message"]
    assert "어떤 시설을 대상으로 하나요" not in result["ai_message"]


def test_target_facility_reply_gets_own_turn_type():
    state = _make_topic_state(
        message="도서관",
        collected_data={"subject": "공공시설 빈자리 확인"},
    )
    state["current_phase"] = "GATHER"
    state["recent_messages"] = [
        "좋아요. 공공시설의 빈자리 확인 문제로 좁혀볼게요. 어떤 시설을 대상으로 하나요? 예: 도서관, 공원, 주민센터, 버스터미널"
    ]

    assert _interpret_turn_type(state, {"subject": "공공시설 빈자리 확인"}) == "provide_target_facility"


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
    assert result["next_phase"] == "GATHER"
    assert "공공시설 예약 효율화" in result["ai_message"]
    assert "목표" in result["ai_message"]


def test_topic_exists_node_broad_subject_moves_to_problem_define():
    result = topic_exists_node(
        _make_topic_state(
            message="주제는 공공시설",
            turn_policy="CAPTURE_TITLE",
        )
    )

    assert result["collected_data"] == {"subject": "공공시설"}
    assert result["next_phase"] == "PROBLEM_DEFINE"
    assert "어떤 문제를 해결하고 싶은지" in result["ai_message"]


def test_route_logic_keeps_title_only_state_in_topic_exists_node():
    state = {
        **_make_state(message="AI 팀 협업 지원 서비스"),
        "current_phase": "EXPLORE",
        "collected_data": {"title": "promate"},
    }

    assert route_logic(state) == "topic_exists_node"


def test_turn_policy_treats_title_only_state_as_missing_topic_in_topic_set():
    request = AIChatRequest(
        roomId=1,
        content="@mates AI 기반 팀프로젝트 협업 지원 서비스",
        actionType="CHAT",
        currentStatus="TOPIC_SET",
        collectedData={"title": "promate1"},
    )

    assert _derive_turn_policy(request) == "CAPTURE_TITLE"


def test_topic_exists_node_prefers_latest_user_topic_over_room_title_like_metadata():
    result = topic_exists_node(
        _make_topic_state(
            message="@mates 캠퍼스 내 문제 해결",
            turn_policy="CAPTURE_TITLE",
            collected_data={"title": "캠퍼스1"},
        )
    )

    assert result["collected_data"]["subject"] == "캠퍼스 내 문제 해결"
    assert "title" not in result["collected_data"]
    assert "캠퍼스1" not in result["ai_message"]


def test_topic_exists_node_explicit_new_topic_overrides_existing_anchor():
    result = topic_exists_node(
        _make_topic_state(
            message="주제: 빈 강의실 찾기",
            turn_policy="CAPTURE_TITLE",
            collected_data={"subject": "캠퍼스", "title": "임시 제목"},
        )
    )

    assert result["collected_data"]["subject"] == "빈 강의실 찾기"
    assert "title" not in result["collected_data"]
    assert "캠퍼스" not in result["ai_message"]


def test_target_facility_candidate_requires_actual_facility_noun():
    state = _make_topic_state(
        message="혼잡도 관리라고",
        collected_data={"subject": "공공시설"},
    )
    state["recent_messages"] = [
        "좋아요. 공공시설의 예약/대기 관리 문제로 좁혀볼게요. 어떤 시설을 대상으로 하나요? 예: 도서관, 공원, 주민센터, 버스터미널"
    ]

    assert _extract_target_facility_candidate(state, {"subject": "공공시설"}) == ""


def test_topic_exists_node_normalizes_subject_only_once_per_turn():
    with patch("app.ai.graph.nodes.settings.OPENAI_API_KEY", "test-key"), patch(
        "app.ai.graph.nodes._invoke_llm",
        return_value=SimpleNamespace(content='{"normalized":"공공시설 이용"}'),
    ) as invoke_mock:
        result = topic_exists_node(
            _make_topic_state(
                message="주제는 공공시설 ㅇ용",
                turn_policy="CAPTURE_TITLE",
            )
        )

    assert result["collected_data"] == {"subject": "공공시설 이용"}
    assert invoke_mock.call_count == 1


def test_gather_node_allows_llm_subject_merge_without_existing_topic_anchor():
    with patch("app.ai.graph.nodes._fetch_rag_context", return_value=""), patch(
        "app.ai.graph.nodes._extract_direct_fact_updates",
        return_value={},
    ), patch(
        "app.ai.graph.nodes._interpret_turn_type",
        return_value="general",
    ), patch(
        "app.ai.graph.nodes._invoke_llm",
        return_value=SimpleNamespace(
            content='{"intent":"provide_topic","ai_message":"좋아요. 주제로 기록할게요.","updated_data":{"subject":"학생 협업 일정 조율 서비스"},"is_sufficient":false}'
        ),
    ):
        result = gather_information_node(
            {
                **_make_state(message="이런 방향 생각 중이야"),
                "current_phase": "GATHER",
            }
        )

    assert result["collected_data"]["subject"] == "학생 협업 일정 조율 서비스"
    assert result["approved_updates"]["subject"] == "학생 협업 일정 조율 서비스"
    assert "기록" in result["ai_message"]


def test_gather_node_blocks_llm_subject_override_when_topic_anchor_exists():
    with patch("app.ai.graph.nodes._fetch_rag_context", return_value=""), patch(
        "app.ai.graph.nodes._extract_direct_fact_updates",
        return_value={},
    ), patch(
        "app.ai.graph.nodes._interpret_turn_type",
        return_value="general",
    ), patch(
        "app.ai.graph.nodes._invoke_llm",
        return_value=SimpleNamespace(
            content='{"intent":"provide_topic","ai_message":"좋아요. 주제로 기록할게요.","updated_data":{"subject":"학생 협업 일정 조율 서비스"},"is_sufficient":false}'
        ),
    ):
        result = gather_information_node(
            {
                **_make_state(
                    message="이런 방향 생각 중이야",
                    collected_data={"subject": "기존 팀 협업 주제", "title": "promate"},
                ),
                "current_phase": "GATHER",
            }
        )

    assert result["collected_data"] == {"subject": "기존 팀 협업 주제", "title": "promate"}
    assert result["approved_updates"] == {}
    assert "기록" not in result["ai_message"]
    assert "어떤 문제를 해결하고 싶은지" in result["ai_message"]


def test_problem_area_candidate_ignores_title_only_anchor():
    state = _make_topic_state(
        message="@mates 대학생 팀은 포지션이 명확하지 않아서 진행이 어려워",
        collected_data={"title": "promate1"},
    )

    assert _extract_problem_area_candidate(state, {"title": "promate1"}) == ""


def test_meta_feedback_is_classified_as_meta_request_and_not_problem_area():
    state = _make_state(
        message="@mates 엥 무슨 소리임",
        collected_data={"subject": "AI 기반 팀프로젝트 협업 지원 서비스"},
    )

    assert _interpret_turn_type(state, state["collected_data"]) == "meta_request"
    assert _extract_problem_area_candidate(state, state["collected_data"]) == ""


def test_transcript_flow_preserves_subject_and_blocks_meta_subject_pollution():
    first = topic_exists_node(
        _make_topic_state(
            message="@mates 주제는 AI 기반 팀프로젝트 협업 지원 서비스로 생각하고 있어",
            turn_policy="CAPTURE_TITLE",
            collected_data={"title": "promate1"},
        )
    )

    assert first["collected_data"]["subject"] == "AI 기반 팀프로젝트 협업 지원 서비스"
    assert "title" not in first["collected_data"]

    second = gather_information_node(
        {
            **_make_state(
                message="@mates 대학생 팀은 포지션이 명확하지않아서 프로젝트 진행에 어려움을 겪고 있어",
                collected_data=first["collected_data"],
            ),
            "current_phase": first["next_phase"],
            "recent_messages": [first["ai_message"]],
        }
    )

    assert second["collected_data"]["subject"] == "AI 기반 팀프로젝트 협업 지원 서비스"
    assert "워크숍" not in second["ai_message"]
    assert (
        "대상이나 상황" in second["ai_message"]
        or "가장 먼저 겪는 대상" in second["ai_message"]
    )

    third = gather_information_node(
        {
            **_make_state(
                message="@mates 엥 무슨 소리임",
                collected_data=second["collected_data"],
            ),
            "current_phase": second["next_phase"],
            "recent_messages": [first["ai_message"], second["ai_message"]],
        }
    )

    assert third["collected_data"]["subject"] == "AI 기반 팀프로젝트 협업 지원 서비스"
    assert "무슨 소리야" not in third["ai_message"]
    assert "무슨 소리야" not in str(third["collected_data"])


def test_due_date_candidate_is_extracted_before_llm_flow():
    assert _extract_direct_fact_updates("26년 말로 최종 마감하고 싶어") == {
        "dueDate": "2026년 말"
    }


def test_coerce_gather_llm_result_salvages_invalid_field_types():
    result = _coerce_gather_llm_result(
        {
            "intent": "ask_summary",
            "ai_message": "주제를 기록했습니다.",
            "updated_data": {
                "subject": "공공시설 접근성 강화",
                "roles": [],
                "teamSize": None,
                "dueDate": "...",
            },
            "is_sufficient": False,
        }
    )

    assert result["ai_message"] == "주제를 기록했습니다."
    assert result["updated_data"] == {"subject": "공공시설 접근성 강화"}


def test_chat_turn_policy_treats_topic_presence_button_label_as_ask_only():
    request = AIChatRequest(
        roomId=1,
        content="예, 프로젝트 주제가 있습니다",
        actionType="CHAT",
        currentStatus="TOPIC_SET",
        collectedData={},
    )

    assert _derive_turn_policy(request) == "ASK_ONLY"


def test_explicit_topic_message_does_not_use_topic_presence_classifier():
    TOPIC_PRESENCE_CLASSIFICATION_CACHE.clear()
    request = AIChatRequest(
        roomId=2,
        content="주제는 도서관 혼잡도 개선이에요",
        actionType="CHAT",
        currentStatus="TOPIC_SET",
        collectedData={},
    )

    with patch("app.ai.graph.topic_presence.settings.OPENAI_API_KEY", "test-key"), patch(
        "app.ai.graph.topic_presence._invoke_llm",
    ) as invoke_mock:
        assert _matches_topic_presence_button_message("주제는 도서관 혼잡도 개선이에요") is False
        assert _derive_turn_policy(request) == "CAPTURE_TITLE"

    assert invoke_mock.call_count == 0


def test_request_normalization_keeps_project_name_separate_from_title():
    normalized = normalize_collected_data(
        {
            "projectName": "test5",
            "goal": "팀 협업 문제를 줄이는 서비스로 만든다",
        }
    )

    assert normalized["projectName"] == "test5"
    assert "title" not in normalized


def test_direct_goal_query_does_not_extract_tail_fragment():
    assert _extract_direct_fact_updates("기존 목표가 뭔데") == {}


def test_pronoun_goal_statement_does_not_store_ambiguous_tail():
    assert _extract_direct_fact_updates("이걸 목표로 하자") == {}


def test_help_needed_repeats_same_prompted_slot_with_examples():
    state = _make_state(
        message="잘 모르겠어",
        collected_data={"subject": "공공시설 예약"},
    )
    state["current_phase"] = "GATHER"
    state["recent_messages"] = ["현재 이 프로젝트를 함께하는 인원은 몇 명인가요?"]

    result = gather_information_node(state)

    assert result["followup_fields"] == ["teamSize"]
    assert result["next_question_field"] == "teamSize"
    assert "예:" in result["ai_message"]


def test_problem_area_and_target_facility_are_preserved_in_collected_data():
    state = _make_topic_state(
        message="도서관",
        collected_data={"subject": "공공시설"},
    )
    state["current_phase"] = "GATHER"
    state["recent_messages"] = [
        "좋아요. 공공시설의 예약 문제로 좁혀볼게요. 어떤 시설을 대상으로 하나요? 예: 도서관, 공원"
    ]
    state["collected_data"]["problemArea"] = "예약 문제"

    result = gather_information_node(state)

    assert result["collected_data"]["problemArea"] == "예약 문제"
    assert result["collected_data"]["targetFacility"] == "도서관"


def test_ready_confirmation_phrase_is_not_problem_area_candidate():
    state = _make_state(
        message="@mates 분명하다니까",
        collected_data={
            "subject": "1인 가구를 위한 식품 소비기한 관리 앱",
            "title": "1인 가구를 위한 식품 소비기한 관리 앱",
            "goal": "식재료 소비기한을 자동으로 알려준다",
            "teamSize": 3,
            "roles": ["PM", "프론트엔드", "백엔드"],
            "dueDate": "6월 27일",
            "deliverables": "앱 프로토타입",
        },
    )
    state["current_phase"] = "READY"

    assert _extract_problem_area_candidate(state, state["collected_data"]) == ""


def test_shared_role_count_expression_is_normalized():
    assert normalize_roles("개발자와 PM 2명씩") == ["개발자 1", "개발자 2", "PM 1", "PM 2"]


def test_approved_snapshot_serializes_roles_for_java_dto():
    snapshot = build_approved_collected_data_snapshot(
        {
            "subject": "공공시설 관련",
            "teamSize": 4,
            "roles": ["개발자 1", "개발자 2", "PM 1", "PM 2"],
        }
    )

    assert snapshot["roles"] == "개발자 x2, PM x2"

def test_target_user_counts_as_problem_definition_context():
    data = {
        "subject": "teen depression prevention",
        "title": "teen depression prevention",
        "targetUser": "teenagers",
    }

    assert has_problem_definition_context(data) is True
    assert derive_phase_from_collected_data(data, current_phase="GATHER") == "GATHER"


def test_approved_snapshot_preserves_public_auxiliary_context():
    snapshot = build_approved_collected_data_snapshot(
        {
            "subject": "teen depression prevention",
            "title": "teen depression prevention",
            "targetUser": "teenagers",
            "problemArea": "difficulty recognizing low mood",
            "targetFacility": "mobile app",
        }
    )

    assert snapshot["targetUser"] == "teenagers"
    assert snapshot["problemArea"] == "difficulty recognizing low mood"
    assert snapshot["targetFacility"] == "mobile app"


def test_request_normalization_keeps_goal_starting_with_again_word():
    goal = "다시 네이버에서 레시피 안찾아봐도되게하기"

    normalized = normalize_collected_data({"goal": goal})

    assert normalized["goal"] == goal


def test_ai_chat_request_accepts_top_level_goal_as_collected_data():
    goal = "다시 네이버에서 레시피 안찾아봐도되게하기"

    request = AIChatRequest(
        roomId=1035,
        content="지금 목표 알려줘",
        actionType="CHAT",
        currentStatus="GATHER",
        goal=goal,
    )

    assert request.rawCollectedData == {"goal": goal}
    assert request.collectedData["goal"] == goal
