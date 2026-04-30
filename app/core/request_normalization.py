from typing import Any, Dict, List

from app.ai.graph.collected_data import CollectedData


PHASE_ALIASES = {
    "": "EXPLORE",
    "INIT": "EXPLORE",
    "IDEA": "EXPLORE",
    "START": "EXPLORE",
    "EXPLORE": "EXPLORE",
    "TOPIC": "TOPIC_SET",
    "TOPIC_SET": "TOPIC_SET",
    "PROBLEM_DEFINE": "PROBLEM_DEFINE",
    "COLLECT": "GATHER",
    "COLLECT_DATA": "GATHER",
    "COLLECTING": "GATHER",
    "DATA_COLLECTION": "GATHER",
    "GATHER": "GATHER",
    "READY": "READY",
    "READY_PLAN": "READY",
    "READY_DEV": "READY",
    "DONE": "DONE",
    "COMPLETE": "DONE",
    "COMPLETED": "DONE",
}

ACTION_ALIASES = {
    "": "CHAT",
    "CHAT": "CHAT",
    "BTN_YES": "BTN_YES",
    "BTN_NO": "BTN_NO",
    "BTN_GO_DEF": "BTN_GO_DEF",
    "BTN_PLAN": "BTN_PLAN",
    "BTN_DEV": "BTN_DEV",
}

COLLECTED_DATA_ALIASES = {
    "subject": "subject",
    "topic": "subject",
    "projectsubject": "subject",
    "title": "title",
    "projecttitle": "title",
    "projectname": "projectName",
    "name": "title",
    "goal": "goal",
    "projectgoal": "goal",
    "objective": "goal",
    "problemarea": "problemArea",
    "problem_area": "problemArea",
    "problem": "problemArea",
    "targetfacility": "targetFacility",
    "target_facility": "targetFacility",
    "facility": "targetFacility",
    "targetuser": "targetUser",
    "target_user": "targetUser",
    "teamsize": "teamSize",
    "team_size": "teamSize",
    "members": "teamSize",
    "roles": "roles",
    "role": "roles",
    "duedate": "dueDate",
    "deadline": "dueDate",
    "deliverables": "deliverables",
    "deliverable": "deliverables",
    "output": "deliverables",
    "outputs": "deliverables",
}

TEXT_COLLECTED_DATA_FIELDS = {
    "subject",
    "title",
    "goal",
    "dueDate",
    "deliverables",
    "projectName",
    "problemArea",
    "targetFacility",
    "targetUser",
}

PLACEHOLDER_VALUES = {
    "...",
    "…",
    "..",
    "-",
    "--",
    "n/a",
    "na",
    "null",
    "none",
    "미정",
    "미정임",
    "아직 없음",
    "아직없음",
    "없음",
    "tbd",
    "unknown",
}

ROOM_TITLE_METADATA_PREFIXES = {
    "room",
    "chat",
    "project",
    "team",
    "group",
    "캠퍼스",
    "프로젝트",
    "채팅방",
    "팀",
}


def normalize_phase(value: Any, default: str = "EXPLORE") -> str:
    raw = str(value or "").strip().upper()
    if raw in PHASE_ALIASES:
        return PHASE_ALIASES[raw]
    return default


def normalize_action_type(value: Any, default: str = "CHAT") -> str:
    raw = str(value or "").strip().upper()
    if raw in ACTION_ALIASES:
        return ACTION_ALIASES[raw]
    return default


def normalize_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def normalize_optional_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _is_hangul_syllable(ch: str) -> bool:
    return "가" <= ch <= "힣"


def _is_hangul_jamo(ch: str) -> bool:
    return "ㄱ" <= ch <= "ㅎ" or "ㅏ" <= ch <= "ㅣ"


def _is_latin(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def _looks_like_identifier_noise(value: str) -> bool:
    compact = "".join(value.split())
    if len(compact) < 8:
        return False

    digit_count = sum(ch.isdigit() for ch in compact)
    jamo_count = sum(_is_hangul_jamo(ch) for ch in compact)
    hangul_count = sum(_is_hangul_syllable(ch) for ch in compact)
    latin_count = sum(_is_latin(ch) for ch in compact)

    return (
        digit_count >= max(5, len(compact) // 2)
        and hangul_count < 2
        and (jamo_count > 0 or latin_count < 3)
    )


def _looks_like_room_title_metadata(value: str) -> bool:
    compact = "".join(value.split())
    if len(compact) < 3 or len(compact) > 16:
        return False

    split_at = len(compact)
    while split_at > 0 and compact[split_at - 1].isdigit():
        split_at -= 1
    if split_at == len(compact):
        return False

    prefix = compact[:split_at].lower()
    suffix = compact[split_at:]
    return 1 <= len(suffix) <= 3 and prefix in ROOM_TITLE_METADATA_PREFIXES


def _looks_like_unconfirmed_goal(value: str) -> bool:
    compact = "".join(value.split())
    lowered = value.lower()
    return (
        "정하지못했" in compact
        or "못정했" in compact
        or "이렇게 잡아볼 수 있어요" in value
        or lowered in {"help", "recommend", "suggest"}
        or value.endswith("도와줘")
        or value.endswith("추천해줘")
    )


def _looks_like_embedded_user_context(field: str, value: str) -> bool:
    if field != "problemArea":
        return False
    return "주 사용자는" in value or "대상 사용자는" in value


def _normalize_text_field(field: str, value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    if cleaned.lower() in PLACEHOLDER_VALUES:
        return None
    if "@mates" in cleaned.lower():
        return None
    if field in {"subject", "title"} and _looks_like_identifier_noise(cleaned):
        return None
    if field in {"subject", "title"} and _looks_like_room_title_metadata(cleaned):
        return None
    if field == "goal" and _looks_like_unconfirmed_goal(cleaned):
        return None
    if _looks_like_embedded_user_context(field, cleaned):
        return None
    return cleaned


def _normalize_team_size(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value.is_integer() and value > 0 else None
    if not isinstance(value, str):
        return None

    cleaned = value.strip()
    if cleaned.endswith("명"):
        cleaned = cleaned[:-1].strip()
    if cleaned.isdecimal():
        parsed = int(cleaned)
        return parsed if parsed > 0 else None
    return None


def _normalize_role_label(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    if lowered == "pm":
        return "PM"
    if lowered == "po":
        return "PO"
    if lowered == "ai":
        return "AI"
    if lowered == "ios":
        return "iOS"
    return cleaned


def _split_roles(value: str) -> list[str]:
    normalized = value
    for delimiter in [",", "/", " 그리고 ", " 및 ", "와 ", "과 "]:
        normalized = normalized.replace(delimiter, "|")
    return [
        _normalize_role_label(part)
        for part in normalized.split("|")
        if _normalize_role_label(part)
    ]


def _normalize_roles(value: Any, team_size: int | None = None) -> list[str] | str | None:
    if isinstance(value, list):
        roles = [_normalize_role_label(item) for item in value]
        roles = [role for role in roles if role]
    elif isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        roles = _split_roles(cleaned)
        if team_size is not None and len(roles) > team_size:
            return cleaned
    else:
        return None

    return roles or None


def normalize_collected_data(value: Any) -> CollectedData:
    if not isinstance(value, dict):
        return {}

    canonicalized: Dict[str, Any] = {}
    for key, raw_item in value.items():
        alias_key = str(key).strip()
        canonical_key = COLLECTED_DATA_ALIASES.get(alias_key.lower(), alias_key)
        canonicalized[canonical_key] = raw_item

    normalized: CollectedData = {}
    team_size = _normalize_team_size(canonicalized.get("teamSize"))
    if team_size is not None:
        normalized["teamSize"] = team_size

    for field in TEXT_COLLECTED_DATA_FIELDS:
        normalized_value = _normalize_text_field(field, canonicalized.get(field))
        if normalized_value is not None:
            normalized[field] = normalized_value

    roles = _normalize_roles(canonicalized.get("roles"), team_size=team_size)
    if roles is not None:
        normalized["roles"] = roles

    return normalized
