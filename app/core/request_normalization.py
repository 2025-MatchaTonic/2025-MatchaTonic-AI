from typing import Any, Dict, List

from app.ai.graph.collected_data import (
    CollectedData,
    has_role_team_size_conflict,
    normalize_collected_value,
    normalize_team_size,
)


PHASE_ALIASES = {
    "": "EXPLORE",
    "INIT": "EXPLORE",
    "IDEA": "EXPLORE",
    "START": "EXPLORE",
    "EXPLORE": "EXPLORE",
    "TOPIC": "TOPIC_SET",
    "TOPIC_SET": "TOPIC_SET",
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
    "projectname": "title",
    "name": "title",
    "goal": "goal",
    "projectgoal": "goal",
    "objective": "goal",
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


def normalize_collected_data(value: Any) -> CollectedData:
    if not isinstance(value, dict):
        return {}

    canonicalized: Dict[str, Any] = {}
    for key, raw_item in value.items():
        alias_key = str(key).strip()
        canonical_key = COLLECTED_DATA_ALIASES.get(alias_key.lower(), alias_key)
        canonicalized[canonical_key] = raw_item

    normalized: CollectedData = {}
    team_size = normalize_team_size(canonicalized.get("teamSize"))
    if team_size is not None:
        normalized["teamSize"] = team_size

    for key, raw_item in canonicalized.items():
        if key == "teamSize":
            continue

        normalized_value = normalize_collected_value(key, raw_item, team_size=team_size)
        if normalized_value is not None:
            normalized[key] = normalized_value
            continue

        if key == "roles" and raw_item is not None:
            if has_role_team_size_conflict(raw_item, team_size):
                if isinstance(raw_item, str) and raw_item.strip():
                    normalized[key] = raw_item.strip()
                continue

        if isinstance(raw_item, str) and raw_item.strip():
            normalized[key] = raw_item.strip()

    return normalized
