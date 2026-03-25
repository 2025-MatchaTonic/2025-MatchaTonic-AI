from typing import Any, Dict, List


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
    "title": "title",
    "projecttitle": "title",
    "projectname": "title",
    "name": "title",
    "goal": "goal",
    "projectgoal": "goal",
    "objective": "goal",
    "subject": "goal",
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


def normalize_collected_data(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}

    normalized: Dict[str, str] = {}
    for key, raw_item in value.items():
        item = ""
        if isinstance(raw_item, str):
            item = raw_item.strip()
        elif isinstance(raw_item, (int, float)) and not isinstance(raw_item, bool):
            item = str(raw_item).strip()
        elif isinstance(raw_item, list):
            parts = [
                str(part).strip()
                for part in raw_item
                if isinstance(part, (str, int, float)) and not isinstance(part, bool) and str(part).strip()
            ]
            item = ", ".join(parts).strip()

        if not item:
            continue

        alias_key = str(key).strip()
        canonical_key = COLLECTED_DATA_ALIASES.get(alias_key.lower())
        if canonical_key is None:
            canonical_key = alias_key
        normalized[canonical_key] = item

    return normalized
