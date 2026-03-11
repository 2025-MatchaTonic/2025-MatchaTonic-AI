from typing import Dict


COLLECTED_DATA_FIELDS: Dict[str, str] = {
    "title": "프로젝트 제목",
    "goal": "프로젝트 목표",
    "teamSize": "팀 인원",
    "roles": "역할",
    "dueDate": "마감일",
    "deliverables": "산출물",
}


def build_collected_data_guide() -> str:
    return ", ".join(
        f'"{key}" ({label})' for key, label in COLLECTED_DATA_FIELDS.items()
    )


def build_collected_data_json_example() -> str:
    lines = [f'            "{key}": "..."' for key in COLLECTED_DATA_FIELDS]
    return "{\n" + ",\n".join(lines) + "\n        }"


def merge_collected_data(
    current_data: Dict[str, str], updated_data: Dict[str, str] | None
) -> Dict[str, str]:
    merged = dict(current_data or {})

    for key in COLLECTED_DATA_FIELDS:
        value = (updated_data or {}).get(key)
        if isinstance(value, str) and value.strip():
            merged[key] = value.strip()

    return merged
