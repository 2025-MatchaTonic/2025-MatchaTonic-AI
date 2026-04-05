import re
from typing import Dict, Mapping, TypeAlias


COLLECTED_DATA_FIELDS: Dict[str, str] = {
    "subject": "프로젝트 주제",
    "title": "프로젝트 제목",
    "goal": "프로젝트 목표",
    "teamSize": "팀 인원",
    "roles": "역할",
    "dueDate": "마감일",
    "deliverables": "산출물",
}
CollectedDataValue: TypeAlias = str | int | list[str]
CollectedData: TypeAlias = Dict[str, CollectedDataValue]

PHASE_ORDER = {
    "EXPLORE": 0,
    "TOPIC_SET": 1,
    "PROBLEM_DEFINE": 2,
    "GATHER": 3,
    "READY": 4,
}

SUBJECT_CONCRETE_KEYWORDS: tuple[str, ...] = (
    "서비스",
    "시스템",
    "플랫폼",
    "앱",
    "웹",
    "예약",
    "혼잡",
    "접근성",
    "개선",
    "관리",
    "안내",
    "분석",
    "추천",
    "예측",
    "효율화",
    "정보",
    "해결",
)
SUBJECT_CONCRETE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r".+를\s+위한\s+.+"),
    re.compile(r".+문제\s*해결"),
    re.compile(r".+개선"),
    re.compile(r".+안내\s*(?:서비스|시스템|앱|웹)?"),
)

REQUEST_LIKE_VALUE_KEYWORDS: tuple[str, ...] = (
    "요약해줘",
    "요약해 줘",
    "정리해줘",
    "정리해 줘",
    "채워줘",
    "채워 줘",
    "정해줘",
    "정해 줘",
    "세워줘",
    "세워 줘",
    "만들어줘",
    "만들어 줘",
    "추천해줘",
    "추천해 줘",
    "알려줘",
    "알려 줘",
    "미정인 항목",
    "정의되지 않은 부분",
    "남은 항목",
    "지금 모인 정보",
    "세션 요약",
    "fill missing",
    "summary",
)

NON_COMMITTAL_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(?:아니|아니야|아뇨)\s*(?:도와(?:줘|주세요)|추천해(?:줘|주세요)|정해(?:줘|주세요))?\s*$"),
    re.compile(r"^\s*(?:그게\s+아니라|다시|잠깐)\s*$"),
    re.compile(r"^\s*(?:잘\s*)?모르겠(?:어|어요|네|다)?\s*$"),
    re.compile(r"^\s*(?:잘\s*)?모르겠.*(?:도와|추천해|정해|같이)\S*\s*$"),
    re.compile(r"^\s*.*잘\s*모르겠.*(?:도와|추천해|정해|같이)\S*\s*$"),
    re.compile(r"^\s*도와(?:줘|주세요|주라)\s*$"),
    re.compile(r"^\s*추천해(?:줘|주세요|주라)\s*$"),
    re.compile(r"^\s*정해(?:줘|주세요|주라)\s*$"),
    re.compile(r"^\s*같이\s*(?:정하|해보)\S*\s*$"),
    re.compile(r"^\s*아직\s*(?:고민\s*중|못\s*정했|미정)\S*\s*$"),
    re.compile(r"^\s*(?:뭘|뭐를|무엇을|어떤\s*걸?)\s*(?:해야|만들어야|하고\s*싶은지)\s*잘\s*모르겠.*\s*$"),
)

NEGATIVE_VALUE_KEYWORDS: tuple[str, ...] = (
    "모르겠",
    "모름",
    "없음",
    "없어",
    "미정",
    "tbd",
    "unknown",
    "not sure",
    "idk",
    "don't know",
    "dont know",
    "no idea",
)

PLACEHOLDER_VALUE_KEYWORDS: tuple[str, ...] = (
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
)

META_CONVERSATION_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(?:아니|아니요|그게 아니라|다시|잠깐)\b"),
    re.compile(r"^\s*(?:왜|뭔 말|무슨 말|이게 무슨)\b"),
)

IDENTIFIER_LIKE_NOISE_PATTERN = re.compile(r"^(?=.*\d{4,})(?=.*[ㄱ-ㅎㅏ-ㅣ])[A-Za-z0-9ㄱ-ㅎㅏ-ㅣ_-]+$")
TEAM_SIZE_VALUE_PATTERN = re.compile(r"^\s*(\d{1,2})(?:\s*명)?\s*$")
ROLE_VALUE_PREFIX_PATTERN = re.compile(
    r"^\s*(?:역할|역할은|구성|구성은|담당|담당은)\s*[:은는이가]?\s*",
    re.IGNORECASE,
)
ROLE_VALUE_SPLIT_PATTERN = re.compile(r"\s*(?:,|/| 및 | 와 | 과 | 그리고 )\s*")
ROLE_COUNT_TOKEN_PATTERN = re.compile(
    r"^\s*(.+?)\s+(\d{1,2})\s*명(?:\s*씩)?(?:\s*(?:으로|로).*)?\s*$"
)
ROLE_TRAILING_PARTICLE_PATTERN = re.compile(r"(?:으로|로|은|는|이|가)$")


def _clean_string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def is_placeholder_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0 or all(is_placeholder_value(item) for item in value)

    cleaned = _clean_string(value)
    if not cleaned:
        return True

    normalized = cleaned.lower()
    return normalized in PLACEHOLDER_VALUE_KEYWORDS


def is_request_like_value(value: object) -> bool:
    cleaned = _clean_string(value)
    if not cleaned:
        return False
    normalized = cleaned.lower()
    return any(keyword in normalized for keyword in REQUEST_LIKE_VALUE_KEYWORDS)


def is_undecided_value(value: object) -> bool:
    cleaned = _clean_string(value)
    if not cleaned:
        return False
    normalized = cleaned.lower()
    if any(keyword in normalized for keyword in NEGATIVE_VALUE_KEYWORDS):
        return True
    return any(pattern.match(cleaned) for pattern in NON_COMMITTAL_VALUE_PATTERNS)


def is_meta_conversation(value: object) -> bool:
    cleaned = _clean_string(value)
    if not cleaned:
        return False
    return any(pattern.match(cleaned) for pattern in META_CONVERSATION_VALUE_PATTERNS)


def _looks_like_identifier_noise(value: object) -> bool:
    cleaned = _clean_string(value)
    if not cleaned:
        return False

    compact = re.sub(r"\s+", "", cleaned)
    if len(compact) < 8:
        return False
    if IDENTIFIER_LIKE_NOISE_PATTERN.fullmatch(compact):
        return True

    digit_count = sum(ch.isdigit() for ch in compact)
    jamo_count = len(re.findall(r"[ㄱ-ㅎㅏ-ㅣ]", compact))
    hangul_syllable_count = len(re.findall(r"[가-힣]", compact))
    latin_count = len(re.findall(r"[A-Za-z]", compact))

    return (
        digit_count >= max(5, len(compact) // 2)
        and hangul_syllable_count < 2
        and (jamo_count > 0 or latin_count < 3)
    )


def looks_like_non_committal_value(value: object) -> bool:
    cleaned = _clean_string(value)
    if not cleaned:
        return False

    return is_request_like_value(cleaned) or is_undecided_value(cleaned) or is_meta_conversation(
        cleaned
    )


def normalize_team_size(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        if value.is_integer() and value > 0:
            return int(value)
        return None

    cleaned = _clean_string(value)
    if not cleaned:
        return None

    match = TEAM_SIZE_VALUE_PATTERN.fullmatch(cleaned)
    if not match:
        return None

    parsed = int(match.group(1))
    return parsed if parsed > 0 else None


def _normalize_role_label(token: object) -> str:
    cleaned = _clean_string(token)
    if not cleaned:
        return ""

    cleaned = ROLE_VALUE_PREFIX_PATTERN.sub("", cleaned)
    cleaned = ROLE_TRAILING_PARTICLE_PATTERN.sub("", cleaned).strip()
    cleaned = cleaned.strip(" .,!?:;\"'()[]")
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


def _number_duplicate_roles(roles: list[str]) -> list[str]:
    if not roles:
        return []

    counts: dict[str, int] = {}
    for role in roles:
        counts[role] = counts.get(role, 0) + 1

    indexes: dict[str, int] = {}
    normalized: list[str] = []
    for role in roles:
        if counts[role] == 1:
            normalized.append(role)
            continue

        indexes[role] = indexes.get(role, 0) + 1
        normalized.append(f"{role} {indexes[role]}")

    return normalized


def _split_role_tokens(text: str) -> list[str]:
    stripped = ROLE_VALUE_PREFIX_PATTERN.sub("", text).strip()
    if not stripped:
        return []
    return [part.strip() for part in ROLE_VALUE_SPLIT_PATTERN.split(stripped) if part.strip()]


def _parse_explicit_role_counts(value: str) -> list[tuple[str, int]] | None:
    tokens = _split_role_tokens(value)
    if not tokens:
        return None

    parsed: list[tuple[str, int]] = []
    for token in tokens:
        match = ROLE_COUNT_TOKEN_PATTERN.fullmatch(token)
        if not match:
            return None

        role = _normalize_role_label(match.group(1))
        count = int(match.group(2))
        if not role or count <= 0:
            return None
        parsed.append((role, count))

    return parsed or None


def normalize_roles(value: object) -> list[str] | None:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (list, tuple, set)):
        normalized = [_normalize_role_label(item) for item in value]
        cleaned = [role for role in normalized if role]
        return _number_duplicate_roles(cleaned) or None

    if not isinstance(value, str):
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    counted_roles = _parse_explicit_role_counts(cleaned)
    if counted_roles:
        expanded_roles: list[str] = []
        for role, count in counted_roles:
            expanded_roles.extend([role] * count)
        return _number_duplicate_roles(expanded_roles) or None

    split_roles = [_normalize_role_label(token) for token in _split_role_tokens(cleaned)]
    normalized = [role for role in split_roles if role]
    if normalized:
        return _number_duplicate_roles(normalized)

    single_role = _normalize_role_label(cleaned)
    return [single_role] if single_role else None


def format_collected_value(key: str, value: object) -> str:
    if key == "teamSize":
        team_size = normalize_team_size(value)
        return str(team_size) if team_size is not None else _clean_string(value)

    if key == "roles":
        roles = normalize_roles(value)
        if roles:
            return ", ".join(roles)
        return _clean_string(value)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    return _clean_string(value)


def has_role_team_size_conflict(roles: object, team_size: object) -> bool:
    normalized_roles = normalize_roles(roles)
    normalized_team_size = normalize_team_size(team_size)
    if not normalized_roles or normalized_team_size is None:
        return False
    return len(normalized_roles) > normalized_team_size


def normalize_collected_value(
    key: str,
    value: object,
    *,
    team_size: object = None,
) -> CollectedDataValue | None:
    if is_placeholder_value(value):
        return None

    if key == "teamSize":
        return normalize_team_size(value)

    if key == "roles":
        normalized_roles = normalize_roles(value)
        if not normalized_roles:
            return None
        if has_role_team_size_conflict(normalized_roles, team_size):
            return None
        return normalized_roles

    cleaned = _clean_string(value)
    return cleaned or None


def _clean(key: str, value: object) -> str:
    normalized = normalize_collected_value(key, value)
    if normalized is None:
        return ""
    return format_collected_value(key, normalized)


def _effective_team_size(
    current_data: Mapping[str, object] | None,
    updated_data: Mapping[str, object] | None = None,
) -> int | None:
    updated_team_size = normalize_team_size((updated_data or {}).get("teamSize"))
    if updated_team_size is not None:
        return updated_team_size
    return normalize_team_size((current_data or {}).get("teamSize"))


def build_role_team_size_conflict_message(roles: object, team_size: object) -> str:
    normalized_roles = normalize_roles(roles) or []
    normalized_team_size = normalize_team_size(team_size)
    if not normalized_roles or normalized_team_size is None:
        return ""

    return (
        f"역할 인원 합계가 {len(normalized_roles)}명인데 현재 팀 인원은 "
        f"{normalized_team_size}명입니다. 역할 구성이나 팀 인원을 다시 확인해 주세요."
    )


def is_valid_collected_value(key: str, value: object, *, team_size: object = None) -> bool:
    cleaned = _clean(key, value)
    if not cleaned:
        return False

    normalized = cleaned.lower()
    if "@mates" in normalized or "?" in cleaned:
        return False
    if is_placeholder_value(cleaned):
        return False
    if any(keyword in normalized for keyword in NEGATIVE_VALUE_KEYWORDS):
        return False
    if looks_like_non_committal_value(cleaned):
        return False
    if key in {"subject", "title"} and _looks_like_identifier_noise(cleaned):
        return False
    if key in {"title", "goal"} and re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
        return False
    if key == "roles" and has_role_team_size_conflict(value, team_size):
        return False
    return True


def normalize_scalar_field(value: object) -> str | None:
    if is_placeholder_value(value):
        return None
    cleaned = _clean_string(value)
    return cleaned or None


def normalize_roles_field(value: object) -> list[str] | None:
    if is_placeholder_value(value):
        return None
    return normalize_roles(value)


def sanitize_llm_updated_data(raw_updated_data: object) -> CollectedData:
    if not isinstance(raw_updated_data, Mapping):
        return {}

    sanitized: CollectedData = {}
    normalized_team_size = normalize_team_size(raw_updated_data.get("teamSize"))
    if normalized_team_size is not None:
        sanitized["teamSize"] = normalized_team_size

    for key in COLLECTED_DATA_FIELDS:
        if key == "teamSize":
            continue

        value = raw_updated_data.get(key)
        if key == "roles":
            normalized_value = normalize_roles_field(value)
        else:
            normalized_value = normalize_collected_value(
                key,
                normalize_scalar_field(value),
                team_size=normalized_team_size,
            )

        if normalized_value is None:
            continue
        if not is_valid_collected_value(key, normalized_value, team_size=normalized_team_size):
            continue
        sanitized[key] = normalized_value

    return sanitized


def sanitize_collected_data(data: Mapping[str, object] | None) -> CollectedData:
    sanitized: CollectedData = {}
    team_size = normalize_team_size((data or {}).get("teamSize"))
    if team_size is not None:
        sanitized["teamSize"] = team_size

    for key in COLLECTED_DATA_FIELDS:
        if key == "teamSize":
            continue

        value = (data or {}).get(key)
        if not is_valid_collected_value(key, value, team_size=team_size):
            continue

        normalized_value = normalize_collected_value(key, value, team_size=team_size)
        if normalized_value is not None:
            sanitized[key] = normalized_value

    return sanitized


def sanitize_candidate_updates(
    updated_data: Mapping[str, object] | None,
    *,
    current_data: Mapping[str, object] | None = None,
) -> CollectedData:
    sanitized: CollectedData = {}
    team_size = _effective_team_size(current_data, updated_data)
    normalized_team_size = normalize_team_size((updated_data or {}).get("teamSize"))
    if normalized_team_size is not None:
        sanitized["teamSize"] = normalized_team_size

    for key in COLLECTED_DATA_FIELDS:
        if key == "teamSize":
            continue

        value = (updated_data or {}).get(key)
        if not is_valid_collected_value(key, value, team_size=team_size):
            continue

        normalized_value = normalize_collected_value(key, value, team_size=team_size)
        if normalized_value is not None:
            sanitized[key] = normalized_value

    return sanitized


def missing_collected_fields(data: Mapping[str, object] | None) -> list[str]:
    sanitized = sanitize_collected_data(data)
    missing: list[str] = []

    if not (
        is_valid_collected_value("subject", sanitized.get("subject"))
        or is_valid_collected_value("title", sanitized.get("title"))
    ):
        missing.append("subject")

    for key in COLLECTED_DATA_FIELDS:
        if key in {"subject", "title"}:
            continue
        if not is_valid_collected_value(
            key,
            sanitized.get(key),
            team_size=sanitized.get("teamSize"),
        ):
            missing.append(key)

    return missing


def is_template_ready(data: Mapping[str, object] | None) -> bool:
    return not missing_collected_fields(data)


def has_title(data: Mapping[str, object] | None) -> bool:
    sanitized = sanitize_collected_data(data)
    return is_valid_collected_value("title", sanitized.get("title"))


def has_subject(data: Mapping[str, object] | None) -> bool:
    sanitized = sanitize_collected_data(data)
    return is_valid_collected_value("subject", sanitized.get("subject"))


def subject_needs_problem_definition(subject: object) -> bool:
    cleaned = _clean_string(subject)
    if not cleaned:
        return False

    normalized = re.sub(r"\s+", " ", cleaned).strip()
    lowered = normalized.lower()

    if any(keyword in lowered for keyword in SUBJECT_CONCRETE_KEYWORDS):
        return False
    if any(pattern.search(normalized) for pattern in SUBJECT_CONCRETE_PATTERNS):
        return False

    parts = [part for part in normalized.split(" ") if part]
    if len(parts) <= 2 and len(normalized) <= 12:
        return True

    return not any(token in normalized for token in ("문제", "기능", "대상", "사용자", "프로젝트"))


def has_any_collected_fact(data: Mapping[str, object] | None) -> bool:
    return bool(sanitize_collected_data(data))


def derive_phase_from_collected_data(
    data: Mapping[str, object] | None,
    *,
    current_phase: str = "EXPLORE",
) -> str:
    sanitized = sanitize_collected_data(data)
    subject = sanitized.get("subject")
    derived = "EXPLORE"
    if is_template_ready(sanitized):
        derived = "READY"
    elif has_title(sanitized):
        derived = "GATHER"
    elif has_subject(sanitized):
        has_execution_fact = any(
            is_valid_collected_value(
                key,
                sanitized.get(key),
                team_size=sanitized.get("teamSize"),
            )
            for key in ("goal", "teamSize", "roles", "dueDate", "deliverables")
        )
        if has_execution_fact:
            derived = "GATHER"
        elif subject_needs_problem_definition(subject):
            derived = "PROBLEM_DEFINE"
        else:
            derived = "GATHER"
    elif sanitized:
        derived = "TOPIC_SET"

    if derived == "EXPLORE" and current_phase == "TOPIC_SET":
        return "TOPIC_SET"
    if (
        derived == "PROBLEM_DEFINE"
        and current_phase in {"GATHER", "READY"}
        and has_subject(sanitized)
        and not has_title(sanitized)
        and not any(
            is_valid_collected_value(
                key,
                sanitized.get(key),
                team_size=sanitized.get("teamSize"),
            )
            for key in ("goal", "teamSize", "roles", "dueDate", "deliverables")
        )
    ):
        return "PROBLEM_DEFINE"
    if PHASE_ORDER.get(derived, 0) >= PHASE_ORDER.get(current_phase, 0):
        return derived
    return current_phase


def build_collected_data_guide() -> str:
    return ", ".join(f'"{key}" ({label})' for key, label in COLLECTED_DATA_FIELDS.items())


def build_collected_data_json_example() -> str:
    lines = [
        '            "subject": "..."',
    ]
    return "{\n" + ",\n".join(lines) + "\n        }"


def merge_collected_data(
    current_data: Mapping[str, object],
    updated_data: Mapping[str, object] | None,
) -> CollectedData:
    merged = sanitize_collected_data(current_data)
    sanitized_updates = sanitize_candidate_updates(updated_data, current_data=current_data)

    for key in COLLECTED_DATA_FIELDS:
        value = sanitized_updates.get(key)
        if value is not None:
            merged[key] = value

    return merged
