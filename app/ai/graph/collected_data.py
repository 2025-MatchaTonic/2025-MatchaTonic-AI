import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, TypeAlias


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

FIELD_POLICY: Dict[str, dict[str, object]] = {
    "title": {"overwrite": "strict", "source_bias": "context", "allow_additive": False},
    "subject": {"overwrite": "guarded", "source_bias": "context", "allow_additive": False},
    "goal": {"overwrite": "strict", "source_bias": "context", "allow_additive": False},
    "teamSize": {
        "overwrite": "guarded",
        "source_bias": "structured",
        "allow_additive": False,
    },
    "roles": {"overwrite": "guarded", "source_bias": "mixed", "allow_additive": True},
    "dueDate": {"overwrite": "strict", "source_bias": "structured", "allow_additive": False},
    "deliverables": {
        "overwrite": "strict",
        "source_bias": "context",
        "allow_additive": True,
    },
}


class OverwriteMode(str, Enum):
    NONE = "NONE"
    EXPLICIT = "EXPLICIT"
    STRONG_RESTATEMENT = "STRONG_RESTATEMENT"


class ConflictSeverity(str, Enum):
    NONE = "NONE"
    SOFT = "SOFT"
    HARD = "HARD"


@dataclass
class CandidateDecision:
    key: str
    approved: bool
    normalized_value: object | None
    reason: str
    overwrite_mode: str
    source: str
    requires_followup_question: bool
    conflict_severity: str = "NONE"

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
    re.compile(r"^\s*(?:엥|에엥)\b"),
    re.compile(r"^\s*(?:엥|에엥)\s*(?:무슨|뭔)\s*소리"),
    re.compile(r"^\s*(?:무슨|뭔)\s*(?:말|소리)"),
    re.compile(r"^\s*(?:이상한데|이상해|말이\s*안\s*되|이해가\s*안)"),
)

IDENTIFIER_LIKE_NOISE_PATTERN = re.compile(r"^(?=.*\d{4,})(?=.*[ㄱ-ㅎㅏ-ㅣ])[A-Za-z0-9ㄱ-ㅎㅏ-ㅣ_-]+$")
ROOM_TITLE_METADATA_PATTERN = re.compile(r"^([A-Za-z가-힣]+)\s*([0-9]{1,3})$")
ROOM_TITLE_METADATA_PREFIXES: tuple[str, ...] = (
    "room",
    "chat",
    "project",
    "team",
    "group",
    "캠퍼스",
    "프로젝트",
    "채팅방",
    "팀",
)
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


EXPLICIT_CORRECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:아니|아니요|아니고|아니라|정정하면|수정하면|수정할게|수정은)\b"),
    re.compile(r"\b(?:다시|바꾸면|바꿀게|정확히는|말고)\b"),
)
STRONG_RESTATEMENT_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "__all__": (
        re.compile(r"정리하면"),
        re.compile(r"최종적으로"),
        re.compile(r"최종안은"),
        re.compile(r"확정하면"),
    ),
    "dueDate": (
        re.compile(r"마감은"),
        re.compile(r"데드라인은"),
    ),
    "goal": (
        re.compile(r"목표는"),
    ),
    "deliverables": (
        re.compile(r"산출물은"),
        re.compile(r"결과물은"),
    ),
    "roles": (
        re.compile(r"역할은"),
        re.compile(r"역할 구성이"),
    ),
}
STRONG_RESTATEMENT_ENDINGS: tuple[re.Pattern[str], ...] = (
    re.compile(r".+로 잡았어요"),
    re.compile(r".+입니다(?:\s*\.?)?$"),
)
APPROXIMATE_DUE_DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"쯤"),
    re.compile(r"정도"),
    re.compile(r"예정"),
    re.compile(r"아마"),
    re.compile(r"중순"),
    re.compile(r"초[에쯤]?"),
    re.compile(r"말[쯤]?"),
)


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


def _looks_like_room_title_metadata(value: object) -> bool:
    cleaned = _clean_string(value)
    if not cleaned:
        return False

    compact = re.sub(r"\s+", "", cleaned)
    if len(compact) < 3 or len(compact) > 16:
        return False

    match = ROOM_TITLE_METADATA_PATTERN.fullmatch(compact)
    if not match:
        return False

    prefix = match.group(1).lower()
    return prefix in ROOM_TITLE_METADATA_PREFIXES


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


def classify_role_team_size_conflict(
    roles: object,
    team_size: object,
) -> ConflictSeverity:
    normalized_roles = normalize_roles(roles)
    normalized_team_size = normalize_team_size(team_size)
    if not normalized_roles or normalized_team_size is None:
        return ConflictSeverity.NONE

    role_count = len(normalized_roles)
    if role_count <= normalized_team_size:
        return ConflictSeverity.NONE

    hard_threshold = (
        normalized_team_size + 4
        if normalized_team_size <= 2
        else normalized_team_size + max(3, normalized_team_size // 2 + 1)
    )
    if role_count >= hard_threshold:
        return ConflictSeverity.HARD
    return ConflictSeverity.SOFT


def is_explicit_correction_message(user_message: str) -> bool:
    cleaned = _clean_string(user_message)
    if not cleaned:
        return False
    return any(pattern.search(cleaned) for pattern in EXPLICIT_CORRECTION_PATTERNS)


def _strong_restatement_matches_key(key: str, user_message: str) -> bool:
    patterns = STRONG_RESTATEMENT_PATTERNS.get("__all__", ()) + STRONG_RESTATEMENT_PATTERNS.get(
        key,
        (),
    )
    return any(pattern.search(user_message) for pattern in patterns)


def detect_overwrite_mode(
    *,
    key: str,
    current_value: object,
    incoming_value: object,
    user_message: str,
) -> OverwriteMode:
    current_normalized = normalize_collected_value(key, current_value)
    incoming_normalized = normalize_collected_value(key, incoming_value)
    if current_normalized is None or incoming_normalized is None:
        return OverwriteMode.NONE
    if current_normalized == incoming_normalized:
        return OverwriteMode.NONE

    cleaned_message = _clean_string(user_message)
    if not cleaned_message:
        return OverwriteMode.NONE
    if is_explicit_correction_message(cleaned_message):
        return OverwriteMode.EXPLICIT
    if _strong_restatement_matches_key(key, cleaned_message) or any(
        pattern.search(cleaned_message) for pattern in STRONG_RESTATEMENT_ENDINGS
    ):
        return OverwriteMode.STRONG_RESTATEMENT
    return OverwriteMode.NONE


def _merge_additive_values(
    key: str,
    current_value: object,
    incoming_value: object,
) -> object | None:
    if key == "roles":
        current_roles = normalize_roles(current_value) or []
        incoming_roles = normalize_roles(incoming_value) or []
        if not incoming_roles:
            return None

        merged_roles = list(current_roles)
        current_compact = {role.lower() for role in current_roles}
        added = False
        for role in incoming_roles:
            if role.lower() in current_compact:
                continue
            merged_roles.append(role)
            current_compact.add(role.lower())
            added = True
        return merged_roles if added else None

    if key == "deliverables":
        current_cleaned = _clean_string(current_value)
        incoming_cleaned = _clean_string(incoming_value)
        if not current_cleaned or not incoming_cleaned:
            return None
        if incoming_cleaned in current_cleaned or current_cleaned in incoming_cleaned:
            return incoming_cleaned if len(incoming_cleaned) >= len(current_cleaned) else None
    return None


def _is_approximate_due_date(value: object) -> bool:
    cleaned = _clean_string(value)
    if not cleaned:
        return False
    return any(pattern.search(cleaned) for pattern in APPROXIMATE_DUE_DATE_PATTERNS)


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
    if is_meta_conversation(cleaned):
        return False
    if is_placeholder_value(cleaned):
        return False
    if any(keyword in normalized for keyword in NEGATIVE_VALUE_KEYWORDS):
        return False
    if looks_like_non_committal_value(cleaned):
        return False
    if key in {"subject", "title"} and _looks_like_identifier_noise(cleaned):
        return False
    if key in {"subject", "title"} and _looks_like_room_title_metadata(cleaned):
        return False
    if key in {"title", "goal"} and re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
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
        return normalized_roles or None

    cleaned = _clean_string(value)
    return cleaned or None


def build_role_team_size_conflict_message(roles: object, team_size: object) -> str:
    normalized_roles = normalize_roles(roles) or []
    normalized_team_size = normalize_team_size(team_size)
    if not normalized_roles or normalized_team_size is None:
        return ""

    severity = classify_role_team_size_conflict(normalized_roles, normalized_team_size)
    if severity == ConflictSeverity.HARD:
        return (
            f"역할 인원 합계가 {len(normalized_roles)}명인데 현재 총 인원은 "
            f"{normalized_team_size}명입니다. 현재 정보로는 설명이 어려워서 역할 구성이나 총 인원을 다시 확인해 주세요."
        )
    return (
        f"역할 인원 합계가 {len(normalized_roles)}명인데 현재 총 인원은 "
        f"{normalized_team_size}명입니다. 겸임 기준인지 포함해서 역할 구성이나 총 인원을 다시 확인해 주세요."
    )


def evaluate_candidate_update(
    *,
    key: str,
    current_value: object,
    incoming_value: object,
    source: str,
    user_message: str,
    current_phase: str,
    current_data: Mapping[str, object] | None,
    candidate_updates: Mapping[str, object] | None,
) -> CandidateDecision:
    policy = FIELD_POLICY.get(
        key,
        {"overwrite": "guarded", "source_bias": "mixed", "allow_additive": False},
    )
    effective_team_size = _effective_team_size(current_data, candidate_updates)
    team_size_for_key = incoming_value if key == "teamSize" else effective_team_size
    normalized_incoming = normalize_collected_value(key, incoming_value, team_size=team_size_for_key)
    overwrite_mode = detect_overwrite_mode(
        key=key,
        current_value=current_value,
        incoming_value=incoming_value,
        user_message=user_message,
    )

    if normalized_incoming is None:
        return CandidateDecision(
            key=key,
            approved=False,
            normalized_value=None,
            reason="invalid_or_empty_candidate",
            overwrite_mode=overwrite_mode.value,
            source=source,
            requires_followup_question=False,
        )

    if not is_valid_collected_value(key, normalized_incoming, team_size=team_size_for_key):
        return CandidateDecision(
            key=key,
            approved=False,
            normalized_value=None,
            reason="non_committal_or_placeholder_candidate",
            overwrite_mode=overwrite_mode.value,
            source=source,
            requires_followup_question=False,
        )

    current_team_size = normalize_team_size((current_data or {}).get("teamSize"))
    current_normalized = normalize_collected_value(
        key,
        current_value,
        team_size=current_team_size,
    )
    if current_normalized == normalized_incoming:
        return CandidateDecision(
            key=key,
            approved=True,
            normalized_value=normalized_incoming,
            reason="same_as_current",
            overwrite_mode=OverwriteMode.NONE.value,
            source=source,
            requires_followup_question=False,
        )

    additive_value = None
    if policy.get("allow_additive"):
        additive_value = _merge_additive_values(key, current_normalized, normalized_incoming)

    conflict_severity = ConflictSeverity.NONE
    requires_followup_question = False
    if key in {"roles", "teamSize"}:
        conflict_roles = normalized_incoming if key == "roles" else (candidate_updates or {}).get(
            "roles",
            (current_data or {}).get("roles"),
        )
        conflict_team_size = (
            normalized_incoming
            if key == "teamSize"
            else (candidate_updates or {}).get("teamSize", (current_data or {}).get("teamSize"))
        )
        conflict_severity = classify_role_team_size_conflict(conflict_roles, conflict_team_size)
        if conflict_severity == ConflictSeverity.HARD:
            return CandidateDecision(
                key=key,
                approved=False,
                normalized_value=None,
                reason="hard_role_team_size_conflict",
                overwrite_mode=overwrite_mode.value,
                source=source,
                requires_followup_question=True,
                conflict_severity=conflict_severity.value,
            )
        if conflict_severity == ConflictSeverity.SOFT:
            requires_followup_question = True

    if key == "dueDate" and current_normalized is not None and _is_approximate_due_date(
        normalized_incoming,
    ):
        return CandidateDecision(
            key=key,
            approved=False,
            normalized_value=None,
            reason="approximate_due_date_cannot_overwrite_confirmed_value",
            overwrite_mode=overwrite_mode.value,
            source=source,
            requires_followup_question=True,
        )

    if current_normalized is None:
        return CandidateDecision(
            key=key,
            approved=True,
            normalized_value=normalized_incoming,
            reason="new_fact_approved",
            overwrite_mode=overwrite_mode.value,
            source=source,
            requires_followup_question=requires_followup_question,
            conflict_severity=conflict_severity.value,
        )

    if key in {"subject", "title"} and source == "direct_structured" and current_phase in {
        "TOPIC_SET",
        "PROBLEM_DEFINE",
    }:
        return CandidateDecision(
            key=key,
            approved=True,
            normalized_value=normalized_incoming,
            reason="topic_field_direct_structured_override",
            overwrite_mode=overwrite_mode.value,
            source=source,
            requires_followup_question=requires_followup_question,
            conflict_severity=conflict_severity.value,
        )

    if additive_value is not None:
        return CandidateDecision(
            key=key,
            approved=True,
            normalized_value=additive_value,
            reason="additive_update_approved",
            overwrite_mode=overwrite_mode.value,
            source=source,
            requires_followup_question=requires_followup_question,
            conflict_severity=conflict_severity.value,
        )

    if overwrite_mode in {OverwriteMode.EXPLICIT, OverwriteMode.STRONG_RESTATEMENT}:
        return CandidateDecision(
            key=key,
            approved=True,
            normalized_value=normalized_incoming,
            reason="overwrite_approved",
            overwrite_mode=overwrite_mode.value,
            source=source,
            requires_followup_question=requires_followup_question,
            conflict_severity=conflict_severity.value,
        )

    overwrite_policy = str(policy.get("overwrite") or "guarded")
    if overwrite_policy == "strict":
        reason = "protected_field_requires_explicit_or_strong_restatement"
    else:
        reason = "guarded_field_kept_existing_value"

    return CandidateDecision(
        key=key,
        approved=False,
        normalized_value=None,
        reason=reason,
        overwrite_mode=overwrite_mode.value,
        source=source,
        requires_followup_question=requires_followup_question,
        conflict_severity=conflict_severity.value,
    )


def is_placeholder_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return False
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0 or all(is_placeholder_value(item) for item in value)

    cleaned = _clean_string(value)
    if not cleaned:
        return True

    normalized = cleaned.lower()
    return normalized in PLACEHOLDER_VALUE_KEYWORDS


def normalize_roles(value: object) -> list[str] | None:
    if value is None or isinstance(value, bool):
        return None

    def _clean_role_label(token: object) -> str:
        cleaned = _clean_string(token)
        if not cleaned:
            return ""
        cleaned = re.sub(r"^\s*(?:역할|구성|멤버|담당)\s*[:：]?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(?:으로|로|이가|은|는)$", "", cleaned).strip()
        cleaned = cleaned.strip(" .,!?:;\"'()[]")
        lowered = cleaned.lower()
        mapping = {"pm": "PM", "po": "PO", "ai": "AI", "ios": "iOS"}
        return mapping.get(lowered, cleaned)

    if isinstance(value, (list, tuple, set)):
        roles = [_clean_role_label(item) for item in value]
        cleaned_roles = [role for role in roles if role]
        return _number_duplicate_roles(cleaned_roles) or None

    if not isinstance(value, str):
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    tokens = [
        token.strip()
        for token in re.split(r"\s*(?:,|/| 그리고 | 및 | 와 | 과 )\s*", cleaned)
        if token.strip()
    ]
    parsed_counts: list[tuple[str, int]] = []
    if tokens:
        for token in tokens:
            match = re.fullmatch(r"(.+?)\s+(\d{1,2})\s*명(?:씩)?(?:\s*(?:으로|로.*)?)?", token)
            if not match:
                parsed_counts = []
                break
            role = _clean_role_label(match.group(1))
            count = int(match.group(2))
            if not role or count <= 0:
                parsed_counts = []
                break
            parsed_counts.append((role, count))
    if parsed_counts:
        expanded_roles: list[str] = []
        for role, count in parsed_counts:
            expanded_roles.extend([role] * count)
        return _number_duplicate_roles(expanded_roles) or None

    normalized_roles = [_clean_role_label(token) for token in tokens]
    cleaned_roles = [role for role in normalized_roles if role]
    if cleaned_roles:
        return _number_duplicate_roles(cleaned_roles)

    single_role = _clean_role_label(cleaned)
    return [single_role] if single_role else None


def detect_overwrite_mode(
    *,
    key: str,
    current_value: object,
    incoming_value: object,
    user_message: str,
) -> OverwriteMode:
    current_normalized = normalize_collected_value(key, current_value)
    incoming_normalized = normalize_collected_value(key, incoming_value)
    if current_normalized is None or incoming_normalized is None:
        return OverwriteMode.NONE
    if current_normalized == incoming_normalized:
        return OverwriteMode.NONE

    cleaned_message = _clean_string(user_message)
    if not cleaned_message:
        return OverwriteMode.NONE
    if is_explicit_correction_message(cleaned_message):
        return OverwriteMode.EXPLICIT
    if key in {"subject", "title", "goal", "dueDate", "deliverables", "roles"} and re.match(
        r"^\s*(?:주제|제목|목표|마감|산출물|결과물|역할)\s*[:：]",
        cleaned_message,
    ):
        return OverwriteMode.EXPLICIT
    if _strong_restatement_matches_key(key, cleaned_message) or any(
        pattern.search(cleaned_message) for pattern in STRONG_RESTATEMENT_ENDINGS
    ):
        return OverwriteMode.STRONG_RESTATEMENT
    return OverwriteMode.NONE


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

    if (
        "roles" not in missing
        and classify_role_team_size_conflict(
            sanitized.get("roles"),
            sanitized.get("teamSize"),
        )
        == ConflictSeverity.HARD
    ):
        missing.append("roles")

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
    has_execution_fact = any(
        is_valid_collected_value(
            key,
            sanitized.get(key),
            team_size=sanitized.get("teamSize"),
        )
        for key in ("goal", "teamSize", "roles", "dueDate", "deliverables")
    )
    derived = "EXPLORE"
    if is_template_ready(sanitized):
        derived = "READY"
    elif has_subject(sanitized):
        if has_execution_fact:
            derived = "GATHER"
        elif subject_needs_problem_definition(subject):
            derived = "PROBLEM_DEFINE"
        else:
            derived = "GATHER"
    elif has_title(sanitized):
        derived = "GATHER" if has_execution_fact else "TOPIC_SET"
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


_original_is_request_like_value = is_request_like_value
_original_is_undecided_value = is_undecided_value
_original_is_valid_collected_value = is_valid_collected_value
_original_evaluate_candidate_update = evaluate_candidate_update


def is_request_like_value(value: object) -> bool:
    if _original_is_request_like_value(value):
        return True

    cleaned = _clean_string(value)
    if not cleaned:
        return False

    normalized = cleaned.lower()
    extra_keywords = ("도와줘", "도와 줘", "도움 필요", "같이 정하자")
    return any(keyword in normalized for keyword in extra_keywords)


def is_undecided_value(value: object) -> bool:
    if _original_is_undecided_value(value):
        return True

    cleaned = _clean_string(value)
    if not cleaned:
        return False

    return bool(
        re.match(r"^\s*아직\s+정하지\s+못했", cleaned)
        or re.match(r"^\s*아직\s+못\s+정했", cleaned)
    )


def looks_like_non_committal_value(value: object) -> bool:
    cleaned = _clean_string(value)
    if not cleaned:
        return False

    return is_request_like_value(cleaned) or is_undecided_value(cleaned) or is_meta_conversation(
        cleaned
    )


def is_valid_collected_value(key: str, value: object, *, team_size: object = None) -> bool:
    if not _original_is_valid_collected_value(key, value, team_size=team_size):
        return False

    cleaned = _clean(key, value)
    if key == "goal" and (is_request_like_value(cleaned) or is_undecided_value(cleaned)):
        return False
    return True


def evaluate_candidate_update(
    *,
    key: str,
    current_value: object,
    incoming_value: object,
    source: str,
    user_message: str,
    current_phase: str,
    current_data: Mapping[str, object] | None,
    candidate_updates: Mapping[str, object] | None,
) -> CandidateDecision:
    effective_team_size = _effective_team_size(current_data, candidate_updates)
    team_size_for_key = incoming_value if key == "teamSize" else effective_team_size
    normalized_incoming = normalize_collected_value(key, incoming_value, team_size=team_size_for_key)
    current_team_size = normalize_team_size((current_data or {}).get("teamSize"))
    current_normalized = normalize_collected_value(
        key,
        current_value,
        team_size=current_team_size,
    )

    if (
        key == "goal"
        and current_normalized is not None
        and not is_valid_collected_value(key, current_normalized, team_size=current_team_size)
        and normalized_incoming is not None
        and is_valid_collected_value(key, normalized_incoming, team_size=team_size_for_key)
    ):
        overwrite_mode = detect_overwrite_mode(
            key=key,
            current_value=current_value,
            incoming_value=incoming_value,
            user_message=user_message,
        )
        return CandidateDecision(
            key=key,
            approved=True,
            normalized_value=normalized_incoming,
            reason="replace_unconfirmed_goal",
            overwrite_mode=overwrite_mode.value,
            source=source,
            requires_followup_question=False,
        )

    return _original_evaluate_candidate_update(
        key=key,
        current_value=current_value,
        incoming_value=incoming_value,
        source=source,
        user_message=user_message,
        current_phase=current_phase,
        current_data=current_data,
        candidate_updates=candidate_updates,
    )
