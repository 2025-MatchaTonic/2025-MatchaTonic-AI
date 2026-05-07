import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, TypeAlias


logger = logging.getLogger(__name__)


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

AUXILIARY_STATE_FIELDS: tuple[str, ...] = (
    "problemArea",
    "targetFacility",
    "projectName",
    "targetUser",
)

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

LLM_DECISION_MIN_CONFIDENCE = 0.65


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
    re.compile(r"^\s*(?:잘\s*)?모르(?:겠|게)(?:어|어요|네|다)?\s*$"),
    re.compile(r"^\s*(?:잘\s*)?모르겠.*(?:도와|추천해|정해|같이)\S*\s*$"),
    re.compile(r"^\s*.*잘\s*모르겠.*(?:도와|추천해|정해|같이)\S*\s*$"),
    re.compile(r"^\s*도와(?:줘|주세요|주라)\s*$"),
    re.compile(r"^\s*추천해(?:줘|주세요|주라)\s*$"),
    re.compile(r"^\s*정해(?:줘|주세요|주라)\s*$"),
    re.compile(r"^\s*같이\s*(?:정하|해보)\S*\s*$"),
    re.compile(r"^\s*아직\s*(?:고민\s*중|못\s*정했|미정)\S*\s*$"),
    re.compile(r"^\s*(?:뭘|뭐를|무엇을|어떤\s*걸?)\s*(?:해야|만들어야|하고\s*싶은지)\s*잘\s*모르겠.*\s*$"),
    # "[주제] 잘 모르겠어/모르게어" 형태 — 앞에 명사구가 붙은 불확실 표현
    re.compile(r"^.+\s+(?:잘\s*)?모르(?:겠|게)(?:어|어요|네|다)?\s*$"),
    re.compile(r"^.+\s+(?:잘\s*)?모르겠(?:는데|는지|겠어)\s*$"),
)

NEGATIVE_VALUE_KEYWORDS: tuple[str, ...] = (
    "모르겠",
    "모르게어",
    "모름",
    "없음",
    "없어",
    "미정",
    "글쎄",      # 글쎄요, 글쎄 — 비확정 표현, 유효한 필드 값에 절대 등장 안 함
    "뭐가 좋",   # 뭐가 좋은지, 뭐가 좋을지 — 질문형, 유효한 필드 값에 등장 안 함
    "감이 안",   # 감이 안 와, 감이 안 잡혀 — 불확실 표현
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
    re.compile(r"^\s*(?:아니|아니요|그게 아니라|잠깐)\b"),
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
ROLE_TRAILING_PARTICLE_PATTERN = re.compile(r"(?:으로|로|은|는|이|가)$")


def _clean_string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def is_request_like_value(value: object) -> bool:
    cleaned = _clean_string(value)
    if not cleaned:
        return False
    normalized = cleaned.lower()
    if any(keyword in normalized for keyword in REQUEST_LIKE_VALUE_KEYWORDS):
        return True
    extra_keywords = ("도와줘", "도와 줘", "도움 필요", "같이 정하자")
    return any(keyword in normalized for keyword in extra_keywords)


def is_undecided_value(value: object) -> bool:
    cleaned = _clean_string(value)
    if not cleaned:
        return False
    normalized = cleaned.lower()
    if any(keyword in normalized for keyword in NEGATIVE_VALUE_KEYWORDS):
        return True
    if any(pattern.match(cleaned) for pattern in NON_COMMITTAL_VALUE_PATTERNS):
        return True
    return bool(
        re.match(r"^\s*아직\s+정하지\s+못했", cleaned)
        or re.match(r"^\s*아직\s+못\s+정했", cleaned)
    )


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
    # Check promate/test numeric patterns (case-insensitive)
    if re.fullmatch(r"(?:promate|test)\d+", compact.lower()):
        return True

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


def _is_structurally_valid_collected_value(
    key: str,
    value: object,
    *,
    team_size: object = None,
) -> bool:
    cleaned = _clean(key, value)
    if not cleaned:
        return False

    normalized = cleaned.lower()
    if "@mates" in normalized or "?" in cleaned:
        return False
    if is_placeholder_value(cleaned):
        return False
    if key in {"subject", "title"} and _looks_like_identifier_noise(cleaned):
        return False
    if key in {"subject", "title"} and _looks_like_room_title_metadata(cleaned):
        return False
    if key in {"title", "goal"} and re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
        return False
    if _looks_like_guidance_placeholder(key, cleaned):
        return False
    return True


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
    cleaned = cleaned.strip(" .,!?:;\"'[]")
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


def _format_roles_for_backend(value: object) -> str:
    roles = normalize_roles(value)
    if not roles:
        return _clean_string(value)

    counts: dict[str, int] = {}
    order: list[str] = []
    for role in roles:
        base = re.sub(r"\s+\d+$", "", role).strip()
        if not base:
            continue
        if base not in counts:
            order.append(base)
        counts[base] = counts.get(base, 0) + 1

    return ", ".join(
        f"{role} x{counts[role]}" if counts[role] > 1 else role
        for role in order
    )


def _split_role_tokens(text: str) -> list[str]:
    stripped = ROLE_VALUE_PREFIX_PATTERN.sub("", text).strip()
    if not stripped:
        return []
    return [part.strip() for part in ROLE_VALUE_SPLIT_PATTERN.split(stripped) if part.strip()]


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


def normalize_collected_value(
    key: str,
    value: object,
    *,
    team_size: object = None,
) -> CollectedDataValue | None:
    # Handle deliverables list by joining into a semicolon-separated string
    if key == "deliverables" and isinstance(value, (list, tuple, set)):
        parts = [_clean_string(item) for item in value if _clean_string(item)]
        if not parts:
            return None
        return "; ".join(parts)

    if is_placeholder_value(value):
        return None

    if key == "teamSize":
        return normalize_team_size(value)

    if key == "roles":
        normalized_roles = normalize_roles(value)
        return normalized_roles or None

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


def _looks_like_guidance_placeholder(key: str, value: object) -> bool:
    cleaned = _clean_string(value)
    if not cleaned:
        return False

    if key == "goal":
        return any(
            marker in cleaned
            for marker in (
                "목표는 이렇게 잡아볼 수 있어요",
                "이렇게 잡아볼 수 있어요",
                "그 문제라면 목표는",
            )
        )

    if key == "problemArea":
        return bool(
            re.match(r"^\s*(?:주\s*사용자|주요\s*사용자)\s*(?:은|는|이|가|:)", cleaned)
        )

    return False


def is_valid_collected_value(key: str, value: object, *, team_size: object = None) -> bool:
    return _is_structurally_valid_collected_value(
        key,
        value,
        team_size=team_size,
    )


def normalize_scalar_field(value: object) -> str | None:
    if is_placeholder_value(value):
        return None
    cleaned = _clean_string(value)
    return cleaned or None


def normalize_roles_field(value: object) -> list[str] | None:
    if is_placeholder_value(value):
        return None
    return normalize_roles(value)


def _normalize_auxiliary_value(key: str, value: object) -> str | None:
    cleaned = _clean_string(value)
    if not cleaned or is_placeholder_value(cleaned):
        return None
    if _looks_like_guidance_placeholder(key, cleaned):
        return None
    if key == "projectName":
        return cleaned
    return cleaned


def _preserve_auxiliary_state_fields(data: Mapping[str, object] | None) -> CollectedData:
    preserved: CollectedData = {}
    for key in AUXILIARY_STATE_FIELDS:
        normalized_value = _normalize_auxiliary_value(key, (data or {}).get(key))
        if normalized_value is not None:
            preserved[key] = normalized_value
    return preserved


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

    if _looks_like_room_title_metadata(sanitized.get("title")):
        sanitized.pop("title", None)
    sanitized.update(_preserve_auxiliary_state_fields(data))
    return sanitized


def sanitize_candidate_updates(
    updated_data: Mapping[str, object] | None,
    *,
    current_data: Mapping[str, object] | None = None,
) -> CollectedData:
    normalized_payload = dict(updated_data or {})
    if isinstance(normalized_payload.get("deliverables"), (list, tuple, set)):
        normalized_payload["deliverables"] = normalize_collected_value(
            "deliverables",
            normalized_payload.get("deliverables"),
        )

    sanitized: CollectedData = {}
    team_size = _effective_team_size(current_data, normalized_payload)
    normalized_team_size = normalize_team_size(normalized_payload.get("teamSize"))
    if normalized_team_size is not None:
        sanitized["teamSize"] = normalized_team_size

    for key in COLLECTED_DATA_FIELDS:
        if key == "teamSize":
            continue

        value = normalized_payload.get(key)
        if not is_valid_collected_value(key, value, team_size=team_size):
            continue

        normalized_value = normalize_collected_value(key, value, team_size=team_size)
        if normalized_value is not None:
            sanitized[key] = normalized_value

    preserved = _preserve_auxiliary_state_fields(current_data)
    preserved.update(_preserve_auxiliary_state_fields(normalized_payload))
    sanitized.update(preserved)
    return sanitized


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
    source_metadata: Mapping[str, object] | None = None,
) -> CandidateDecision:
    metadata = dict(source_metadata or {})
    effective_team_size = _effective_team_size(current_data, candidate_updates)
    team_size_for_key = incoming_value if key == "teamSize" else effective_team_size
    normalized_incoming = normalize_collected_value(key, incoming_value, team_size=team_size_for_key)

    if normalized_incoming is None or not is_valid_collected_value(key, normalized_incoming, team_size=team_size_for_key):
        return CandidateDecision(
            key=key, approved=False, normalized_value=None,
            reason="invalid_or_empty_candidate", overwrite_mode="NONE", source=source,
            requires_followup_question=False,
        )

    current_team_size = normalize_team_size((current_data or {}).get("teamSize"))
    current_normalized = normalize_collected_value(key, current_value, team_size=current_team_size)

    if current_normalized == normalized_incoming:
        return CandidateDecision(
            key=key, approved=True, normalized_value=normalized_incoming,
            reason="same_as_current", overwrite_mode="NONE", source=source,
            requires_followup_question=False,
        )

    if source == "llm_decision":
        confidence = metadata.get("confidence", 0)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0
        raw_evidence = str(metadata.get("raw_evidence") or "").strip()
        if confidence_value < LLM_DECISION_MIN_CONFIDENCE:
            return CandidateDecision(
                key=key, approved=False, normalized_value=None,
                reason="llm_low_confidence", overwrite_mode="NONE", source=source,
                requires_followup_question=False,
            )
        if not raw_evidence:
            return CandidateDecision(
                key=key, approved=False, normalized_value=None,
                reason="llm_missing_raw_evidence", overwrite_mode="NONE", source=source,
                requires_followup_question=False,
            )

    if key in {"roles", "teamSize"}:
        conflict_roles = normalized_incoming if key == "roles" else (candidate_updates or {}).get("roles", (current_data or {}).get("roles"))
        conflict_team_size = normalized_incoming if key == "teamSize" else (candidate_updates or {}).get("teamSize", (current_data or {}).get("teamSize"))
        conflict = classify_role_team_size_conflict(conflict_roles, conflict_team_size)
        if conflict == ConflictSeverity.HARD:
            return CandidateDecision(
                key=key, approved=False, normalized_value=None,
                reason="hard_role_team_size_conflict", overwrite_mode="NONE", source=source,
                requires_followup_question=True, conflict_severity=conflict.value,
            )

    policy = FIELD_POLICY.get(key, {"allow_additive": False})
    if policy.get("allow_additive") and current_normalized is not None:
        additive = _merge_additive_values(key, current_normalized, normalized_incoming)
        if additive is not None:
            return CandidateDecision(
                key=key, approved=True, normalized_value=additive,
                reason="additive_update_approved", overwrite_mode="NONE", source=source,
                requires_followup_question=False,
            )

    if (
        source == "llm_decision"
        and current_normalized is not None
        and metadata.get("intent") != "correct_info"
    ):
        return CandidateDecision(
            key=key, approved=False, normalized_value=None,
            reason="llm_overwrite_requires_confirmation", overwrite_mode="CONFIRM",
            source=source, requires_followup_question=True,
        )

    return CandidateDecision(
        key=key, approved=True, normalized_value=normalized_incoming,
        reason="approved",
        overwrite_mode="EXPLICIT" if source == "llm_decision" and current_normalized is not None else "NONE",
        source=source,
        requires_followup_question=False,
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
        cleaned = cleaned.strip(" .,!?:;\"'[]")
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

    shared_count_match = re.fullmatch(
        r"(.+?)\s+(?:각각\s*)?(\d{1,2})\s*명\s*씩(?:\s*(?:으로|로.*)?)?",
        cleaned,
    ) or re.fullmatch(
        r"(.+?)\s+각각\s+(\d{1,2})\s*명(?:\s*(?:으로|로.*)?)?",
        cleaned,
    )
    if shared_count_match:
        roles_text = shared_count_match.group(1)
        count = int(shared_count_match.group(2))
        role_tokens = [
            token.strip()
            for token in re.split(
                r"\s*(?:,|/|\s+그리고\s+|\s+및\s+|(?<=[가-힣])와\s+|(?<=[가-힣])과\s+|\s+와\s+|\s+과\s+)\s*",
                roles_text,
            )
            if token.strip()
        ]
        shared_roles = [_clean_role_label(token) for token in role_tokens]
        cleaned_shared_roles = [role for role in shared_roles if role]
        if cleaned_shared_roles and count > 0:
            expanded_roles: list[str] = []
            for role in cleaned_shared_roles:
                expanded_roles.extend([role] * count)
            return _number_duplicate_roles(expanded_roles) or None

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
                match = re.fullmatch(r"(.+?)\s*[xX]\s*(\d{1,2})", token)
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


def has_problem_definition_context(data: Mapping[str, object] | None) -> bool:
    sanitized = sanitize_collected_data(data)
    return bool(
        _normalize_auxiliary_value("problemArea", sanitized.get("problemArea"))
        or _normalize_auxiliary_value("targetFacility", sanitized.get("targetFacility"))
        or _normalize_auxiliary_value("targetUser", sanitized.get("targetUser"))
    )


def build_phase_derivation_trace(
    data: Mapping[str, object] | None,
    *,
    current_phase: str = "EXPLORE",
) -> dict[str, object]:
    sanitized = sanitize_collected_data(data)
    subject = _clean_string(sanitized.get("subject"))
    title = _clean_string(sanitized.get("title"))
    problem_area = _clean_string(sanitized.get("problemArea"))
    target_facility = _clean_string(sanitized.get("targetFacility"))
    execution_fact_keys = [
        key
        for key in ("goal", "teamSize", "roles", "dueDate", "deliverables")
        if is_valid_collected_value(
            key,
            sanitized.get(key),
            team_size=sanitized.get("teamSize"),
        )
    ]
    subject_needs_refinement = bool(subject and subject_needs_problem_definition(subject))
    has_problem_context = has_problem_definition_context(sanitized)

    derived = "EXPLORE"
    reason = "no_committed_data"
    if is_template_ready(sanitized):
        derived = "READY"
        reason = "all_required_fields_committed"
    elif subject:
        if execution_fact_keys:
            derived = "GATHER"
            reason = "execution_fields_available"
        elif subject_needs_refinement and not has_problem_context:
            derived = "PROBLEM_DEFINE"
            reason = "broad_subject_requires_problem_definition"
        else:
            derived = "GATHER"
            if subject_needs_refinement and has_problem_context:
                reason = "problem_definition_context_available"
            else:
                reason = "concrete_subject_available"
    elif title:
        derived = "GATHER" if execution_fact_keys else "TOPIC_SET"
        reason = (
            "title_plus_execution_fields_available"
            if execution_fact_keys
            else "title_available_without_subject"
        )
    elif sanitized:
        derived = "TOPIC_SET"
        reason = "partial_committed_data_without_topic_anchor"

    returned_phase = derived
    if derived == "EXPLORE" and current_phase == "TOPIC_SET":
        returned_phase = "TOPIC_SET"
        reason = "keep_topic_set_without_committed_topic"
    elif (
        derived == "PROBLEM_DEFINE"
        and current_phase in {"GATHER", "READY"}
        and subject
        and not title
        and not execution_fact_keys
    ):
        returned_phase = "PROBLEM_DEFINE"
        reason = "fall_back_to_problem_define_due_to_missing_execution_fields"
    elif PHASE_ORDER.get(derived, 0) < PHASE_ORDER.get(current_phase, 0):
        returned_phase = current_phase
        reason = f"keep_existing_phase_{current_phase.lower()}"

    return {
        "current_phase": current_phase,
        "derived_phase": derived,
        "returned_phase": returned_phase,
        "reason": reason,
        "has_subject": bool(subject),
        "has_title": bool(title),
        "subject_needs_problem_definition": subject_needs_refinement,
        "has_problem_definition_context": has_problem_context,
        "problem_area": problem_area,
        "target_facility": target_facility,
        "execution_fact_keys": execution_fact_keys,
        "sanitized_keys": sorted(sanitized.keys()),
    }


def derive_phase_from_collected_data(
    data: Mapping[str, object] | None,
    *,
    current_phase: str = "EXPLORE",
) -> str:
    trace = build_phase_derivation_trace(data, current_phase=current_phase)
    return str(trace["returned_phase"])


REQUIRED_COLLECTED_DATA_FIELDS: tuple[str, ...] = (
    "subject",
    "title",
    "goal",
    "teamSize",
    "roles",
    "dueDate",
    "deliverables",
)

PUBLIC_AUXILIARY_COLLECTED_DATA_FIELDS: tuple[str, ...] = (
    "problemArea",
    "targetFacility",
    "targetUser",
)

NEXT_FIELD_PRIORITY_BY_PHASE: dict[str, tuple[str, ...]] = {
    "EXPLORE": ("subject",),
    "TOPIC_SET": ("subject", "title", "goal"),
    "PROBLEM_DEFINE": ("subject", "goal", "title"),
    "GATHER": ("goal", "roles", "teamSize", "dueDate", "deliverables", "title"),
    "READY": ("goal", "roles", "teamSize", "dueDate", "deliverables", "title"),
    "DONE": ("goal", "teamSize", "roles", "dueDate", "deliverables"),
}


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

    for key in AUXILIARY_STATE_FIELDS:
        value = sanitized_updates.get(key)
        if value is not None:
            merged[key] = value

    return merged


def choose_next_question_field(
    data: Mapping[str, object] | None,
    *,
    current_phase: str = "GATHER",
    followup_fields: list[str] | None = None,
    rejected_updates: Mapping[str, object] | None = None,
) -> str:
    sanitized = sanitize_collected_data(data)
    pending: list[str] = []
    seen: set[str] = set()

    for field in followup_fields or []:
        if field and field not in seen:
            pending.append(field)
            seen.add(field)
    for field in (rejected_updates or {}).keys():
        if field and field not in seen and field not in sanitized:
            pending.append(field)
            seen.add(field)

    subject = _clean_string(sanitized.get("subject"))
    if (
        current_phase in {"EXPLORE", "TOPIC_SET", "PROBLEM_DEFINE"}
        and subject
        and subject_needs_problem_definition(subject)
        and not has_problem_definition_context(sanitized)
        and "subject" not in seen
    ):
        pending.append("subject")
        seen.add("subject")

    missing = set(missing_collected_fields(sanitized))
    for field in NEXT_FIELD_PRIORITY_BY_PHASE.get(
        current_phase,
        NEXT_FIELD_PRIORITY_BY_PHASE["GATHER"],
    ):
        if field in missing and field not in seen:
            pending.append(field)
            seen.add(field)

    return pending[0] if pending else ""


def apply_collected_data_updates(
    current: Mapping[str, object] | None,
    candidate: Mapping[str, object] | None,
    turn_type: str,
    current_status: str,
    recent_messages: list[str] | None = None,
    selected_message: str | None = None,
    user_message: str = "",
    candidate_sources: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[CollectedData, dict[str, object]]:
    current_data = sanitize_collected_data(current)
    raw_candidate = dict(candidate or {})

    approved_updates: dict[str, object] = {}
    rejected_updates: dict[str, object] = {}
    rejected_reasons: dict[str, str] = {}
    decisions: dict[str, CandidateDecision] = {}

    for key, raw_value in raw_candidate.items():
        if key not in REQUIRED_COLLECTED_DATA_FIELDS:
            continue
        source = str((candidate_sources or {}).get(key, {}).get("source") or "unknown")
        source_metadata = dict((candidate_sources or {}).get(key) or {})
        decision = evaluate_candidate_update(
            key=key,
            current_value=current_data.get(key),
            incoming_value=raw_value,
            source=source,
            source_metadata=source_metadata,
            user_message=user_message,
            current_phase=current_status,
            current_data=current_data,
            candidate_updates=raw_candidate,
        )
        decisions[key] = decision
        if decision.approved and decision.normalized_value is not None:
            approved_updates[key] = decision.normalized_value
        else:
            rejected_updates[key] = raw_value
            rejected_reasons[key] = decision.reason

    for key, raw_value in raw_candidate.items():
        if key not in AUXILIARY_STATE_FIELDS:
            continue
        normalized_value = _normalize_auxiliary_value(key, raw_value)
        if normalized_value is None:
            rejected_updates[key] = raw_value
            rejected_reasons[key] = "invalid_or_empty_candidate"
            continue
        current_value = _normalize_auxiliary_value(key, current_data.get(key))
        if current_value == normalized_value:
            continue
        approved_updates[key] = normalized_value
        rejected_updates.pop(key, None)
        rejected_reasons.pop(key, None)

    next_collected_data = merge_collected_data(current_data, approved_updates)
    needs_confirmation = [
        key
        for key, decision in decisions.items()
        if decision.requires_followup_question
    ]
    for key in raw_candidate:
        if key not in REQUIRED_COLLECTED_DATA_FIELDS and key not in AUXILIARY_STATE_FIELDS:
            continue
        decision = decisions.get(key)
        source_metadata = dict((candidate_sources or {}).get(key) or {})
        logger.info(
            "collected_data_decision field=%s candidate=%r normalized=%r decision=%s reason=%s source=%s confidence=%s raw_evidence=%r overwrite_mode=%s",
            key,
            raw_candidate.get(key),
            decision.normalized_value if decision is not None else approved_updates.get(key),
            "approved" if key in approved_updates else "rejected" if key in rejected_reasons else "ignored",
            rejected_reasons.get(key) or (decision.reason if decision is not None else "unchanged_or_auxiliary"),
            str((candidate_sources or {}).get(key, {}).get("source") or "unknown"),
            source_metadata.get("confidence"),
            source_metadata.get("raw_evidence"),
            decision.overwrite_mode if decision is not None else "NONE",
        )
    logger.info(
        "apply_collected_data_updates turn=%s phase=%s approved=%s rejected=%s before=%s after=%s",
        turn_type,
        current_status,
        approved_updates,
        rejected_reasons,
        current_data,
        next_collected_data,
    )
    return next_collected_data, {
        "approved": approved_updates,
        "rejected": rejected_updates,
        "rejected_reasons": rejected_reasons,
        "needs_confirmation": needs_confirmation,
        "decisions": decisions,
    }


_DISPLAY_SPEECH_ENDING_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\s*거예요\s*$"), " 것"),
    (re.compile(r"\s*거에요\s*$"), " 것"),
    (re.compile(r"\s*이에요\s*$"), ""),
    (re.compile(r"\s*이예요\s*$"), ""),
    (re.compile(r"\s*예요\s*$"), ""),
    (re.compile(r"\s*에요\s*$"), ""),
    (re.compile(r"\s*입니다\s*$"), ""),
    (re.compile(r"\s*습니다\s*$"), ""),
]
_DISPLAY_SPEECH_ENDING_FIELDS: frozenset[str] = frozenset(
    {"goal", "subject", "title", "deliverables"}
)


def _strip_display_endings(field: str, value: object) -> object:
    if field not in _DISPLAY_SPEECH_ENDING_FIELDS:
        return value
    if isinstance(value, str):
        for pattern, replacement in _DISPLAY_SPEECH_ENDING_PATTERNS:
            cleaned = pattern.sub(replacement, value)
            if cleaned != value:
                return cleaned.strip()
        return value
    if isinstance(value, list):
        return [_strip_display_endings(field, item) for item in value]
    return value


def build_approved_collected_data_snapshot(
    data: Mapping[str, object] | None,
) -> CollectedData:
    sanitized = sanitize_collected_data(data)
    snapshot: CollectedData = {}

    for key in REQUIRED_COLLECTED_DATA_FIELDS:
        value = sanitized.get(key)
        if is_valid_collected_value(
            key,
            value,
            team_size=sanitized.get("teamSize"),
        ):
            if key == "roles":
                snapshot[key] = _format_roles_for_backend(value)
            else:
                snapshot[key] = _strip_display_endings(key, value)

    for key in PUBLIC_AUXILIARY_COLLECTED_DATA_FIELDS:
        value = _normalize_auxiliary_value(key, sanitized.get(key))
        if value is not None:
            snapshot[key] = value

    goal = _clean_string(snapshot.get("goal"))
    subject = _clean_string(snapshot.get("subject"))
    if goal and subject and goal == subject:
        snapshot.pop("goal", None)

    return snapshot


def build_public_update_snapshot(
    updates: Mapping[str, object] | None,
    *,
    current_data: Mapping[str, object] | None = None,
) -> CollectedData:
    sanitized_updates = sanitize_candidate_updates(
        updates,
        current_data=current_data,
    )
    snapshot: CollectedData = {}

    for key in REQUIRED_COLLECTED_DATA_FIELDS:
        if key not in sanitized_updates:
            continue
        value = sanitized_updates.get(key)
        if is_valid_collected_value(
            key,
            value,
            team_size=sanitized_updates.get("teamSize", (current_data or {}).get("teamSize")),
        ):
            snapshot[key] = value

    for key in PUBLIC_AUXILIARY_COLLECTED_DATA_FIELDS:
        if key not in sanitized_updates:
            continue
        value = _normalize_auxiliary_value(key, sanitized_updates.get(key))
        if value is not None:
            snapshot[key] = value

    return snapshot
