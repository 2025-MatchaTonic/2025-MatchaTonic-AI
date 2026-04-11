import re

SHORT_MESSAGE_PATTERN = re.compile(r"[^a-z0-9가-힣]+")
BUTTON_ONLY_PATTERN = re.compile(r"[\s\.\,\!\?]+")
TRAILING_TOPIC_ENDINGS_PATTERN = re.compile(
    r"(이에요|예요|입니다|이요|요|입니다요|하고 싶어요|하려고 해요|생각 중이에요|같아요)$"
)
QUESTION_LINE_ENDING_PATTERN = re.compile(
    r"(\?|？)\s*$|"
    r"(인가요|있나요|없나요|어떤가요|뭔가요|무엇인가요|왜인가요|어떨까요|할까요|볼까요|나요|까요)\s*$"
)

TRIVIAL_MESSAGES = {
    "",
    "hi",
    "hello",
    "hey",
    "yo",
    "안녕",
    "안녕하세요",
    "하이",
    "헬로",
    "ㅎㅇ",
    "ㅁㅌ",
    "mates",
}
GREETING_TOKENS = {
    "hi",
    "hello",
    "hey",
    "\uc548\ub155",
    "\uc548\ub155\ud558\uc138\uc694",
    "\ubc18\uac00\uc6cc",
    "\ubc18\uac11\uc2b5\ub2c8\ub2e4",
}

TEAM_SIZE_GENERIC_PATTERN = re.compile(r"(?<!\d)(\d{1,2})\s*명(?!\d)")
ROLE_SPLIT_PATTERN = re.compile(r"\s*(?:,|/|및|그리고|와|과)\s*")
ROLE_PREFIX_PATTERN = re.compile(
    r"^\s*(?:역할|역할은|담당|담당은|구성|구성은|멤버|멤버는)\s*[:은는이가]?\s*"
)
ROLE_TRAILING_SPLIT_PATTERN = re.compile(
    r"\s*(?:이렇게|정도로|정할게|정할 거|정할|나눌 거|나눌게|나눌|나눠|운영|하려고|할게|세부|포지션)"
)

SUMMARY_REQUEST_KEYWORDS = (
    "요약",
    "정리해줘",
    "정리해 줘",
    "지금 모인 정보",
    "확정된 사항",
    "확정된 정보",
    "정리된 상황",
    "정해진 사항",
    "현재 확정",
    "현재까지",
    "세션 요약",
)
TEAM_SIZE_HINT_KEYWORDS = ("팀", "인원", "멤버", "우리", "총", "전체")
ROLE_TOKEN_HINTS = (
    "개발자",
    "개발",
    "기획자",
    "기획",
    "pm",
    "po",
    "디자이너",
    "디자인",
    "백엔드",
    "프론트엔드",
    "프론트",
    "서버",
    "ios",
    "android",
    "안드로이드",
    "데이터",
    "ai",
)

GATHER_FIELD_GUIDE = {
    "subject": {
        "label": "프로젝트 주제",
        "question": "어떤 분야나 문제 영역을 다루는 프로젝트인지 한 줄로 정리하면 무엇인가요?",
    },
    "title": {
        "label": "프로젝트 제목",
        "question": "프로젝트 제목을 한 줄로 어떻게 정리하면 될까요?",
    },
    "goal": {
        "label": "프로젝트 목표",
        "question": "이 프로젝트로 팀이 최종적으로 만들고 싶은 결과를 한 줄로 말하면 무엇인가요?",
    },
    "teamSize": {
        "label": "팀 인원",
        "question": "현재 이 프로젝트를 함께하는 팀원은 몇 명인가요?",
    },
    "roles": {
        "label": "역할",
        "question": "팀원 역할 분담은 어떻게 가져갈 생각인가요? 아직 미정이면 필요한 역할만 말해도 됩니다.",
    },
    "dueDate": {
        "label": "마감일",
        "question": "중간발표나 최종제출 기준으로 생각하는 마감일은 언제인가요?",
    },
    "deliverables": {
        "label": "산출물",
        "question": "최종적으로 제출하거나 보여줘야 하는 산출물은 무엇인가요?",
    },
}

UNSUPPORTED_GATHER_TOPICS = {
    "targetUser": {
        "label": "혜택 대상",
        "instruction": (
            "최근 대화의 초점은 혜택 대상입니다. 사용자의 답변이나 질문에 먼저 반응하되 "
            "이 정보는 collected_data의 기존 키로 억지로 저장하지 마세요."
        ),
    },
    "importance": {
        "label": "중요한 이유",
        "instruction": (
            "최근 대화의 초점은 문제의 중요성입니다. 먼저 이유를 설명하거나 정리해 주고, "
            "이 정보는 collected_data의 기존 키로 억지로 저장하지 마세요."
        ),
    },
}

GATHER_FOCUS_KEYWORDS = {
    "subject": ("주제", "분야", "문제 영역", "큰 방향"),
    "title": ("제목", "프로젝트명", "서비스명", "이름"),
    "goal": ("목표", "무엇을 만들", "최종적으로 만들", "해결하려는 문제"),
    "teamSize": ("몇 명", "팀원", "인원", "팀 규모"),
    "roles": ("역할", "역할 분담", "담당", "누가 맡"),
    "dueDate": ("마감", "마감일", "언제까지", "제출", "발표", "데드라인"),
    "deliverables": ("산출물", "결과물", "제출물", "무엇을 제출", "최종 산출"),
    "targetUser": ("누가 혜택", "대상 사용자", "누가 쓰", "누구를 위한", "누가 받"),
    "importance": ("왜 중요", "왜 필요한", "이유", "왜 문제", "중요한가"),
}

HELP_REQUEST_KEYWORDS = (
    "추천",
    "예시",
    "후보",
    "뭐가 좋",
    "어떻게",
    "왜",
    "이유",
    "설명",
    "알려",
    "모르겠",
    "도와",
)
GUIDANCE_SIGNAL_PATTERNS = (
    re.compile(r"잘\s*모르겠", re.IGNORECASE),
    re.compile(r"모르겠", re.IGNORECASE),
    re.compile(r"도와\s*(?:줘|주세요|주라)?", re.IGNORECASE),
    re.compile(r"추천해\s*(?:줘|주세요|주라)?", re.IGNORECASE),
    re.compile(r"정해\s*(?:줘|주세요|주라)?", re.IGNORECASE),
    re.compile(r"같이\s*(?:정하|골라|좁혀)", re.IGNORECASE),
    re.compile(r"고민\s*중", re.IGNORECASE),
    re.compile(r"감이\s*안", re.IGNORECASE),
    re.compile(r"(?:뭘|뭐를|무엇을|어떤\s*걸?)\s*(?:해야|만들어야|하고\s*싶은지)", re.IGNORECASE),
)
META_CONVERSATION_PATTERNS = (
    re.compile(r"^\s*(?:아니|아니야|아뇨)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:그게\s+아니라|다시|잠깐)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:엥|에엥|어|응)\s*(?:\?|!|\.|$)", re.IGNORECASE),
    re.compile(r"^\s*(?:엥|에엥)\s*(?:무슨|뭔)\s*소리", re.IGNORECASE),
    re.compile(r"^\s*(?:무슨|뭔)\s*(?:말|소리)", re.IGNORECASE),
    re.compile(r"^\s*(?:이상한데|이상해|말이\s*안\s*되|이해가\s*안)", re.IGNORECASE),
)

TITLE_INSTRUCTION_KEYWORDS = (
    "넣어",
    "추천",
    "알려",
    "해줘",
    "보여",
    "데이터",
    "후보",
)
FILL_REMAINING_EXACT_KEYWORDS = (
    "나머지 다 확정해줘",
    "나머지 확정해줘",
    "나머지 다 정해줘",
    "나머지 다 지정해줘",
    "나머지 다 채워줘",
    "남은 것들 다 확정해줘",
    "전부 확정해줘",
    "다 확정해줘",
    "다 정해줘",
    "다 지정해줘",
    "다 채워줘",
    "전부 정해줘",
)
FILL_REMAINING_TRIGGER_KEYWORDS = (
    "확정",
    "정해",
    "지정",
    "채워",
    "반영",
    "업데이트",
    "저장",
    "기록",
    "맞춰",
    "완성",
)
FILL_REMAINING_SCOPE_KEYWORDS = (
    "나머지",
    "남은",
    "전부",
    "전체",
    "모두",
    "세션 요약",
    "세션요약",
    "핵심 결정사항",
    "저 부분",
    "위 내용",
    "방금 내용",
    "이걸로",
)

TITLE_EXPLICIT_PATTERN = re.compile(
    r"^\s*(?:프로젝트\s*주제|프로젝트명|주제|제목|이름)\s*(?:은|는|:)?\s*",
    re.IGNORECASE,
)
TEAM_SIZE_PATTERN = re.compile(r"(?:팀\s*인원|팀원|인원)\D{0,8}(\d{1,2})\s*명?")
ROLE_PATTERN = re.compile(r"(?:역할|롤|role)\s*(?:은|는|:)?\s*(.+)$", re.IGNORECASE)
DUE_DATE_PATTERN = re.compile(
    r"(?:마감(?:일)?|데드라인|due)\s*(?:은|는|:)?\s*"
    r"([0-9]{4}[./-][0-9]{1,2}[./-][0-9]{1,2}|[0-9]{1,2}월\s*[0-9]{1,2}일)",
    re.IGNORECASE,
)
GOAL_PATTERN = re.compile(r"(?:목표|goal)\s*(?:은|는|:)?\s*(.+)$", re.IGNORECASE)
SUBJECT_PATTERN = re.compile(r"(?:주제|subject|topic)\s*(?:은|는|:)?\s*(.+)$", re.IGNORECASE)
DELIVERABLES_PATTERN = re.compile(
    r"(?:산출물|결과물|제출물|deliverable[s]?)\s*(?:은|는|:)?\s*(.+)$",
    re.IGNORECASE,
)
CHOICE_INDEX_PATTERN = re.compile(r"^\s*([1-9])\s*(?:번)?\s*$")
CHOICE_PREFIX_PATTERN = re.compile(r"^\s*([1-9])\s*번(?:\s*.*)?$")
NUMBERED_OPTION_LINE_PATTERN = re.compile(r"^\s*(\d{1,2})[)\.\-:]\s*(.+?)\s*$")
NUMBERED_OPTION_INLINE_PATTERN = re.compile(
    r"(\d{1,2})[)\.\-:]\s*(.+?)(?=\s+\d{1,2}[)\.\-:]|$)"
)
DIRECT_FACT_ENDING_PATTERN = re.compile(r"\s*(?:입니다|이에요|예요|이야|야|요)\s*$")
KOREAN_DUE_DATE_CANDIDATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(\d{2,4}년\s*(?:초|중|말))"),
    re.compile(r"(내년\s*(?:초|중|말))"),
    re.compile(r"(올해\s*(?:초|중|말))"),
    re.compile(r"(다음\s*달\s*(?:초|중|말))"),
    re.compile(r"(\d{1,2}월\s*말(?:쯤)?)"),
    re.compile(r"(이번\s*학기\s*말)"),
    re.compile(r"(중간발표\s*전)"),
    re.compile(r"(최종발표\s*전)"),
)
PROBLEM_AREA_PATTERN = re.compile(r"(.+?)(?:하는\s*문제|문제|쪽|관점|방향)(?:로|을|를)?(?:\s|$)")
TARGET_FACILITY_PROMPT_PATTERN = re.compile(r"어떤\s+시설을\s+대상으로\s+하나요")
TARGET_FACILITY_NOUN_PATTERN = re.compile(
    r"(도서관|공원|주민센터|버스터미널|체육관|체육시설|복지관|주차장|공공화장실|박물관|미술관|학교|강의실|병원|보건소)"
)
PROBLEM_AREA_CONTEXT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"좋아요\.\s*(.+?)의\s+(.+?)\s+문제로\s+좁혀볼게요"),
    re.compile(r"좋아요\.\s*(.+?)에서\s+'(.+?)'\s+방향으로\s+좁혀볼게요"),
)

FAST_RAG_PHASES = {"EXPLORE", "TOPIC_SET", "GATHER"}
RAG_FILTERS_BY_PHASE: dict[str, dict[str, list[str]]] = {
    "EXPLORE": {
        "topics": ["value_proposition", "team_playbook"],
        "doc_types": ["reference", "playbook"],
    },
    "TOPIC_SET": {
        "topics": ["design_sprint", "team_playbook", "scrum_guide"],
        "doc_types": ["reference", "playbook", "guide"],
    },
    "GATHER": {
        "topics": [
            "value_proposition",
            "design_sprint",
            "team_playbook",
            "scrum_guide",
        ],
        "doc_types": ["reference", "playbook", "guide"],
    },
    "READY_PLAN": {
        "topics": ["scrum_guide", "team_playbook", "design_sprint"],
        "doc_types": ["guide", "playbook", "reference"],
    },
    "READY_DEV": {
        "topics": ["api_design", "software_engineering_standard", "scrum_guide"],
        "doc_types": ["reference", "guide"],
    },
}
