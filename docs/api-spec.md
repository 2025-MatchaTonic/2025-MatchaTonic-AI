# API 명세서 (FastAPI)

기준 서버: `uvicorn main:app --reload --port 8000`  
기본 주소 예시: `http://localhost:8000`

## 1) 공통
- Content-Type: `application/json`
- CORS: 전체 허용(`*`)
- 라우터 prefix
  - Chat: `/ai/chat`
  - Project: `/ai/project`

## 2) 헬스 체크

### `GET /`
- 설명: 서버 상태 확인
- 응답 예시:
```json
{
  "message": "AI PM Server is running!"
}
```

## 3) AI Chat API

### `POST /ai/chat/`
- 설명: Spring(또는 클라이언트)에서 AI 대화 처리 요청
- Request Body:
```json
{
  "roomId": 123,
  "content": "@mates 우선순위 정리해줘",
  "actionType": "CHAT",
  "currentStatus": "TOPIC_SET",
  "collectedData": {
    "title": "프로젝트명",
    "goal": "목표",
    "teamSize": "4",
    "roles": "PM, FE, BE, QA",
    "dueDate": "2026-04-01",
    "deliverables": "기획서, PRD"
  },
  "recentMessages": [
    "백로그가 너무 많아요",
    "이번 주 목표가 불명확해요"
  ],
  "selectedMessage": "핵심 문제: 우선순위 합의가 안 됨"
}
```

- 필드 설명:
  - `roomId` (int): 프로젝트/대화 식별자
  - `content` (str): 사용자 최신 메시지
  - `actionType` (str): 이벤트 타입
  - `currentStatus` (str): 현재 단계(클라이언트 관리)
  - `collectedData` (object): 누적 수집 데이터
  - `recentMessages` (string[]): 최근 팀 대화
  - `selectedMessage` (string|null): 사용자 지정 핵심 메시지

- 주요 `actionType` 값:
  - `CHAT`: 일반 채팅
  - `BTN_NO`: 주제 없음(탐색 모드)
  - `BTN_YES`, `BTN_GO_DEF`: 주제 있음
  - `BTN_PLAN`, `BTN_DEV`: 템플릿 생성 트리거

- Response Body:
```json
{
  "content": "AI 응답 텍스트",
  "suggestedQuestions": [],
  "currentStatus": "GATHER",
  "isSufficient": false,
  "collectedData": {
    "title": "프로젝트명",
    "goal": "목표"
  },
  "notionTemplatePayload": null
}
```

- 응답 필드 설명:
  - `content` (str): 사용자에게 보여줄 AI 응답
  - `suggestedQuestions` (string[]): 추천 질문(현재 구현은 빈 배열)
  - `currentStatus` (str): 다음 단계
  - `isSufficient` (bool): 템플릿 생성 가능 여부
  - `collectedData` (object): 병합된 최신 데이터
  - `notionTemplatePayload` (object|null): 템플릿 생성 시 payload

- 템플릿 생성 시 `notionTemplatePayload` 예시:
```json
{
  "projectId": 123,
  "templates": [
    {
      "key": "PROJECT_HOME",
      "parentKey": null,
      "title": "Project Home",
      "content": {
        "project_overview": "..."
      }
    },
    {
      "key": "PLANNING",
      "parentKey": "PROJECT_HOME",
      "title": "기획",
      "content": {
        "project_intro": "...",
        "problem_definition": [],
        "solution": {},
        "target_persona": {}
      }
    },
    {
      "key": "GROUND_RULES",
      "parentKey": "PROJECT_HOME",
      "title": "그라운드룰",
      "content": "..."
    }
  ]
}
```

- 오류 응답:
  - `500 Internal Server Error`
```json
{
  "detail": "AI 처리 중 오류 발생"
}
```

## 4) Project API

### `GET /ai/project/steps`
- 설명: PM 진행 단계 정의 반환
- 응답 타입: `StepInfo[]`
```json
[
  {
    "step": 1,
    "key": "problem_definition",
    "title": "Problem Definition",
    "done_criteria": [
      "target user identified",
      "core problem sentence validated",
      "success metric draft exists"
    ]
  }
]
```

### `GET /ai/project/manual`
- 설명: 팀 공통 룰/체크리스트 반환
- 응답 타입: `Dict[str, List[str]]`
```json
{
  "team_common_rules": [
    "Every decision must reference user problem or metric."
  ],
  "pm_thinking_checklist": [
    "Why now and for whom?"
  ]
}
```

### `GET /ai/project/templates`
- 설명: 문장 템플릿 샘플 반환
- 응답 타입: `Dict[str, Dict[str, str]]`
```json
{
  "problem_statement": {
    "template": "For [target user], [current pain] causes [negative impact].",
    "example": "For first-time project teams, unclear scope causes late delivery."
  }
}
```

## 5) 상태 전이 규칙 (요약)
- `@mates` 포함 메시지: `mates_node`
- `BTN_NO`: `explore_node`
- `BTN_YES`/`BTN_GO_DEF`: `topic_exists_node`
- `BTN_PLAN`/`BTN_DEV`: `generate_node`
- `CHAT + EXPLORE`: `explore_node`
- `CHAT + TOPIC_SET|GATHER|READY`: `gather_node`

## 6) collectedData 권장 키
- `title`
- `goal`
- `teamSize`
- `roles`
- `dueDate`
- `deliverables`
