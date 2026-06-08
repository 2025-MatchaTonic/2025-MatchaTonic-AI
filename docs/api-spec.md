# API Spec

기준 서버 실행:
`uvicorn main:app --reload --port 8000`

기본 주소 예시:
`http://127.0.0.1:8000`

## 공통 사항

- 모든 API 기본 응답 헤더는 `Content-Type: application/json`
- 현재 인증 헤더 또는 쿠키 기반 인증은 이 문서 기준으로 사용하지 않음
- Router prefix
  - `/ai/chat`
  - `/ai/template`
  - `/ai/project`

## `GET /`

서버 상태 확인 API

## Request

### Header

```json
X
```

### Body

```json
X
```

## Response

### Header

```json
Content-Type: application/json
```

### Body(200)

```json
{
  "message": "AI PM Server is running!"
}
```

### Body(500)

```json
{
  "detail": "Internal Server Error"
}
```

---

## `POST /ai/chat/`

일반 채팅, 탐색, 정보 수집, `@mates` 응답 처리 API

## Request

### Header

```json
Content-Type: application/json
```

### Body

```json
{
  "roomId": 123,
  "content": "@mates 지금 기준으로 우선순위 정리해줘",
  "actionType": "CHAT",
  "currentStatus": "TOPIC_SET",
  "collectedData": {
    "title": "회의록 자동 정리 서비스",
    "goal": "회의 요약과 액션 아이템을 빠르게 정리한다",
    "teamSize": "4명",
    "roles": "기획 1, 프론트 1, 백엔드 1, AI 1",
    "dueDate": "2026-03-31",
    "deliverables": "PRD, 화면 설계안, MVP 개발 범위 문서"
  },
  "recentMessages": [
    "회의 끝나고 정리 시간이 너무 오래 걸려.",
    "액션 아이템이 누락돼서 다시 확인하는 경우가 많아."
  ],
  "selectedMessage": "이번에는 MVP 범위를 명확하게 잡고 싶다."
}
```

## Response

### Header

```json
Content-Type: application/json
```

### Body(200)

```json
{
  "content": "AI 응답 텍스트",
  "suggestedQuestions": [],
  "currentStatus": "GATHER",
  "isSufficient": false,
  "collectedData": {
    "title": "회의록 자동 정리 서비스",
    "goal": "회의 요약과 액션 아이템을 빠르게 정리한다"
  },
  "notionTemplatePayload": null
}
```

### Body(422)

```json
{
  "detail": [
    {
      "loc": [
        "body",
        "roomId"
      ],
      "msg": "Field required",
      "type": "missing"
    }
  ]
}
```

### Body(500)

```json
{
  "detail": "AI 처리 중 오류 발생"
}
```

### 비고

- `actionType`
  - `CHAT`: 일반 채팅
  - `BTN_NO`: 주제 없음, 탐색 시작
  - `BTN_YES`: 주제 있음, 팀 대화 안내
  - `BTN_GO_DEF`: 주제 있음, 팀 대화 안내
  - `BTN_PLAN`: 기획용 템플릿 생성 경로
  - `BTN_DEV`: 개발용 템플릿 생성 경로
- `content`에 `@mates`가 포함되면 `actionType`보다 `@mates` 경로가 우선 적용됨
- `suggestedQuestions`는 현재 구현상 항상 빈 배열

---

## `POST /ai/template/`

채팅 완료 후 누적 `collectedData`와 최근 대화 기준으로 최종 템플릿 생성 API

## Request

### Header

```json
Content-Type: application/json
```

### Body

```json
{
  "roomId": 123,
  "templateType": "plan",
  "currentStatus": "READY",
  "content": "기획 템플릿 생성해줘. 팀이 바로 방향을 맞출 수 있게 정리해줘.",
  "collectedData": {
    "title": "회의록 자동 정리 서비스",
    "goal": "회의 요약과 액션 아이템을 빠르게 정리한다",
    "teamSize": "4명",
    "roles": "기획 1, 프론트 1, 백엔드 1, AI 1",
    "dueDate": "2026-03-31",
    "deliverables": "PRD, 화면 설계안, MVP 개발 범위 문서"
  },
  "recentMessages": [
    "회의 끝나고 정리 시간이 너무 오래 걸려.",
    "액션 아이템이 누락돼서 다시 확인하는 경우가 많아.",
    "이번에는 MVP 범위를 명확하게 잡고 싶다."
  ],
  "selectedMessage": "이번에는 MVP 범위를 명확하게 잡고 싶다."
}
```

## Response

### Header

```json
Content-Type: application/json
```

### Body(200)

```json
{
  "content": "기획용 템플릿 초안을 생성했습니다.",
  "currentStatus": "DONE",
  "notionTemplatePayload": {
    "projectId": 123,
    "templates": [
      {
        "key": "PROJECT_HOME",
        "parentKey": null,
        "title": "Project Home",
        "content": {
          "project_overview": "내용"
        }
      },
      {
        "key": "PLANNING",
        "parentKey": "PROJECT_HOME",
        "title": "기획",
        "content": {
          "project_intro": "프로젝트 한 줄 소개",
          "problem_definition": [
            {
              "id": 1,
              "situation": "불편한 상황 (상황·경험 중심)",
              "reason": "왜 문제인가?",
              "limitation": "기존 해결 방식의 한계"
            }
          ],
          "solution": {
            "core_summary": "핵심 솔루션 한 줄 요약",
            "problem_solutions": [
              {
                "problem_id": 1,
                "solution_desc": "문제 1에 대한 우리 서비스의 해결 방식"
              }
            ],
            "features": [
              "우리 솔루션의 특징 1",
              "우리 솔루션의 특징 2",
              "우리 솔루션의 특징 3"
            ]
          },
          "target_persona": {
            "name": "이름(가명)",
            "age": "나이",
            "job_role": "직업 / 역할",
            "main_activities": "주요 활동",
            "pain_points": [
              "불편함 1",
              "불편함 2"
            ],
            "needs": [
              "니즈 1",
              "니즈 2"
            ]
          }
        }
      },
      {
        "key": "GROUND_RULES",
        "parentKey": "PROJECT_HOME",
        "title": "그라운드룰",
        "content": " "
      }
    ]
  }
}
```

### Body(400)

```json
{
  "detail": {
    "message": "템플릿 생성은 READY 상태에서만 가능합니다.",
    "currentStatus": "GATHER"
  }
}
```

```json
{
  "detail": {
    "message": "필수 collectedData가 모두 채워져야 템플릿을 생성할 수 있습니다.",
    "missingFields": [
      {
        "key": "goal",
        "label": "프로젝트 목표"
      }
    ]
  }
}
```

### Body(422)

```json
{
  "detail": [
    {
      "loc": [
        "body",
        "templateType"
      ],
      "msg": "Input should be 'plan' or 'dev'",
      "type": "literal_error"
    }
  ]
}
```

### Body(500)

```json
{
  "detail": "템플릿 생성 중 오류가 발생했습니다."
}
```

### 비고

- `templateType`
  - `plan`: 기획용 템플릿
  - `dev`: 개발용 템플릿
- `currentStatus`는 반드시 `READY`
- 필수 `collectedData`
  - `title`
  - `goal`
  - `teamSize`
  - `roles`
  - `dueDate`
  - `deliverables`

---

## `GET /ai/project/steps`

PM 진행 단계 정의 조회 API

## Request

### Header

```json
X
```

### Body

```json
X
```

## Response

### Header

```json
Content-Type: application/json
```

### Body(200)

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

### Body(500)

```json
{
  "detail": "Internal Server Error"
}
```

---

## `GET /ai/project/manual`

팀 공통 규칙 및 체크리스트 조회 API

## Request

### Header

```json
X
```

### Body

```json
X
```

## Response

### Header

```json
Content-Type: application/json
```

### Body(200)

```json
{
  "team_common_rules": [
    "Every decision must reference user problem or metric.",
    "Limit one topic per discussion turn.",
    "Track unresolved items explicitly."
  ],
  "pm_thinking_checklist": [
    "Why now and for whom?",
    "What is the minimum testable value?",
    "What is the riskiest assumption this week?"
  ]
}
```

### Body(500)

```json
{
  "detail": "Internal Server Error"
}
```

---

## `GET /ai/project/templates`

문장 템플릿 조회 API

## Request

### Header

```json
X
```

### Body

```json
X
```

## Response

### Header

```json
Content-Type: application/json
```

### Body(200)

```json
{
  "problem_statement": {
    "template": "For [target user], [current pain] causes [negative impact].",
    "example": "For first-time project teams, unclear scope causes late delivery."
  },
  "mvp_scope": {
    "template": "In-scope: [3 bullets], Out-of-scope: [3 bullets].",
    "example": "In-scope: login, task board, progress update."
  },
  "role_card": {
    "template": "[Role] owns [deliverable] and reviews [handoff artifact].",
    "example": "PM owns scope notes and reviews the implementation handoff document."
  }
}
```

### Body(500)

```json
{
  "detail": "Internal Server Error"
}
```

---

## 현재 라우팅 요약

- `@mates` 포함 메시지: `mates_node`
- `BTN_NO`: `explore_node`
- `BTN_YES`, `BTN_GO_DEF`: `topic_exists_node`
- `BTN_PLAN`: `generate_plan_node`
- `BTN_DEV`: `generate_dev_node`
- `CHAT + EXPLORE`: `explore_node`
- `CHAT + TOPIC_SET | GATHER | READY`: `gather_node`
