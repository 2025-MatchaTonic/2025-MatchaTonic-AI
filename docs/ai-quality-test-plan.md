# AI Quality Test Plan

## 목적
- 채팅 응답 품질을 케이스별로 반복 검증한다.
- `explore`, `gather`, `@mates`, `template generation`이 의도대로 동작하는지 확인한다.
- RAG가 맞는 문서군을 참고하는지 간접적으로 점검한다.

## 테스트 전 준비
1. 서버 실행
```bash
uvicorn main:app --reload --port 8000
```
2. Swagger 열기
- `http://127.0.0.1:8000/docs`
3. 기본 확인
- `GET /` 호출 시 `"AI PM Server is running!"` 응답

## 공통 품질 기준
- 답변이 너무 딱딱하지 않고 초보 팀도 이해할 수 있는 쉬운 한국어여야 한다.
- RAG 전문용어를 그대로 복붙하지 않고, 필요한 경우 짧게 풀어서 설명해야 한다.
- 최근 대화 문맥을 반영해야 한다.
- 없는 사실을 확정형으로 지어내지 않아야 한다.
- `@mates` 답변은 문장 중간에서 잘리지 않아야 한다.

## 1. Explore 품질 테스트

### 목적
- 주제가 없을 때 질문을 한 번에 하나만 던지는지 확인한다.

### API
- `POST /ai/chat/`

### Request Body
```json
{
  "roomId": 1001,
  "content": "아직 아이템이 없어서 뭐부터 생각해야 할지 모르겠어",
  "actionType": "BTN_NO",
  "currentStatus": "INIT",
  "collectedData": {},
  "recentMessages": [],
  "selectedMessage": null
}
```

### 기대 결과
- `currentStatus`가 `EXPLORE`
- 질문이 정확히 1개
- 딱딱한 분석 문장이 아니라 탐색형 질문
- 과도한 전문용어 없음

### 실패 신호
- 질문을 2개 이상 던짐
- 너무 긴 설명만 하고 질문이 없음
- 당장 템플릿 생성 쪽으로 몰고 감

### 실제 결과
{
  "content": "최근 일주일 동안 '아, 이거 진짜 귀찮다' 했던 적이 있나요?",
  "suggestedQuestions": [],
  "currentStatus": "EXPLORE",
  "isSufficient": false,
  "collectedData": {},
  "notionTemplatePayload": null
}

## 2. Gather 품질 테스트

### 목적
- `collectedData`의 빈 값을 하나씩 자연스럽게 채우는지 확인한다.

### API
- `POST /ai/chat/`

### Request Body
```json
{
  "roomId": 1002,
  "content": "제목은 회의록 자동 정리 서비스고, 팀은 4명이야",
  "actionType": "CHAT",
  "currentStatus": "GATHER",
  "collectedData": {
    "title": "",
    "goal": "",
    "teamSize": "",
    "roles": "",
    "dueDate": "",
    "deliverables": ""
  },
  "recentMessages": [
    "회의 끝나고 정리 시간이 너무 오래 걸려.",
    "액션 아이템이 자꾸 누락돼."
  ],
  "selectedMessage": "액션 아이템이 자꾸 누락돼."
}
```

### 기대 결과
- `collectedData.title`, `collectedData.teamSize`가 채워짐
- 답변은 후속 질문 1개만 포함
- 최근 대화를 보고 `goal`이나 문제 맥락을 자연스럽게 반영
- `isSufficient`는 아직 `false`

### 실제 결과
{
  "content": "좋아요 — 제목은 '회의록 자동 정리 서비스', 팀은 4명이군요. 그럼 역할은 어떻게 배분되어 있나요? 각자 맡은 직무를 간단히 알려주실래요? (예: PM 1명, 개발 2명, 디자이너 1명)",
  "suggestedQuestions": [],
  "currentStatus": "GATHER",
  "isSufficient": false,
  "collectedData": {
    "title": "회의록 자동 정리 서비스",
    "goal": "",
    "teamSize": "4명",
    "roles": "",
    "dueDate": "",
    "deliverables": ""
  },
  "notionTemplatePayload": null
}

### 실패 신호
- 한 번에 여러 필드를 묻는 긴 설문형 응답
- 이미 받은 정보를 다시 물음
- 최근 대화와 무관한 질문

## 3. @mates 품질 테스트

### 목적
- 최근 대화 기준으로 실무형 답변을 주는지, 문장이 잘리지 않는지 확인한다.

### API
- `POST /ai/chat/`

### Request Body
```json
{
  "roomId": 1003,
  "content": "@mates 지금 기준으로 우선순위랑 다음 액션 3개만 정리해줘",
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
    "액션 아이템이 누락돼서 다시 확인하는 경우가 많아.",
    "이번에는 MVP 범위를 명확하게 잡고 싶다.",
    "구현 전에 역할하고 범위 먼저 고정해야 해."
  ],
  "selectedMessage": "구현 전에 역할하고 범위 먼저 고정해야 해."
}
```

### 기대 결과
- 우선순위와 다음 액션 중심 답변
- 최근 대화 내용을 실제로 반영
- 문장이 중간에 끊기지 않음
- 고객센터식 `"무엇을 도와드릴까요?"` 금지

### 실제 결과
{
  "content": "우선순위(중요한 순): 1) MVP 범위 확정, 2) 역할·책임 고정, 3) 스프린트 백로그로 작업 분해.  \n1) 오늘 30분: PO 주도 'MVP 범위 잠금' 미팅 — 핵심 화면/기능 최대 5개와 우선순위 표로 확정(아웃풋: 확정 리스트).  \n2) 오늘~내일 오전: 팀별 역할·책임 문서화(개발/디자인/QA별 소유 항목과 '완료 기준'(작동 + 기본 테스트) 명시).  \n3) 스프린트 준비 세션(1시간): 각 백로그 항목을 하루 단위 작업으로 쪼개고 담당자·마감일 배정 후 데일리 체크리스트로 등록.",
  "suggestedQuestions": [],
  "currentStatus": "TOPIC_SET",
  "isSufficient": false,
  "collectedData": {
    "title": "회의록 자동 정리 서비스",
    "goal": "회의 요약과 액션 아이템을 빠르게 정리한다",
    "teamSize": "4명",
    "roles": "기획 1, 프론트 1, 백엔드 1, AI 1",
    "dueDate": "2026-03-31",
    "deliverables": "PRD, 화면 설계안, MVP 개발 범위 문서"
  },
  "notionTemplatePayload": null
}

### 실패 신호
- 최근 대화 복기 없이 일반론만 답함
- 문장 중간이 끊김
- 불필요한 섹션 헤더, 장황한 설명

## 4. Plan Template 품질 테스트

### 목적
- 기획용 템플릿이 사용자 문제와 팀 정렬 중심으로 생성되는지 확인한다.

### API
- `POST /ai/template/`

### Request Body
```json
{
  "roomId": 1004,
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

### 기대 결과
- `currentStatus`가 `DONE`
- `notionTemplatePayload.templates` 길이가 3
- `PLANNING.content.problem_definition`이 사용자 문제 중심
- 너무 기술 구현 중심으로 치우치지 않음
- 쉬운 한국어 사용

### 실제 결과
{
  "content": "MVP 범위와 산출물을 마감(2026-03-31) 기준으로 한눈에 정리한 실행 템플릿입니다.",
  "currentStatus": "DONE",
  "notionTemplatePayload": {
    "projectId": 1004,
    "templates": [
      {
        "key": "PROJECT_HOME",
        "parentKey": null,
        "title": "Project Home",
        "content": {
          "project_overview": "회의록 자동 정리 서비스입니다. 목표는 회의 요약과 액션 아이템을 빠르게 정리하는 것입니다. 팀 규모는 4명이며 역할은 기획 1, 프론트 1, 백엔드 1, AI 1입니다. 주요 산출물은 PRD, 화면 설계안, MVP 개발 범위 문서이며, 마감일은 2026-03-31입니다."
        }
      },
      {
        "key": "PLANNING",
        "parentKey": "PROJECT_HOME",
        "title": "기획",
        "content": {
          "project_intro": "회의록 자동 정리 서비스: 회의 요약과 액션 아이템을 빠르게 정리합니다",
          "problem_definition": [
            {
              "id": 1,
              "situation": "회의 후 정리 시간이 오래 걸리고, 액션 아이템이 누락되는 일이 반복됩니다.",
              "reason": "정리 기준과 역할 분담이 문서로 명확히 정리되어 있지 않아 우선순위와 실행 주체가 불명확해집니다.",
              "limitation": "현재 수집된 정보로는 세부 사용자 시나리오나 기능 우선순위를 정확히 확정할 수 없습니다."
            }
          ],
          "solution": {
            "core_summary": "목표·역할·산출물·마감일을 기준으로 팀이 바로 실행할 수 있는 템플릿을 제공합니다.",
            "problem_solutions": [
              {
                "problem_id": 1,
                "solution_desc": "팀 구성(기획1, 프론트1, 백엔드1, AI1), 산출물(PRD·화면 설계안·MVP 범위 문서), 마감일(2026-03-31)을 기준으로 책임과 핵심 산출물을 정리해 회의 후 바로 공유하고 실행할 수 있게 합니다."
              }
            ],
            "features": [
              "확정된 산출물 중심 정리: PRD, 화면 설계안, MVP 개발 범위 문서를 기준으로 템플릿을 구성합니다.",
              "역할 기반 책임 명시: 기획 1, 프론트 1, 백엔드 1, AI 1로 각 산출물과 작업 책임을 간결하게 표기합니다.",
              "마감 기준 중심 우선순위: 마감일(2026-03-31)을 기준으로 제출 우선순위와 확인 항목을 명확히 합니다."
            ]
          },
          "target_persona": {
            "name": "추가 논의 필요",
            "age": "추가 논의 필요",
            "job_role": "기획 1, 프론트 1, 백엔드 1, AI 1",
            "main_activities": "PRD 작성, 화면 설계, MVP 개발 범위 정의 및 산출물 검토",
            "pain_points": [
              "회의 후 정리에 시간이 많이 소요됩니다.",
              "액션 아이템이 누락되거나 실행 주체가 불명확합니다."
            ],
            "needs": [
              "2026-03-31까지 공유하고 실행할 수 있는 명확한 실행 템플릿이 필요합니다.",
              "각 산출물과 역할을 한눈에 확인할 수 있는 문서가 필요합니다."
            ]
          }
        }
      },
      {
        "key": "GROUND_RULES",
        "parentKey": "PROJECT_HOME",
        "title": "그라운드룰",
        "content": "1. 모든 결정은 수집된 데이터(목표, 역할, 산출물, 마감일)를 기준으로 문서화합니다.\n2. 산출물( PRD, 화면 설계안, MVP 범위 문서)은 마감일 우선으로 정리하고, 책임자를 명시합니다.\n3. 회의에서 나온 액션 아이템은 템플릿의 '실행 항목' 필드에 즉시 기록하고 담당·기한을 배정합니다.\n4. 세부 사용자 시나리오나 기능 우선순위는 추가 논의 후 문서로 확정합니다.\n5. 문서는 모든 팀원이 이해할 수 있도록 짧고 명확한 문장으로 작성합니다."
      }
    ]
  }
}

### 실패 신호
- 개발 작업 목록 같은 표현이 중심이 됨
- `job_role`에 팀 전체 구성이 그대로 들어감
- 과도하게 구체적인 가짜 정보 생성

## 5. Dev Template 품질 테스트

### 목적
- 개발용 템플릿이 역할, 구현 범위, 실행 기준 중심으로 생성되는지 확인한다.

### API
- `POST /ai/template/`

### Request Body
```json
{
  "roomId": 1005,
  "templateType": "dev",
  "currentStatus": "READY",
  "content": "개발 템플릿 생성해줘. 구현 범위와 역할 기준이 바로 보이게 정리해줘.",
  "collectedData": {
    "title": "회의록 자동 정리 서비스",
    "goal": "회의 요약과 액션 아이템을 빠르게 정리한다",
    "teamSize": "4명",
    "roles": "기획 1, 프론트 1, 백엔드 1, AI 1",
    "dueDate": "2026-03-31",
    "deliverables": "PRD, 화면 설계안, MVP 개발 범위 문서"
  },
  "recentMessages": [
    "구현 전에 역할하고 범위 먼저 고정해야 해.",
    "API 설계 기준이 있으면 좋겠어.",
    "개발 시작 전에 범위와 산출물부터 묶자."
  ],
  "selectedMessage": "구현 전에 역할하고 범위 먼저 고정해야 해."
}
```

### 기대 결과
- plan 결과보다 실행 기준, 역할, 범위가 더 강조됨
- `solution.features`가 구현/운영 기준 쪽으로 기울어짐
- 개발용인데도 과도한 기술 전문용어 남발은 없음

### 실제 결과
{
  "content": "목표·역할·산출물·마감 기준을 한 문서로 정리해 구현 전 합의하세요.",
  "currentStatus": "DONE",
  "notionTemplatePayload": {
    "projectId": 1005,
    "templates": [
      {
        "key": "PROJECT_HOME",
        "parentKey": null,
        "title": "Project Home",
        "content": {
          "project_overview": "회의록 자동 정리 서비스 개발 실행 초안입니다. 목표는 회의 요약과 액션 아이템을 빠르게 정리하는 것입니다. 팀 규모는 4명(기획 1, 프론트 1, 백엔드 1, AI 1)입니다. 주요 산출물은 PRD, 화면 설계안, MVP 개발 범위 문서이며 마감 기준은 2026-03-31입니다."
        }
      },
      {
        "key": "PLANNING",
        "parentKey": "PROJECT_HOME",
        "title": "기획",
        "content": {
          "project_intro": "회의록 자동 정리 서비스: 회의 요약과 액션 아이템을 빠르게 정리합니다.",
          "problem_definition": [
            {
              "id": 1,
              "situation": "개발 범위와 역할, 산출물 연결 방식이 한 문서로 정리되어 있지 않으면 구현 우선순위와 핸드오프가 불명확해집니다.",
              "reason": "팀이 4명으로 작고 마감일이 정해져 있으므로 역할과 실행 기준을 미리 고정해야 속도와 품질을 유지할 수 있습니다.",
              "limitation": "현재 정보로는 기술 스택, API 상세 명세, 데이터 모델, 세부 작업 단위는 확정할 수 없습니다."
            }
          ],
          "solution": {
            "core_summary": "목표·역할·산출물·마감 기준을 바탕으로 개발 범위와 책임을 단일 문서로 정리해 구현 전 합의합니다.",
            "problem_solutions": [
              {
                "problem_id": 1,
                "solution_desc": "기획 1, 프론트 1, 백엔드 1, AI 1 역할을 기준으로 PRD·화면 설계안·MVP 개발 범위 문서에 각 책임과 산출물 연결을 명시합니다. 구현 전 공통 기준(예: API 설계 체크리스트 초안, 항목별 완료 기준)을 팀 합의로 확정하고, 합의된 문서대로 작업을 시작합니다."
              }
            ],
            "features": [
              "개발 목표 명확화: 회의 요약과 액션 아이템을 빠르게 정리하도록 우선 기능을 좁혀 정의합니다.",
              "역할·책임 고정: 기획 1 / 프론트 1 / 백엔드 1 / AI 1로 각 산출물과 검수 책임을 연결합니다.",
              "산출물 중심 진행: PRD, 화면 설계안, MVP 개발 범위 문서를 기준으로 구현 범위와 마감(2026-03-31)을 관리합니다."
            ]
          },
          "target_persona": {
            "name": "추가 논의 필요",
            "age": "추가 논의 필요",
            "job_role": "개발 실행 담당 팀",
            "main_activities": "PRD 작성, 화면 설계안 검토, MVP 개발 범위 확정 및 검증",
            "pain_points": [
              "구현 범위와 책임이 문서화되지 않으면 작업 우선순위가 흔들립니다.",
              "산출물 기준이 분산되면 개발과 검토에서 반복 조율이 발생합니다."
            ],
            "needs": [
              "2026-03-31까지 바로 실행할 수 있는 개발 기준 문서가 필요합니다.",
              "역할·산출물·우선순위를 한눈에 볼 수 있는 실행 초안이 필요합니다."
            ]
          }
        }
      },
      {
        "key": "GROUND_RULES",
        "parentKey": "PROJECT_HOME",
        "title": "그라운드룰",
        "content": "1) 개발 시작 전 반드시 개발 범위와 역할별 책임을 문서에 고정한다. 2) 각 산출물(PRD, 화면 설계안, MVP 범위)은 소유자와 승인 기준을 명시해 연결한다. 3) 기술적 세부사항(기술 스택·API 설계 등)은 구현 전 팀 합의로 체크리스트 형태로 정리해 최소 기준을 맞춘다. 4) 마감에 영향을 주는 변경사항은 문서에 즉시 반영하고 관련자 승인 후 진행한다. 5) MVP는 수직 단위로 쪼개어 각 단위에 대한 '완료 기준(검증 기준)'을 정의한 뒤 우선순위에 따라 구현한다."
      }
    ]
  }
}

### 실패 신호
- plan 템플릿과 거의 구분이 안 됨
- 기술 세부를 지어냄
- RAG 문서의 용어를 그대로 복붙함

## 6. Template Validation 테스트

### 목적
- `/ai/template`가 잘못된 입력을 서버에서 막는지 확인한다.

### 케이스 A: 상태 오류
```json
{
  "roomId": 1006,
  "templateType": "plan",
  "currentStatus": "GATHER",
  "content": "",
  "collectedData": {
    "title": "회의록 자동 정리 서비스",
    "goal": "회의 요약과 액션 아이템을 빠르게 정리한다",
    "teamSize": "4명",
    "roles": "기획 1, 프론트 1, 백엔드 1, AI 1",
    "dueDate": "2026-03-31",
    "deliverables": "PRD, 화면 설계안, MVP 개발 범위 문서"
  },
  "recentMessages": [],
  "selectedMessage": null
}
```

### 기대 결과:
- `400`
- `"템플릿 생성은 READY 상태에서만 가능합니다."`



### 케이스 B: collectedData 누락
```json
{
  "roomId": 1007,
  "templateType": "plan",
  "currentStatus": "READY",
  "content": "",
  "collectedData": {
    "title": "회의록 자동 정리 서비스"
  },
  "recentMessages": [],
  "selectedMessage": null
}
```

기대 결과:
- `400`
- `missingFields`에 누락 키 목록 포함

## 테스트 결과 기록 템플릿

### 기록 항목
- 테스트 일시
- 테스트 API
- 입력 케이스
- 실제 응답 요약
- 통과/실패
- 수정 필요 포인트

### 간단 기록 예시
```md
- 일시: 2026-03-14 20:30
- 케이스: @mates 품질 테스트
- 결과: 최근 대화 반영됨, 문장 안 잘림
- 판정: 통과
- 메모: 액션 3개 중 2개만 더 구체적이면 좋음
```
