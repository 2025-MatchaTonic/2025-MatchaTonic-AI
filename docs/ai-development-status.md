# AI 개발 현황 (2026-03-10 기준)

## 1) 현재 구현 범위 요약
- 백엔드 프레임워크: FastAPI
- AI 오케스트레이션: LangGraph (`app/ai/graph/workflow.py`)
- LLM 호출: `ChatOpenAI` + `gpt-5-mini-2025-08-07` (`app/ai/graph/nodes.py`)
- RAG: Pinecone 기반 검색 + phase 필터 fallback (`app/rag/retriever.py`)
- 주요 엔드포인트: `/ai/chat/`, `/ai/project/*`

## 2) 완료된 기능
- 대화 라우팅 분기
  - `@mates` 멘션 전용 응답 노드 분기
  - 버튼 액션(`BTN_NO`, `BTN_YES`, `BTN_PLAN`, `BTN_DEV`) 및 일반 채팅(`CHAT`) 분기
- 정보 수집 플로우
  - `collected_data` 스키마 기반 JSON 수집
  - 기존 데이터 + 신규 데이터 merge 처리
  - 충분성 판단(`is_sufficient`) 기반 단계 전환(`GATHER` -> `READY`)
- 템플릿 생성 플로우
  - JSON 응답 강제 후 노션 템플릿 payload 조립
  - 기본 섹션 + 모델 생성 결과 병합
- @mates 응답 품질 제어
  - 기본 응답 길이 제한(짧은 응답 우선)
  - 상세 요청 키워드 시 제한 완화
  - 후처리 기반 line/char 컷오프
- RAG 기본 동작
  - phase 필터 검색 실패 시 non-filter 검색 재시도
  - prompt 삽입용 snippet 포맷팅

## 3) 진행 중/보완 필요 항목
- DB 영속화 미구현
  - `app/db/models.py`, `app/db/session.py`는 스켈레톤 상태
  - 현재 상태/대화/로그는 FastAPI 내부에서만 처리
- `suggestedQuestions` 미사용
  - `/ai/chat/` 응답에 필드는 있으나 항상 빈 배열 반환
- project API는 정적 가이드 성격
  - `/ai/project/steps`, `/manual`, `/templates`는 고정 응답
  - 프로젝트별 동적 데이터 연동 미구현
- Notion 실제 API 연동 미구현
  - 현재는 payload 생성까지만 구현, 외부 업로드 없음
- 테스트 코드/CI 미구성
  - 자동 검증 파이프라인(단위/통합 테스트) 없음

## 4) 운영 전 체크 포인트
- 필수 환경변수
  - `OPENAI_API_KEY`
  - RAG 사용 시 `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` (옵션: `PINECONE_NAMESPACE`)
- RAG 튜닝 파라미터
  - `RAG_TOP_K`, `RAG_MAX_CONTEXT_CHARS`, `RAG_MAX_DOC_CHARS`, `RAG_PHASE_FILTER_ENABLED`
- CORS
  - 현재 `allow_origins=["*"]`로 전체 허용 상태
- 오류 처리
  - chat endpoint는 예외 시 HTTP 500 + 고정 메시지 반환

## 5) 리스크/정합성 이슈
- `actionType` 주석과 실제 라우팅 상수 일부 불일치
  - 주석에는 `BTN_GO_DEF`가 없지만 workflow 분기에는 사용
- phase 문자열은 enum 검증 없이 자유 문자열 사용
  - 클라이언트-서버 간 오타 발생 시 라우팅 불일치 가능
- `collectedData` 타입은 `Dict[str, str]`
  - 구조화 데이터 확장 시 타입 확장 필요

## 6) 권장 다음 작업
1. Pydantic Enum 도입(`actionType`, `currentStatus`)으로 계약 강제
2. `/ai/chat/` 응답에 `suggestedQuestions` 생성 로직 연결
3. DB 영속화(대화 로그, 단계 상태, template_payload)
4. Notion API export 엔드포인트 또는 비동기 작업 큐 추가
5. 회귀 방지용 API 스키마 테스트 및 라우팅 테스트 추가
