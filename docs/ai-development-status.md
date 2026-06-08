# AI Development Status

## 현재 구현 상태
- `/ai/chat/`은 일반 채팅, 탐색, 정보 수집, `@mates` 응답을 처리한다.
- `/ai/template/`은 채팅 완료 후 최종 템플릿 생성 전용 API다.
- 템플릿 생성은 `plan`과 `dev` 모드로 분리되어 있다.
- 템플릿 생성 로직은 `app/ai/services/template_generation.py`로 공통화되어 있다.
- 템플릿 응답 스키마는 `app/api/schemas/template.py`로 공용화되어 있다.
- `gather`와 `template generation` LLM 출력은 Pydantic schema로 검증한다.
- RAG는 `phase`뿐 아니라 `topic`, `doc_type` 필터까지 지원한다.
- 프롬프트에는 전문용어를 쉬운 한국어로 풀어쓰라는 규칙이 반영되어 있다.

## 현재 강점
- 채팅과 템플릿 생성 책임이 분리되어 테스트와 운영이 쉬워졌다.
- `BTN_PLAN`과 `BTN_DEV`가 실제로 다른 생성 경로를 탄다.
- `@mates` 응답을 후처리로 자르지 않아서 문장 중간 절단 문제가 줄었다.
- 최근 대화 문맥이 `explore`, `gather`, `template`, `@mates`에 모두 반영된다.
- RAG 문서셋이 phase별 목적에 맞게 더 좁혀진다.

## 남은 주요 개선 과제
1. context summarizer 도입
- 최근 메시지 bullet 나열 대신 `current_goal`, `open_questions`, `latest_decision` 같은 구조화된 상태를 저장해야 한다.

2. 프롬프트/헬퍼 분리
- `nodes.py`에 남아 있는 prompt string, context helper, template helper를 별도 모듈로 분리해야 한다.

3. 운영 관측성 추가
- request id, selected RAG docs, model latency, validation failure 로그를 남겨야 한다.

4. endpoint/request schema 정리
- chat/template request schema도 공용 모델 일부를 분리할 수 있다.

## 현재 리스크
- `nodes.py`가 여전히 크고 역할이 많아서 변경 영향 범위가 넓다.
- RAG 품질은 corpus 설계와 metadata 품질에 계속 크게 의존한다.
- `suggestedQuestions`는 아직 비어 있는 필드라 실제 활용이 없다.

## 다음 추천 작업 순서
1. context summarizer 추가
2. prompt/context builder 분리
3. structured logging 추가
4. retrieval debug endpoint 또는 internal logging 추가
