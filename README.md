# 2025-MatchaTonic-AI

MatchaTonic 프로젝트의 AI PM 기능을 담당하는 FastAPI 기반 AI 서버입니다. 사용자의 프로젝트 설명과 대화 내용을 바탕으로 프로젝트 정보를 수집하고, Spring 서버와 호환되는 AI 채팅 및 Notion 템플릿 생성 API를 제공합니다.

## 1. 프로젝트 개요

이 저장소는 MatchaTonic 서비스에서 AI 관련 기능을 분리해 구현한 서버입니다. 주요 역할은 프로젝트 기획 대화 지원, 프로젝트 정보 정규화, 템플릿 생성, RAG 기반 문맥 검색, Spring 백엔드와의 API 연동입니다.

채점 및 재현 기준에 맞게 소스 코드, 실행 스크립트, 테스트 케이스, 평가 데이터, API 문서를 함께 포함합니다.

## 2. 주요 기능

- AI 채팅 기반 프로젝트 정보 수집
- 수집된 프로젝트 데이터 정규화 및 누락 항목 판단
- Notion 템플릿 생성 API 제공
- Spring 서버 요청 형식과 호환되는 응답 제공
- Pinecone 기반 RAG 검색 지원
- 수동 회귀 테스트 및 LangSmith 평가 스크립트 제공

## 3. 기술 스택

- Python
- FastAPI
- Uvicorn
- Pydantic
- LangGraph
- LangChain OpenAI
- Pinecone
- Docker

## 4. 폴더 구조

```text
2025-MatchaTonic-AI/
+-- app/
|   +-- api/              # FastAPI 라우터와 요청/응답 스키마
|   +-- ai/               # LLM 프롬프트, 그래프 워크플로, 템플릿 생성 로직
|   +-- core/             # 환경 설정, Spring 연동, 요청 정규화
|   +-- rag/              # Pinecone 벡터 저장소, 검색, 문서 적재
+-- data/                 # RAG 또는 실험에 필요한 샘플 데이터 위치
+-- docs/                 # API 명세, 개발 현황, 품질 테스트 문서
+-- eval/                 # LangSmith 평가 코드와 생성 데이터셋
+-- scripts/              # 수동 재생, 비교, 회귀 테스트 스크립트
+-- tests/                # 자동 테스트와 수동 테스트 입력 케이스
+-- main.py               # FastAPI 애플리케이션 진입점
+-- requirements.txt      # 실행 의존성 목록
+-- Dockerfile            # 컨테이너 실행 설정
+-- .env.example          # 환경 변수 예시
+-- README.md
```

## 5. 설치 방법

Python 3.11 이상 환경을 권장합니다.

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

테스트를 실행하려면 `pytest`가 필요합니다.

```powershell
pip install pytest
```

## 6. 환경 변수 설정

`.env.example`을 참고해 루트 경로에 `.env` 파일을 생성합니다.

```env
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5-mini-2025-08-07
OPENAI_TIMEOUT_SECONDS=120
OPENAI_CHAT_TIMEOUT_SECONDS=15
OPENAI_MAX_RETRIES=1

PINECONE_API_KEY=
PINECONE_INDEX_NAME=ai-pm-knowledge
PINECONE_NAMESPACE=

AI_CORS_ALLOW_ORIGINS=*
REQUIRE_OPENAI_API_KEY=true
REQUIRE_PINECONE_FOR_RAG=false

SPRING_API_BASE_URL=
SPRING_SUMMARY_PATH_TEMPLATE=/api/projects/{project_id}/summary
SPRING_TIMEOUT_SECONDS=10
SPRING_AUTH_BEARER_TOKEN=
SPRING_SUMMARY_SYNC_ENABLED=false
SPRING_SUMMARY_SYNC_STRICT=true
```

필수 값은 실행 목적에 따라 다릅니다.

- `OPENAI_API_KEY`: LLM 기반 채팅 및 템플릿 생성에 필요합니다.
- `PINECONE_API_KEY`: RAG 검색을 Pinecone으로 실행할 때 필요합니다.
- `SPRING_API_BASE_URL`: Spring 서버와 프로젝트 요약을 동기화할 때 사용합니다.

## 7. 실행 방법

로컬 서버 실행:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000
```

정상 실행 확인:

```powershell
curl http://127.0.0.1:8000/
```

Docker 실행:

```powershell
docker build -t matchatonic-ai .
docker run --env-file .env -p 8000:8000 matchatonic-ai
```

## 8. API 요약

### 기본 상태 확인

- `GET /`
  - AI 서버 실행 상태를 확인합니다.

### AI 채팅

- `POST /ai/chat`
  - 프로젝트 기획 대화를 처리하고 현재 수집 상태와 다음 질문을 반환합니다.

### 템플릿 생성

- `POST /ai/template/`
  - Spring 호환 형식의 템플릿 응답을 반환합니다.
- `POST /ai/template/rich`
  - AI 클라이언트용 상세 응답을 반환합니다.
- `POST /ai/template/spring`
  - Spring 서버 연동용 템플릿 응답을 반환합니다.
- `POST /ai/generate`
  - `/ai/template/spring`과 동일한 호환 API입니다.

### 프로젝트 연동

- `POST /ai/project`
  - 프로젝트 관련 AI 처리와 Spring 요약 동기화에 사용합니다.

상세 요청/응답 형식은 `docs/api-spec.md`를 참고합니다.

## 9. Spring 호환 요청 처리

템플릿 API는 AI 서버 고유 요청과 Spring 서버 친화 요청을 모두 받을 수 있습니다.

허용되는 주요 필드:

- `roomId` 또는 `projectId`
- `templateType` 또는 `actionType`
- `currentStatus`
- `content`
- `collectedData`
- `recentMessages`
- `selectedMessage`
- `selectedAnswers`

호환 처리 방식:

- `projectId`는 내부적으로 `roomId`로 정규화됩니다.
- `actionType=BTN_PLAN`, `actionType=BTN_DEV`는 `templateType`으로 변환됩니다.
- `selectedAnswers`는 `recentMessages`, `selectedMessage`, `content`의 보조 입력으로 재사용됩니다.
- `collectedData`가 일부 비어 있어도 기본값으로 보완해 템플릿 생성을 시도합니다.

## 10. 테스트 및 재현 방법

자동 테스트:

```powershell
python -m pytest
```

로컬 서버를 실행한 뒤 수동 채팅 케이스 재생:

```powershell
python scripts/replay_chat_cases.py --base-url http://127.0.0.1:8000 --compact
```

두 서버 응답 비교:

```powershell
python scripts/replay_chat_cases.py `
  --baseline-url http://127.0.0.1:8000 `
  --base-url http://127.0.0.1:8001 `
  --compact
```

로컬 스모크 체크:

```powershell
python -m eval.local_smoke_check
```

LangSmith 평가 데이터셋 생성:

```powershell
python -m eval.langsmith.generate_dataset
```

LangSmith 평가 실행:

```powershell
python -m eval.langsmith.run_eval
```

수동 테스트 입력 예시는 `tests/manual_cases/chat/`에 있습니다. 실행 결과 로그와 비교 산출물은 `tests/manual_cases/out/`에 생성될 수 있으며, 해당 경로는 Git 추적 대상에서 제외합니다.

## 11. 제출 자료 위치

- 소스 코드: `main.py`, `app/`
- API 문서: `docs/api-spec.md`
- 개발 현황 문서: `docs/ai-development-status.md`
- 품질 테스트 계획: `docs/ai-quality-test-plan.md`
- 수동 테스트 스크립트: `scripts/`
- 자동/수동 테스트 케이스: `tests/`
- 평가 데이터 및 평가 코드: `eval/`
- 샘플 또는 프로토타입 데이터 위치: `data/`

## 12. 참고 문서

- `docs/api-spec.md`
- `docs/ai-development-status.md`
- `docs/ai-quality-test-plan.md`
