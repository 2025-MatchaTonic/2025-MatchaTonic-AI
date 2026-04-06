# 2025-MatchaTonic-AI

FastAPI-based AI server for MatchaTonic.

## Template API

- `POST /ai/template/`
  - Rich response for AI clients.
  - Returns `content`, `currentStatus`, and `notionTemplatePayload`.
- `POST /ai/template/spring`
  - Spring compatibility response.
  - Returns flat `{ "projectId": ..., "templates": [...] }`.
- `POST /ai/generate`
  - Alias of `/ai/template/spring`.

## Spring Compatibility

The template API now accepts both native AI payloads and Spring-friendly payloads.

Accepted request fields:

- `roomId` or `projectId`
- `templateType` or `actionType` (`BTN_PLAN`, `BTN_DEV`)
- `currentStatus`
- `content`
- `collectedData`
- `recentMessages`
- `selectedMessage`
- `selectedAnswers`

Compatibility behavior:

- `projectId` is normalized to `roomId`
- `actionType=BTN_PLAN|BTN_DEV` is normalized to `templateType`
- `selectedAnswers` is reused as fallback input for `recentMessages`, `selectedMessage`, and `content`
- sparse `collectedData` is allowed; the generator fills missing sections with defaults

## Manual Replay

Use `scripts/replay_chat_cases.py` to replay saved request payloads against a local or remote server.

Examples:

```powershell
python scripts/replay_chat_cases.py --base-url http://127.0.0.1:8000 --compact
```

```powershell
python scripts/replay_chat_cases.py `
  --baseline-url http://127.0.0.1:8000 `
  --base-url http://127.0.0.1:8001 `
  --compact
```

Sample payloads live under `tests/manual_cases/chat/`.
