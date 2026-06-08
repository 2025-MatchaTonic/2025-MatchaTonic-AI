`tests/manual_cases/chat/` contains replay-ready `/ai/chat/` request payloads.

Recommended usage:

```powershell
python scripts/replay_chat_cases.py --base-url http://127.0.0.1:8000 --compact
```

Compare baseline vs target:

```powershell
python scripts/replay_chat_cases.py `
  --baseline-url http://127.0.0.1:8000 `
  --base-url http://127.0.0.1:8001 `
  --compact
```

Each file is a raw request body. Responses are saved under `tests/manual_cases/out/`.
