from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.api.endpoints.chat import AIChatRequest, process_chat
from eval.langsmith.io import load_jsonl


async def run_payload(name: str, payload: dict) -> None:
    response = await process_chat(AIChatRequest(**payload))
    print(
        json.dumps(
            {
                "case": name,
                "status": response.currentStatus,
                "sufficient": response.isSufficient,
                "content": response.content,
                "suggested": response.suggestedQuestions,
                "keys": list(dict(response.collectedData).keys()),
            },
            ensure_ascii=False,
        )
    )


async def main() -> None:
    for filename in [
        "tc01_subject_only.json",
        "tc02_goal_clear.json",
        "tc12_roles_team_size_conflict.json",
        "tc30_ready_but_insufficient.json",
    ]:
        payload = json.loads(
            Path("tests/manual_cases/chat", filename).read_text(encoding="utf-8")
        )
        await run_payload(filename, payload)

    example_inputs = load_jsonl("eval/datasets/generated/team_project_v1.jsonl")[0][
        "inputs"
    ]
    await run_payload(
        "project_progress_v1_sample",
        {
            "roomId": 9999,
            "content": example_inputs["messages"][0]["content"],
            "actionType": "CHAT",
            "currentStatus": "GATHER",
            "collectedData": {},
            "recentMessages": [],
            "responseMode": example_inputs["response_mode"],
            "backendSchemaName": example_inputs["backend_schema_name"],
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
