from __future__ import annotations

import asyncio
import time
from typing import Any

from app.api.endpoints.chat import AIChatRequest, process_chat


def _latest_user_content(inputs: dict[str, Any]) -> str:
    messages = inputs.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


async def _ainvoke_chat(inputs: dict[str, Any]) -> dict[str, Any]:
    content = _latest_user_content(inputs)
    request = AIChatRequest(
        roomId=int(inputs.get("room_id") or 1),
        content=content,
        actionType=str(inputs.get("action_type") or "CHAT"),
        currentStatus=str(inputs.get("current_status") or "GATHER"),
        projectName=inputs.get("project_name"),
        collectedData=inputs.get("collected_data") or {},
        recentMessages=inputs.get("recent_messages") or [],
        responseMode=inputs.get("response_mode"),
        backendSchemaName=inputs.get("backend_schema_name"),
    )
    started_at = time.perf_counter()
    response = await process_chat(request)
    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    return {
        "content": response.content,
        "suggestedQuestions": response.suggestedQuestions,
        "currentStatus": response.currentStatus,
        "isSufficient": response.isSufficient,
        "collectedData": dict(response.collectedData),
        "latency_ms": latency_ms,
    }


def invoke_chat(inputs: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(_ainvoke_chat(inputs))
